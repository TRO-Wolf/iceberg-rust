/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

// IMPORTANT: this file lives in `package org.apache.iceberg` ON PURPOSE — that is the only way to reach
// the package-private `@VisibleForTesting SchemaUpdate(Schema schema, int lastColumnId)` constructor that
// drives the UpdateSchema state machine without a live TableOperations / catalog. This class is a
// TEST-ONLY ORACLE (a dev tool, like dev/spark/); it is not part of the shipped Rust library.
package org.apache.iceberg;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Function;
import org.apache.iceberg.expressions.Literal;
import org.apache.iceberg.types.Types;

/**
 * Java reference oracle for the UpdateSchema interop pilot.
 *
 * <p>Two modes, selected by the first program argument:
 *
 * <ul>
 *   <li><b>generate</b> — for each named scenario, build a base {@link Schema}, wrap it in a
 *       format-version-2 {@link TableMetadata}, apply the scenario's UpdateSchema op-sequence via the
 *       package-private {@code @VisibleForTesting SchemaUpdate(Schema, int)} constructor to get the
 *       evolved {@link Schema}, build the evolved {@link TableMetadata} via
 *       {@code buildFrom(base).setCurrentSchema(evolved, evolvedLastColumnId).build()}, and write
 *       {@code base.metadata.json} + {@code java_evolved.metadata.json} (via {@link
 *       TableMetadataParser#toJson}) into the scenario's testdata directory.
 *   <li><b>verify</b> — read {@code rust_evolved.metadata.json} from each scenario directory via
 *       {@link TableMetadataParser#read}, assert Java parses it without error AND its current schema is
 *       structurally equal (recursive field id / name / type / required / doc / default, plus
 *       current-schema-id and last-column-id) to Java's own {@code java_evolved}. Prints PASS/FAIL per
 *       scenario; exits non-zero on any FAIL.
 * </ul>
 */
public final class InteropOracle {
  private static final String LOCATION = "s3://interop-bucket/update_schema";

  private InteropOracle() {}

  public static void main(String[] args) throws IOException {
    if (args.length < 1) {
      System.err.println("usage: InteropOracle <generate|verify>");
      System.exit(2);
      return;
    }
    String fixturesDirProperty = System.getProperty("interop.fixtures.dir");
    if (fixturesDirProperty == null || fixturesDirProperty.isEmpty()) {
      System.err.println("system property interop.fixtures.dir must be set");
      System.exit(2);
      return;
    }
    Path fixturesDir = Paths.get(fixturesDirProperty).toAbsolutePath().normalize();

    String mode = args[0];
    switch (mode) {
      case "generate":
        generate(fixturesDir);
        break;
      case "verify":
        verify(fixturesDir);
        break;
      default:
        System.err.println("unknown mode: " + mode + " (expected generate|verify)");
        System.exit(2);
    }
  }

  // ===========================================================================================
  // Scenario registry — each entry is (base schema, base lastColumnId, op-sequence). The op-sequence
  // is applied via `new SchemaUpdate(baseSchema, baseLastColumnId)` (the @VisibleForTesting ctor). The
  // Rust test mirrors EACH of these op-sequences exactly against the same `base.metadata.json`.
  // ===========================================================================================

  private static Map<String, Scenario> scenarios() {
    Map<String, Scenario> scenarios = new LinkedHashMap<>();

    // add_top_level_columns — append two optional and one required-with-default top-level columns.
    // The required-with-default add needs an initial default, which is V3-only in Java.
    scenarios.put(
        "add_top_level_columns",
        Scenario.v3(
            new Schema(
                Types.NestedField.required(1, "id", Types.LongType.get()),
                Types.NestedField.optional(2, "data", Types.StringType.get())),
            2,
            update ->
                update
                    .addColumn("count", Types.IntegerType.get())
                    .addColumn("note", Types.StringType.get(), "a free-text note")
                    .addRequiredColumn(
                        "category", Types.StringType.get(), Literal.of("uncategorized"))));

    // add_nested_struct_and_map — THE level-order fresh-field-id case. Adding a map<struct,struct> to a
    // 1-column schema must assign key=3, value=4, then the key struct's fields 5..8, then the value
    // struct's fields 9..10 (Java AssignFreshIds / CustomOrderSchemaVisitor level order). Also add a
    // nested struct to pin the struct path. The incoming ids are deliberately scrambled to prove they
    // are reassigned.
    scenarios.put(
        "add_nested_struct_and_map",
        Scenario.v2(
            new Schema(Types.NestedField.required(1, "id", Types.IntegerType.get())),
            1,
            update ->
                update.addColumn(
                    "locations",
                    Types.MapType.ofOptional(
                        11,
                        12,
                        Types.StructType.of(
                            Types.NestedField.required(20, "address", Types.StringType.get()),
                            Types.NestedField.required(21, "city", Types.StringType.get()),
                            Types.NestedField.required(22, "state", Types.StringType.get()),
                            Types.NestedField.required(23, "zip", Types.IntegerType.get())),
                        Types.StructType.of(
                            Types.NestedField.required(30, "lat", Types.IntegerType.get()),
                            Types.NestedField.optional(31, "long", Types.IntegerType.get()))))));

    // rename_and_move — rename a column and reorder columns (move first + move after). Java resolves
    // move targets by their ORIGINAL name (renames are tracked in `updates`, not in name resolution),
    // so the moved column is referenced as `email`, not `email_address`. This pins that the rename is
    // applied to the field-id-stable record while the reorder operates on the original identity.
    scenarios.put(
        "rename_and_move",
        Scenario.v2(
            new Schema(
                Types.NestedField.required(1, "id", Types.LongType.get()),
                Types.NestedField.optional(2, "first_name", Types.StringType.get()),
                Types.NestedField.optional(3, "last_name", Types.StringType.get()),
                Types.NestedField.optional(4, "email", Types.StringType.get())),
            4,
            update ->
                update
                    .renameColumn("email", "email_address")
                    .moveFirst("email")
                    .moveAfter("id", "first_name")));

    // update_type_promotion — int->long, float->double, decimal(9,2)->decimal(18,2) widen.
    scenarios.put(
        "update_type_promotion",
        Scenario.v2(
            new Schema(
                Types.NestedField.required(1, "id", Types.IntegerType.get()),
                Types.NestedField.optional(2, "measure", Types.FloatType.get()),
                Types.NestedField.optional(3, "amount", Types.DecimalType.of(9, 2))),
            3,
            update ->
                update
                    .updateColumn("id", Types.LongType.get())
                    .updateColumn("measure", Types.DoubleType.get())
                    .updateColumn("amount", Types.DecimalType.of(18, 2))));

    // make_optional_and_delete — relax a required column to optional, and delete another column.
    scenarios.put(
        "make_optional_and_delete",
        Scenario.v2(
            new Schema(
                Types.NestedField.required(1, "id", Types.LongType.get()),
                Types.NestedField.required(2, "name", Types.StringType.get()),
                Types.NestedField.optional(3, "legacy", Types.StringType.get())),
            3,
            update -> update.makeColumnOptional("name").deleteColumn("legacy")));

    // set_identifier_fields — promote a required field to the identifier-field set.
    scenarios.put(
        "set_identifier_fields",
        Scenario.v2(
            new Schema(
                Types.NestedField.required(1, "id", Types.LongType.get()),
                Types.NestedField.required(2, "tenant", Types.StringType.get()),
                Types.NestedField.optional(3, "data", Types.StringType.get())),
            3,
            update -> update.setIdentifierFields("id", "tenant")));

    // add_required_with_default_and_update_default — add a required column WITH a default (legal without
    // allowIncompatibleChanges because the default backfills existing rows), then change its write
    // default via updateColumnDefault (sets only the write default, leaving the initial default).
    scenarios.put(
        "add_required_with_default_and_update_default",
        Scenario.v3(
            new Schema(Types.NestedField.required(1, "id", Types.LongType.get())),
            1,
            update ->
                update
                    .addRequiredColumn("status", Types.StringType.get(), Literal.of("active"))
                    .updateColumnDefault("status", Literal.of("pending"))));

    return scenarios;
  }

  /**
   * A base schema + last column id + the UpdateSchema op-sequence to apply, plus the base format
   * version. Column initial defaults are a V3-only feature in Java iceberg-core (a non-null initial
   * default is rejected on V2 metadata), so default-bearing scenarios use format version 3; all others
   * use format version 2. {@code baseLastColumnId} is informational — the actual value flows through the
   * built {@link TableMetadata#lastColumnId()}.
   */
  private static final class Scenario {
    final Schema baseSchema;
    final int baseLastColumnId;
    final int formatVersion;
    final Function<UpdateSchema, UpdateSchema> ops;

    Scenario(
        Schema baseSchema,
        int baseLastColumnId,
        int formatVersion,
        Function<UpdateSchema, UpdateSchema> ops) {
      this.baseSchema = baseSchema;
      this.baseLastColumnId = baseLastColumnId;
      this.formatVersion = formatVersion;
      this.ops = ops;
    }

    /** Convenience for the common format-version-2 case. */
    static Scenario v2(
        Schema baseSchema, int baseLastColumnId, Function<UpdateSchema, UpdateSchema> ops) {
      return new Scenario(baseSchema, baseLastColumnId, 2, ops);
    }

    /** Format-version-3 scenario (needed for column initial defaults). */
    static Scenario v3(
        Schema baseSchema, int baseLastColumnId, Function<UpdateSchema, UpdateSchema> ops) {
      return new Scenario(baseSchema, baseLastColumnId, 3, ops);
    }
  }

  // ===========================================================================================
  // generate
  // ===========================================================================================

  private static void generate(Path fixturesDir) throws IOException {
    Map<String, Scenario> scenarios = scenarios();
    for (Map.Entry<String, Scenario> entry : scenarios.entrySet()) {
      String name = entry.getKey();
      Scenario scenario = entry.getValue();

      // Base metadata at the scenario's format version (V3 for default-bearing scenarios, else V2).
      Map<String, String> props = new LinkedHashMap<>();
      props.put(TableProperties.FORMAT_VERSION, Integer.toString(scenario.formatVersion));
      TableMetadata base =
          TableMetadata.newTableMetadata(
              scenario.baseSchema,
              PartitionSpec.unpartitioned(),
              SortOrder.unsorted(),
              LOCATION + "/" + name,
              props);

      // Apply the op-sequence via the @VisibleForTesting ctor → evolved schema.
      SchemaUpdate update = new SchemaUpdate(base.schema(), base.lastColumnId());
      Schema evolved = scenario.ops.apply(update).apply();

      // Evolved metadata: rebuild from base, set the new current schema. The new last-column-id must
      // never DECREASE below the base's (a delete lowers `highestFieldId()` but ids are never reused),
      // so pass max(base.lastColumnId, evolved.highestFieldId) — exactly what Java's addSchema does
      // internally.
      int evolvedLastColumnId = Math.max(base.lastColumnId(), evolved.highestFieldId());
      TableMetadata javaEvolved =
          TableMetadata.buildFrom(base)
              .setCurrentSchema(evolved, evolvedLastColumnId)
              .build();

      Path scenarioDir = fixturesDir.resolve(name);
      Files.createDirectories(scenarioDir);
      writeJson(scenarioDir.resolve("base.metadata.json"), TableMetadataParser.toJson(base));
      writeJson(
          scenarioDir.resolve("java_evolved.metadata.json"), TableMetadataParser.toJson(javaEvolved));
      System.out.println("generated: " + name);
    }
    System.out.println("generate: wrote " + scenarios.size() + " scenarios to " + fixturesDir);
  }

  // ===========================================================================================
  // verify — read rust_evolved.metadata.json, assert Java accepts it and its current schema matches.
  // ===========================================================================================

  private static void verify(Path fixturesDir) throws IOException {
    Map<String, Scenario> scenarios = scenarios();
    int failures = 0;
    for (Map.Entry<String, Scenario> entry : scenarios.entrySet()) {
      String name = entry.getKey();
      Scenario scenario = entry.getValue();
      Path scenarioDir = fixturesDir.resolve(name);
      Path rustEvolvedPath = scenarioDir.resolve("rust_evolved.metadata.json");

      if (!Files.exists(rustEvolvedPath)) {
        System.out.println("FAIL " + name + ": missing rust_evolved.metadata.json (run the Rust gen)");
        failures++;
        continue;
      }

      // Recompute Java's evolved schema for comparison (the same op-sequence as generate).
      Path basePath = scenarioDir.resolve("base.metadata.json");
      TableMetadata base = TableMetadataParser.fromJson(basePath.toString(), readString(basePath));
      SchemaUpdate update = new SchemaUpdate(base.schema(), base.lastColumnId());
      Schema javaEvolvedSchema = scenario.ops.apply(update).apply();

      TableMetadata rustEvolved;
      try {
        rustEvolved =
            TableMetadataParser.fromJson(rustEvolvedPath.toString(), readString(rustEvolvedPath));
      } catch (RuntimeException parseError) {
        System.out.println("FAIL " + name + ": Java could not parse rust_evolved: " + parseError);
        failures++;
        continue;
      }

      Schema rustSchema = rustEvolved.schema();
      String mismatch = structuralMismatch(javaEvolvedSchema, rustSchema);
      if (mismatch != null) {
        System.out.println("FAIL " + name + ": " + mismatch);
        failures++;
        continue;
      }
      // Also assert last-column-id agrees. It is max(base lastColumnId, evolved highestFieldId) — a
      // delete lowers highestFieldId but never lowers lastColumnId (ids are never reused).
      int expectedLastColumnId =
          Math.max(base.lastColumnId(), javaEvolvedSchema.highestFieldId());
      if (expectedLastColumnId != rustEvolved.lastColumnId()) {
        System.out.println(
            "FAIL "
                + name
                + ": last-column-id mismatch: java="
                + expectedLastColumnId
                + " rust="
                + rustEvolved.lastColumnId());
        failures++;
        continue;
      }
      System.out.println("PASS " + name);
    }

    System.out.println(
        "verify: " + (scenarios.size() - failures) + "/" + scenarios.size() + " scenarios passed");
    if (failures > 0) {
      System.exit(1);
    }
  }

  /**
   * Compare two schemas structurally (recursive field id / name / type / required / doc / default plus
   * identifier-field ids). Returns null when equal, or a human-readable mismatch message.
   *
   * <p>{@link Schema#sameSchema} compares {@code asStruct()} (which includes field id, name, type,
   * required, and doc recursively) plus identifier-field ids — exactly the recursive structural
   * equality we need. Default values are part of {@code NestedField.equals}, so they are covered too.
   */
  private static String structuralMismatch(Schema expected, Schema actual) {
    if (!expected.asStruct().equals(actual.asStruct())) {
      return "schema struct mismatch:\n  java= " + expected.asStruct() + "\n  rust= " + actual.asStruct();
    }
    if (!expected.identifierFieldIds().equals(actual.identifierFieldIds())) {
      return "identifier-field-id mismatch: java="
          + expected.identifierFieldIds()
          + " rust="
          + actual.identifierFieldIds();
    }
    return null;
  }

  // ===========================================================================================
  // IO helpers
  // ===========================================================================================

  private static void writeJson(Path path, String json) throws IOException {
    Files.write(path, json.getBytes(StandardCharsets.UTF_8));
  }

  private static String readString(Path path) throws IOException {
    return new String(Files.readAllBytes(path), StandardCharsets.UTF_8);
  }
}
