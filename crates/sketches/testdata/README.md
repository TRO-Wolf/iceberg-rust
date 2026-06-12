<!--
  ~ Licensed to the Apache Software Foundation (ASF) under one
  ~ or more contributor license agreements.  See the NOTICE file
  ~ distributed with this work for additional information
  ~ regarding copyright ownership.  The ASF licenses this file
  ~ to you under the Apache License, Version 2.0 (the
  ~ "License"); you may not use this file except in compliance
  ~ with the License.  You may obtain a copy of the License at
  ~
  ~   http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing,
  ~ software distributed under the License is distributed on an
  ~ "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  ~ KIND, either express or implied.  See the License for the
  ~ specific language governing permissions and limitations
  ~ under the License.
-->

# Theta-sketch test fixtures — provenance

The byte fixtures and hash test vectors pinned in `src/theta.rs` and `src/hash.rs` are
**Java-generated** (not spec-derived). They were produced by the standalone generator
`theta_fixture_generator.java` (in this directory) compiled and run against the Apache DataSketches
Java reference that ships in the local Maven repo:

- `org.apache.datasketches:datasketches-java:3.3.0`
- `org.apache.datasketches:datasketches-memory:2.1.0`

(both already present in `~/.m2`, pulled transitively by `iceberg-spark`'s NDV path).

## Reproduce

```bash
DSJ=~/.m2/repository/org/apache/datasketches/datasketches-java/3.3.0/datasketches-java-3.3.0.jar
DSM=~/.m2/repository/org/apache/datasketches/datasketches-memory/2.1.0/datasketches-memory-2.1.0.jar
# javac requires the file name to match the public class — copy to ThetaFixtureGenerator.java first.
cp theta_fixture_generator.java ThetaFixtureGenerator.java
javac -cp "$DSJ:$DSM" ThetaFixtureGenerator.java
java --add-opens java.base/java.nio=ALL-UNNAMED \
     --add-opens java.base/sun.nio.ch=ALL-UNNAMED \
     -cp ".:$DSJ:$DSM" ThetaFixtureGenerator
```

All hashing uses `Util.DEFAULT_UPDATE_SEED = 9001` — the seed Iceberg's
`apache-datasketches-theta-v1` blob is always built with. `computeSeedHash(9001) = 37836` (0x93cc).

## What each fixture pins

| Fixture (hex const) | Mode | preLongs | Bytes | Notes |
|---|---|---|---|---|
| `EMPTY_HEX` | empty | 1 | 8 | EMPTY flag; no seed hash stored |
| `SINGLE_HEX` | single item | 1 | 16 | SINGLEITEM flag; one 8-byte hash; theta = MAX |
| `EXACT10_HEX` | exact | 2 | 96 | 10 distinct longs; theta = MAX (not stored) |
| `EST1000_LGK4_HEX` | estimation | 3 | 216 | 1000 distinct longs at lgK=4; theta < MAX (stored at byte 16) |
| `EST5000_LGK8_HEX` | estimation | 3 | 3224 | 5000 distinct longs `1_000_000 + i*7` at lgK=8 (a second, seed-independent value set; deep resize + repeated quickselect). retained=400, theta=722749449306535422, est=5104.602733743722 |
| (LGK12/100k contract) | estimation | 3 | 34304 | DEFAULT lgK=12 (4096 nominal — Iceberg's `theta_sketch_agg`), 100k distinct longs. Pinned by retained=4285 / theta=403733047849016500 / est=97891.78614058554 (the contract surface; the full 34 KB blob is not inlined) |

Hash test vectors (`src/hash.rs`) come from the same generator: `MurmurHash3.hash(byte[]/long[], 9001)`
for tail lengths 1..=18, a set of representative longs, multi-block byte inputs (a deterministic
pattern at 32/64/100/1000 bytes, all-zero runs, 0xFF runs across block/tail boundaries), and
multi-element `long[]` arrays (2/3/4 elements). The `EST*` estimate fixtures pin Java's
`getEstimate()` to the **exact f64 bits** (the estimator is `count * (2^63_as_f64 / theta)`, matching
Java `Sketch.estimate(long,int)` — `count / (theta / MAX)` would round one ULP differently).

> NOTE (Y2 risk, flagged): the per-Iceberg-type VALUE serialization fed into the sketch (how a
> date / decimal / timestamp column value becomes the bytes that get hashed) is defined by Iceberg's
> single-value serialization and lives in the `ComputeTableStats` action (Y2), not here. This crate
> pins the hash of given bytes/longs, the build/quickselect, and the compact byte format. The
> mapping from a typed column value to those bytes is **deferred to Y2** and rides MAIN until pinned
> against the Iceberg `theta_sketch_agg` update path.
