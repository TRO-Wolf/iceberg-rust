// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

// Standalone fixture generator for iceberg-sketches. Emits the exact serialized bytes of theta
// CompactSketches in every mode, plus MurmurHash3 test vectors, all with seed 9001. See README.md
// for the compile/run incantation. NOT part of the Cargo build (a dev oracle, like dev/spark/).

import java.nio.charset.StandardCharsets;

import org.apache.datasketches.Util;
import org.apache.datasketches.hash.MurmurHash3;
import org.apache.datasketches.theta.CompactSketch;
import org.apache.datasketches.theta.UpdateSketch;

public class ThetaFixtureGenerator {
    static String hex(byte[] bytes) {
        StringBuilder builder = new StringBuilder();
        for (byte value : bytes) {
            builder.append(String.format("%02x", value & 0xff));
        }
        return builder.toString();
    }

    static void dumpSketch(String name, CompactSketch sketch) {
        byte[] bytes = sketch.toByteArray();
        System.out.println(name
            + " | bytes=" + bytes.length
            + " | empty=" + sketch.isEmpty()
            + " | ordered=" + sketch.isOrdered()
            + " | retained=" + sketch.getRetainedEntries()
            + " | theta=" + sketch.getThetaLong()
            + " | est=" + sketch.getEstimate()
            + " | hex=" + hex(bytes));
    }

    static void dumpHash(String label, byte[] bytes) {
        long[] hash = MurmurHash3.hash(bytes, Util.DEFAULT_UPDATE_SEED);
        System.out.println(label + " | " + Long.toUnsignedString(hash[0]) + " | " + Long.toUnsignedString(hash[1]));
    }

    public static void main(String[] args) {
        long seed = Util.DEFAULT_UPDATE_SEED;
        System.out.println("DEFAULT_UPDATE_SEED=" + seed);
        System.out.println("computeSeedHash(9001)=" + (Util.computeSeedHash(seed) & 0xffff));

        // ---- serialized-sketch fixtures ----
        UpdateSketch empty = UpdateSketch.builder().setSeed(seed).build();
        dumpSketch("EMPTY", empty.compact());

        UpdateSketch single = UpdateSketch.builder().setSeed(seed).build();
        single.update(1L);
        dumpSketch("SINGLE", single.compact());

        UpdateSketch exact = UpdateSketch.builder().setSeed(seed).build();
        for (long i = 0; i < 10; i++) {
            exact.update(i);
        }
        dumpSketch("EXACT10", exact.compact());

        UpdateSketch est = UpdateSketch.builder().setSeed(seed).setLogNominalEntries(4).build();
        for (long i = 0; i < 1000; i++) {
            est.update(i);
        }
        dumpSketch("EST1000_LGK4", est.compact());

        // Estimation-breadth fixtures (Y1-reviewer): a second seed-independent value set at lgK=8,
        // and the DEFAULT lgK=12 (4096 nominal — what Iceberg's theta_sketch_agg uses) at 100k.
        UpdateSketch est5000 = UpdateSketch.builder().setSeed(seed).setLogNominalEntries(8).build();
        for (long i = 0; i < 5000; i++) {
            est5000.update(1_000_000L + i * 7);
        }
        dumpSketch("EST5000_LGK8", est5000.compact());

        UpdateSketch est100k = UpdateSketch.builder().setSeed(seed).build(); // default lgK=12
        for (long i = 0; i < 100000; i++) {
            est100k.update(i);
        }
        dumpSketch("EST100K_LGK12", est100k.compact());

        // Unordered compact form (the ORDERED flag is clear; same set, hash-table order).
        dumpSketch("EXACT10_UNORDERED", exact.compact(false, null));

        // ---- MurmurHash3 vectors (byte tails 1..=18) ----
        String base = "0123456789abcdefXYZ";
        for (int len = 1; len <= 18; len++) {
            dumpHash("len" + len, base.substring(0, len).getBytes(StandardCharsets.US_ASCII));
        }
        // ---- MurmurHash3 vectors (representative longs) ----
        long[] longs = {0L, 1L, 2L, -1L, 123456789L, Long.MIN_VALUE, Long.MAX_VALUE};
        for (long value : longs) {
            long[] hash = MurmurHash3.hash(new long[]{value}, seed);
            System.out.println("long" + value + " | " + Long.toUnsignedString(hash[0]) + " | " + Long.toUnsignedString(hash[1]));
        }

        // ---- MurmurHash3 multi-block byte vectors (Y1-reviewer hash breadth) ----
        // pattern: byte[i] = (i*31+7) & 0xff, spanning many 16-byte blocks.
        for (int n : new int[]{32, 64, 100, 1000}) {
            byte[] pattern = new byte[n];
            for (int i = 0; i < n; i++) {
                pattern[i] = (byte) ((i * 31 + 7) & 0xff);
            }
            dumpHash("pattern" + n, pattern);
        }
        for (int n : new int[]{1, 8, 16, 17, 32, 33}) {
            dumpHash("zeros" + n, new byte[n]);
        }
        for (int n : new int[]{1, 7, 8, 15, 16, 17, 31, 32}) {
            byte[] ones = new byte[n];
            java.util.Arrays.fill(ones, (byte) 0xff);
            dumpHash("ff" + n, ones);
        }
        // ---- MurmurHash3 multi-element long[] vectors ----
        long[][] arrays = {{1L, 2L}, {1L, 2L, 3L}, {1L, 2L, 3L, 4L}, {0L, 0L, 0L}, {-1L, -1L}};
        for (long[] array : arrays) {
            long[] hash = MurmurHash3.hash(array, seed);
            System.out.println("longarr" + java.util.Arrays.toString(array)
                + " | " + Long.toUnsignedString(hash[0]) + " | " + Long.toUnsignedString(hash[1]));
        }
    }
}
