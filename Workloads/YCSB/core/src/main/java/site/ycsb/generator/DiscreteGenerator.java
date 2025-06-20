/**
 * Copyright (c) 2010-2016 Yahoo! Inc., 2017 YCSB contributors. All rights reserved.
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License"); you
 * may not use this file except in compliance with the License. You
 * may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing
 * permissions and limitations under the License. See accompanying
 * LICENSE file.
 */

package site.ycsb.generator;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import static java.util.Objects.requireNonNull;

/**
 * Generates a distribution by choosing from a discrete set of values.
 */
public class DiscreteGenerator extends Generator<String> {
  private static class Pair {
    private double weight;
    private String value;

    Pair(double weight, String value) {
      this.weight = weight;
      this.value = requireNonNull(value);
    }
  }

  private final Collection<Pair> values = new ArrayList<>();
  private String lastvalue;
  private final Random rand; // #RAIN

  public DiscreteGenerator() {
    this.rand = new Random();
    lastvalue = null;
  }

  // #RAIN : 新增构造函数
  public DiscreteGenerator(Random rand) {
    this.rand = rand;
    lastvalue = null;
  }

  /**
   * Generate the next string in the distribution.
   */
  @Override
  public String nextValue() {
    double sum = 0;

    for (Pair p : values) {
      sum += p.weight;
    }

    // double val = ThreadLocalRandom.current().nextDouble();
    double val = rand.nextDouble(); // #RAIN

    for (Pair p : values) {
      double pw = p.weight / sum;
      if (val < pw) {
        return p.value;
      }

      val -= pw;
    }

    throw new AssertionError("oops. should not get here.");

  }

  // #RAIN —— 新增：支持自定义随机源
  public String nextValue(Random rand) {
    double sum = 0;
    for (Pair p : values) {
      sum += p.weight;
    }

    double val = rand.nextDouble();

    for (Pair p : values) {
      double pw = p.weight / sum;
      if (val < pw) {
        lastvalue = p.value;
        return lastvalue;
      }
      val -= pw;
    }

    throw new AssertionError("oops. should not get here.");
  }

  @Override
  public String lastValue() {
    if (lastvalue == null) {
      lastvalue = nextValue();
    }
    return lastvalue;
  }

  public void addValue(double weight, String value) {
    values.add(new Pair(weight, value));
  }
}
