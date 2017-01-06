/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.nn.dnn

import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator

class LinearSpec  extends FlatSpec with Matchers {

  "Linear batch model" should "converate to correct weight and bias" in {
    val inputN = 20
    val outputN = 10

    val linear = new Linear[Float](inputN, outputN)
    val blasLinear = new com.intel.analytics.bigdl.nn.Linear[Float](inputN, outputN)

    val input = Tensor[Float](5, inputN).rand()
    val gradOutput = Tensor[Float](5, outputN).rand()

    val seed = 100
    RandomGenerator.RNG.setSeed(seed)
    blasLinear.weight.copy(linear.weight)
    blasLinear.bias.copy(linear.bias)

    val output = linear.forward(input)
    val gradInput = linear.backward(input, gradOutput)

    val blasOutput = blasLinear.forward(input)
    val blasGradInput = blasLinear.backward(input, gradOutput)

    println(output)
    println(blasOutput)
    output should be (blasOutput)
    gradInput should be (blasGradInput)
    linear.gradWeight should be (blasLinear.gradWeight)
    linear.gradBias should be (blasLinear.gradBias)
  }
}
