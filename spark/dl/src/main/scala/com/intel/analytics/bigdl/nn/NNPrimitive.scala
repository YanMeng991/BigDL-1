/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.nn

import java.nio.ByteBuffer
import java.util

import com.intel.analytics.bigdl.tensor.Tensor

object NNPrimitive {
  def im2colDouble(
    fInput: Tensor[Double], input: Tensor[Double],
    kW: Int, kH: Int,
    dW: Int, dH: Int,
    padW: Int, padH: Int,
    nInputPlane: Int, inputWidth: Int, inputHeight: Int,
    outputWidth: Int, outputHeight: Int): Unit = {

    val inputData = input.storage().array()
    val fInputData = fInput.storage().array()

    var k = 0
    while (k < nInputPlane * kH * kW) {
      val nip = k / (kH * kW)
      val rest = k % (kH * kW)
      val kh = rest / kW
      val kw = rest % kW
      val dstOffset = k * outputHeight * outputWidth + fInput.storageOffset() - 1
      val srcOffset = nip * inputWidth * inputHeight + input.storageOffset() - 1
      if (padW > 0 || padH > 0) {
        var y = 0
        while (y < outputHeight) {
          val iy = y * dH - padH + kh
          if (iy < 0 || iy >= inputHeight) {
            util.Arrays.fill(fInputData, dstOffset + y * outputWidth,
              dstOffset + (y + 1) * outputWidth, 0)
          } else {
            if (dW == 1) {
              val ix = 0 - padW + kw
              val lpad = Math.max(0, padW - kw)
              val rpad = Math.max(0, padW - (kW - kw - 1))
              if (outputWidth - rpad - lpad <= 0) {
                util.Arrays.fill(fInputData, dstOffset + y * outputWidth,
                  dstOffset + (y + 1) * outputWidth, 0)
              } else {
                if (lpad > 0) util.Arrays.fill(fInputData, dstOffset + y * outputWidth,
                  dstOffset + y * outputWidth + lpad, 0)
                System.arraycopy(inputData, srcOffset + iy * inputWidth + ix + lpad, fInputData,
                  dstOffset + y * outputWidth + lpad, outputWidth - rpad - lpad)
                if (rpad > 0) util.Arrays.fill(fInputData, dstOffset + (y + 1) * outputWidth - rpad,
                  dstOffset + (y + 1) * outputWidth, 0)
              }
            } else {
              var x = 0
              while (x < outputWidth) {
                val ix = x * dW - padW + kw
                if (ix < 0 || ix >= inputWidth) {
                  fInputData(dstOffset + y * outputWidth + x) = 0
                } else {
                  fInputData(dstOffset + y * outputWidth + x) =
                    inputData(srcOffset + iy * inputWidth + ix)
                }
                x += 1
              }
            }
          }
          y += 1
        }
      } else {
        var y = 0
        while (y < outputHeight) {
          val iy = y * dH + kh
          val ix = 0 + kw
          if (dW == 1) {
            System.arraycopy(inputData, srcOffset + iy * inputWidth + ix,
              fInputData, dstOffset + y * outputWidth, outputWidth)
          } else {
            var x = 0
            while (x < outputWidth) {
              fInputData(dstOffset + y * outputWidth + x) =
                inputData(srcOffset + iy * inputWidth + ix + x * dW)
              x += 1
            }
          }
          y += 1
        }
      }
      k += 1
    }
  }

  def im2colFloat(
    fInput: Tensor[Float], input: Tensor[Float],
    kW: Int, kH: Int,
    dW: Int, dH: Int,
    padW: Int, padH: Int,
    nInputPlane: Int, inputWidth: Int, inputHeight: Int,
    outputWidth: Int, outputHeight: Int): Unit = {

    val inputData = input.storage().array()
    val fInputData = fInput.storage().array()

    var k = 0
    while (k < nInputPlane * kH * kW) {
      val nip = k / (kH * kW)
      val rest = k % (kH * kW)
      val kh = rest / kW
      val kw = rest % kW
      val dstOffset = k * outputHeight * outputWidth + fInput.storageOffset() - 1
      val srcOffset = nip * inputWidth * inputHeight + input.storageOffset() - 1
      if (padW > 0 || padH > 0) {
        var y = 0
        while (y < outputHeight) {
          val iy = y * dH - padH + kh
          if (iy < 0 || iy >= inputHeight) {
            util.Arrays.fill(fInputData, dstOffset + y * outputWidth,
              dstOffset + (y + 1) * outputWidth, 0)
          } else {
            if (dW == 1) {
              val ix = 0 - padW + kw
              val lpad = Math.max(0, padW - kw)
              val rpad = Math.max(0, padW - (kW - kw - 1))
              if (outputWidth - rpad - lpad <= 0) {
                util.Arrays.fill(fInputData, dstOffset + y * outputWidth,
                  dstOffset + (y + 1) * outputWidth, 0)
              } else {
                if (lpad > 0) util.Arrays.fill(fInputData, dstOffset + y * outputWidth,
                  dstOffset + y * outputWidth + lpad, 0)
                System.arraycopy(inputData, srcOffset + iy * inputWidth + ix + lpad, fInputData,
                  dstOffset + y * outputWidth + lpad, outputWidth - rpad - lpad)
                if (rpad > 0) util.Arrays.fill(fInputData, dstOffset + (y + 1) * outputWidth - rpad,
                  dstOffset + (y + 1) * outputWidth, 0)
              }
            } else {
              var x = 0
              while (x < outputWidth) {
                val ix = x * dW - padW + kw
                if (ix < 0 || ix >= inputWidth) {
                  fInputData(dstOffset + y * outputWidth + x) = 0
                } else {
                  fInputData(dstOffset + y * outputWidth + x) =
                    inputData(srcOffset + iy * inputWidth + ix)
                }
                x += 1
              }
            }
          }
          y += 1
        }
      } else {
        var y = 0
        while (y < outputHeight) {
          val iy = y * dH + kh
          val ix = 0 + kw
          if (dW == 1) {
            System.arraycopy(inputData, srcOffset + iy * inputWidth + ix,
              fInputData, dstOffset + y * outputWidth, outputWidth)
          } else {
            var x = 0
            while (x < outputWidth) {
              fInputData(dstOffset + y * outputWidth + x) =
                inputData(srcOffset + iy * inputWidth + ix + x * dW)
              x += 1
            }
          }
          y += 1
        }
      }
      k += 1
    }
  }

  def col2imDouble(
    fInput: Tensor[Double], input: Tensor[Double],
    kW: Int, kH: Int,
    dW: Int, dH: Int,
    padW: Int, padH: Int,
    nInputPlane: Int,
    inputWidth: Int, inputHeight: Int,
    outputWidth: Int, outputHeight: Int
  ): Unit = {

    val inputData = input.storage().array()
    val fInputData = fInput.storage().array()
    var nPlane = 0
    while (nPlane < nInputPlane) {
      var kh = 0
      while (kh < kH) {
        var kw = 0
        while (kw < kW) {
          val srcOffset = nPlane * (kH * kW * outputHeight * outputWidth) +
            kh * (kW * outputHeight * outputWidth) +
            kw * (outputHeight * outputWidth) + fInput.storageOffset() - 1
          val dstOffset = nPlane * (inputHeight * inputWidth) + input.storageOffset() - 1
          if (padW > 0 || padH > 0) {
            var y = 0
            while (y < outputHeight) {
              val iy = y * dH - padH + kh
              if (iy >= 0 && iy < inputHeight) {
                if (dW == 1) {
                  val ix = 0 - padW + kw
                  val lPad = Math.max(0, padW - kw)
                  val rPad = Math.max(0, padW - (kW - kw - 1))
                  val inputDataOffset = dstOffset + iy * inputWidth + ix + lPad
                  val fInputDataOffset = srcOffset + y * outputWidth + lPad
                  val n = outputWidth - lPad - rPad
                  var i = 0
                  while (i < n) {
                    inputData(inputDataOffset + i) += fInputData(fInputDataOffset + i)
                    i += 1
                  }
                } else {
                  var x = 0
                  while (x < outputWidth) {
                    val ix = x * dW - padW + kw
                    if (ix >= 0 && ix < inputWidth) {
                      inputData(dstOffset + iy * inputWidth + ix) +=
                        fInputData(srcOffset + y * outputWidth + x)
                    }
                    x += 1
                  }
                }
              }
              y += 1
            }
          } else {
            var y = 0
            while (y < outputHeight) {
              val iy = y * dH + kh
              val ix = 0 + kw
              if (dW == 1) {
                var i = 0
                val inputDataOffset = dstOffset + iy * inputWidth + ix
                val fInputDataOffset = srcOffset + y * outputWidth
                while (i < outputWidth) {
                  inputData(inputDataOffset + i) += fInputData(fInputDataOffset + i)
                  i += 1
                }
              } else {
                var x = 0
                while (x < outputWidth) {
                  inputData(dstOffset + iy * inputWidth + ix + x * dW) +=
                    fInputData(srcOffset + y * outputWidth + x)
                  x += 1
                }
              }
              y += 1
            }
          }
          kw += 1
        }
        kh += 1
      }
      nPlane += 1
    }
  }

  def col2imFloat(
    fInput: Tensor[Float], input: Tensor[Float],
    kW: Int, kH: Int,
    dW: Int, dH: Int,
    padW: Int, padH: Int,
    nInputPlane: Int,
    inputWidth: Int, inputHeight: Int,
    outputWidth: Int, outputHeight: Int
  ): Unit = {

    val inputData = input.storage().array()
    val fInputData = fInput.storage().array()
    var nPlane = 0
    while (nPlane < nInputPlane) {
      var kh = 0
      while (kh < kH) {
        var kw = 0
        while (kw < kW) {
          val srcOffset = nPlane * (kH * kW * outputHeight * outputWidth) + kh *
            (kW * outputHeight * outputWidth) +
            kw * (outputHeight * outputWidth) + fInput.storageOffset() - 1
          val dstOffset = nPlane * (inputHeight * inputWidth) + input.storageOffset() - 1
          if (padW > 0 || padH > 0) {
            var y = 0
            while (y < outputHeight) {
              val iy = y * dH - padH + kh
              if (iy >= 0 && iy < inputHeight) {
                if (dW == 1) {
                  val ix = 0 - padW + kw
                  val lPad = Math.max(0, padW - kw)
                  val rPad = Math.max(0, padW - (kW - kw - 1))
                  val inputDataOffset = dstOffset + iy * inputWidth + ix + lPad
                  val fInputDataOffset = srcOffset + y * outputWidth + lPad
                  val n = outputWidth - lPad - rPad
                  var i = 0
                  while (i < n) {
                    inputData(inputDataOffset + i) += fInputData(fInputDataOffset + i)
                    i += 1
                  }
                } else {
                  var x = 0
                  while (x < outputWidth) {
                    val ix = x * dW - padW + kw
                    if (ix >= 0 && ix < inputWidth) {
                      inputData(dstOffset + iy * inputWidth + ix) +=
                        fInputData(srcOffset + y * outputWidth + x)
                    }
                    x += 1
                  }
                }
              }
              y += 1
            }
          } else {
            var y = 0
            while (y < outputHeight) {
              val iy = y * dH + kh
              val ix = 0 + kw
              if (dW == 1) {
                var i = 0
                val inputDataOffset = dstOffset + iy * inputWidth + ix
                val fInputDataOffset = srcOffset + y * outputWidth
                while (i < outputWidth) {
                  inputData(inputDataOffset + i) += fInputData(fInputDataOffset + i)
                  i += 1
                }
              } else {
                var x = 0
                while (x < outputWidth) {
                  inputData(dstOffset + iy * inputWidth + ix + x * dW) +=
                    fInputData(srcOffset + y * outputWidth + x)
                  x += 1
                }
              }
              y += 1
            }
          }
          kw += 1
        }
        kh += 1
      }
      nPlane += 1
    }
  }

  def maxPoolingForwardDouble(
    input: Array[Double], inputOffset: Int,
    output: Array[Double], outputOffset: Int,
    indices: Array[Double], indicesOffset: Int,
    nSlices: Int, iWidth: Int, iHeight: Int, oWidth: Int, oHeight: Int,
    kW: Int, kH: Int, dW: Int, dH: Int, padW: Int, padH: Int) {

    val slices = Range(0, nSlices).iterator
    while (slices.hasNext) {
      val k = slices.next()
      var i = 0
      while (i < oHeight) {
        var j = 0
        while (j < oWidth) {
          // k, i, j output indexers
          var hstart = i * dH - padH
          var wstart = j * dW - padW
          val hend = math.min(hstart + kH, iHeight)
          val wend = math.min(wstart + kW, iWidth)
          hstart = math.max(hstart, 0)
          wstart = math.max(wstart, 0)

          var maxindex = 0  // default is 0
          var maxval = Double.MinValue
          var tcntr = 0
          var y = hstart
          while (y < hend) {
            var x = wstart
            while (x < wend) {
              // k, y, x input indexers
              tcntr = y * iWidth + x
              val value = input(tcntr + inputOffset + k * iWidth * iHeight)
              if (value > maxval) {
                maxval = value
                maxindex = tcntr
              }
              x += 1
            }
            y += 1
          }
          output(outputOffset + k * oWidth * oHeight + i * oWidth + j) = maxval
          indices(indicesOffset + k * oWidth * oHeight + i * oWidth + j) = maxindex + 1
          j += 1
        }
        i += 1
      }
    }
  }

  def maxPoolingForwardFloat(
    input: Array[Float], inputOffset: Int,
    output: Array[Float], outputOffset: Int,
    indices: Array[Float], indicesOffset: Int,
    nSlices: Int, iWidth: Int, iHeight: Int, oWidth: Int, oHeight: Int,
    kW: Int, kH: Int, dW: Int, dH: Int, padW: Int, padH: Int) {

    val slices = Range(0, nSlices).iterator
    while (slices.hasNext) {
      val k = slices.next()
      var i = 0
      while (i < oHeight) {
        var j = 0
        while (j < oWidth) {
          // k, i, j output indexers
          var hstart = i * dH - padH
          var wstart = j * dW - padW
          val hend = math.min(hstart + kH, iHeight)
          val wend = math.min(wstart + kW, iWidth)
          hstart = math.max(hstart, 0)
          wstart = math.max(wstart, 0)

          var maxindex = 0  // default is 0
          var maxval = Float.MinValue
          var tcntr = 0
          var y = hstart
          while (y < hend) {
            var x = wstart
            while (x < wend) {
              // k, y, x input indexers
              tcntr = y * iWidth + x
              val value = input(tcntr + inputOffset + k * iWidth * iHeight)
              if (value > maxval) {
                maxval = value
                maxindex = tcntr
              }
              x += 1
            }
            y += 1
          }
          output(outputOffset + k * oWidth * oHeight + i * oWidth + j) = maxval
          indices(indicesOffset + k * oWidth * oHeight + i * oWidth + j) = maxindex + 1
          j += 1
        }
        i += 1
      }
    }
  }

  def maxPoolingBackwardFloat(
    gradInput: Array[Float], gradInputOffset: Int,
    gradOutput: Array[Float], gradOutputOffset: Int,
    indices: Array[Float], indicesOffset: Int,
    nSlices: Int, iwidth: Int, iheight: Int, owidth: Int, oheight: Int): Unit = {
    val slices = Range(0, nSlices).iterator
    while (slices.hasNext) {
      val k = slices.next()
      var i = 0
      while (i < oheight) {
        var j = 0
        while (j < owidth) {
          val maxp = indices(i * owidth + j + indicesOffset + k * owidth * oheight).toInt - 1
          gradInput(maxp + k * iwidth * iheight + gradInputOffset) +=
            gradOutput(gradOutputOffset + k * owidth * oheight + i * owidth + j)
          j += 1
        }
        i += 1
      }
    }
  }

  def maxPoolingBackwardDouble(
    gradInput: Array[Double], gradInputOffset: Int,
    gradOutput: Array[Double], gradOutputOffset: Int,
    indices: Array[Double], indicesOffset: Int,
    nSlices: Int, iwidth: Int, iheight: Int, owidth: Int, oheight: Int): Unit = {
    val slices = Range(0, nSlices).iterator
    while (slices.hasNext) {
      val k = slices.next()
      var i = 0
      while (i < oheight) {
        var j = 0
        while (j < owidth) {
          val maxp = indices(i * owidth + j + indicesOffset + k * owidth * oheight).toInt - 1
          gradInput(maxp + k * iwidth * iheight + gradInputOffset) += gradOutput(gradOutputOffset
            + k * owidth * oheight + i * owidth + j)
          j += 1
        }
        i += 1
      }
    }
  }

  // For SpatialFullConvolution
  def col2imWithDilationDouble(columns : Tensor[Double], image : Tensor[Double],
    channels : Int, height : Int, width : Int,
    kernelH : Int, kernelW : Int,
    padH : Int, padW : Int,
    strideH : Int, strideW : Int,
    dilationH : Int, dilationW : Int) {

    val dataIm = image.storage().array()
    val dataImOffset = image.storageOffset() - 1
    val dataCol = columns.storage().array()
    val dataColOffset = columns.storageOffset() - 1

    val heightCol = (height + 2 * padH -
      (dilationH * (kernelH - 1) + 1)) / strideH + 1
    val widthCol = (width + 2 * padW -
      (dilationW * (kernelW - 1) + 1)) / strideW + 1
    val channelsCol = channels * kernelH * kernelW
    var cCol = 0
    while (cCol < channelsCol) {
      val wOffset = cCol % kernelW
      val hOffset = (cCol / kernelW) % kernelH
      val cIm = cCol / kernelH / kernelW
      var hCol = 0
      while (hCol < heightCol) {
        var wCol = 0
        while (wCol < widthCol) {
          val hIm = hCol * strideH - padH + hOffset * dilationH
          val wIm = wCol * strideW - padW + wOffset * dilationW
          if (hIm >= 0 && hIm < height && wIm >= 0 && wIm < width) {
            dataIm((cIm * height + hIm) * width + wIm + dataImOffset) +=
              dataCol((cCol * heightCol + hCol) * widthCol + wCol + dataColOffset)
          }
          wCol += 1
        }
        hCol += 1
      }
      cCol += 1
    }
  }

  def col2imWithDilationFloat(columns : Tensor[Float], image : Tensor[Float],
    channels : Int, height : Int, width : Int,
    kernelH : Int, kernelW : Int,
    padH : Int, padW : Int,
    strideH : Int, strideW : Int,
    dilationH : Int, dilationW : Int) {

    val dataIm = image.storage().array()
    val dataImOffset = image.storageOffset() - 1
    val dataCol = columns.storage().array()
    val dataColOffset = columns.storageOffset() - 1

    val heightCol = (height + 2 * padH -
      (dilationH * (kernelH - 1) + 1)) / strideH + 1
    val widthCol = (width + 2 * padW -
      (dilationW * (kernelW - 1) + 1)) / strideW + 1
    val channelsCol = channels * kernelH * kernelW
    var cCol = 0
    while (cCol < channelsCol) {
      val wOffset = cCol % kernelW
      val hOffset = (cCol / kernelW) % kernelH
      val cIm = cCol / kernelH / kernelW
      var hCol = 0
      while (hCol < heightCol) {
        var wCol = 0
        while (wCol < widthCol) {
          val hIm = hCol * strideH - padH + hOffset * dilationH
          val wIm = wCol * strideW - padW + wOffset * dilationW
          if (hIm >= 0 && hIm < height && wIm >= 0 && wIm < width) {
            dataIm((cIm * height + hIm) * width + wIm + dataImOffset) +=
              dataCol((cCol * heightCol + hCol) * widthCol + wCol + dataColOffset)
          }
          wCol += 1
        }
        hCol += 1
      }
      cCol += 1
    }
  }

  def im2colWithDilationDouble(image: Tensor[Double], columns: Tensor[Double],
    channels : Int, height : Int, width : Int,
    kernelH : Int, kernelW : Int,
    padH : Int, padW : Int,
    strideH : Int, strideW : Int,
    dilationH : Int, dilationW : Int): Unit = {

    val dataIm = image.storage().array()
    val dataImOffset = image.storageOffset() - 1
    val dataCol = columns.storage().array()
    val dataColOffset = columns.storageOffset() - 1

    val heightCol = (height + 2 * padH -
      (dilationH * (kernelH - 1) + 1)) / strideH + 1
    val widthCol = (width + 2 * padW -
      (dilationW * (kernelW - 1) + 1)) / strideW + 1
    val channelsCol = channels * kernelH * kernelW
    var cCol = 0
    while (cCol < channelsCol) {
      val wOffset = cCol % kernelW
      val hOffset = (cCol / kernelW) % kernelH
      val cIm = cCol / kernelH / kernelW
      var hCol = 0
      while (hCol < heightCol) {
        var wCol = 0
        while (wCol < widthCol) {
          val hIm = hCol * strideH - padH + hOffset * dilationH
          val wIm = wCol * strideW - padW + wOffset * dilationW
          dataCol((cCol * heightCol + hCol) * widthCol + wCol + dataColOffset) =
            if (hIm >= 0 && wIm >= 0 && hIm < height && wIm < width) {
              dataIm((cIm * height + hIm) * width + wIm + dataImOffset)
            }
            else {
              0
            }
          wCol += 1
        }
        hCol += 1
      }
      cCol += 1
    }
  }

  def im2colWithDilationFloat(image: Tensor[Float], columns: Tensor[Float],
    channels : Int, height : Int, width : Int,
    kernelH : Int, kernelW : Int,
    padH : Int, padW : Int,
    strideH : Int, strideW : Int,
    dilationH : Int, dilationW : Int): Unit = {

    val dataIm = image.storage().array()
    val dataImOffset = image.storageOffset() - 1
    val dataCol = columns.storage().array()
    val dataColOffset = columns.storageOffset() - 1

    val heightCol = (height + 2 * padH -
      (dilationH * (kernelH - 1) + 1)) / strideH + 1
    val widthCol = (width + 2 * padW -
      (dilationW * (kernelW - 1) + 1)) / strideW + 1
    val channelsCol = channels * kernelH * kernelW
    var cCol = 0
    while (cCol < channelsCol) {
      val wOffset = cCol % kernelW
      val hOffset = (cCol / kernelW) % kernelH
      val cIm = cCol / kernelH / kernelW
      var hCol = 0
      while (hCol < heightCol) {
        var wCol = 0
        while (wCol < widthCol) {
          val hIm = hCol * strideH - padH + hOffset * dilationH
          val wIm = wCol * strideW - padW + wOffset * dilationW
          dataCol((cCol * heightCol + hCol) * widthCol + wCol + dataColOffset) =
            if (hIm >= 0 && wIm >= 0 && hIm < height && wIm < width) {
              dataIm((cIm * height + hIm) * width + wIm + dataImOffset)
            }
            else {
              0
            }
          wCol += 1
        }
        hCol += 1
      }
      cCol += 1
    }
  }

  def unfoldedCopyVolDouble(fInput: Tensor[Double], input: Tensor[Double],
    kT: Int, kW: Int, kH: Int,
    dT: Int, dW: Int, dH: Int, pT: Int, pW: Int, pH: Int, nInputPlane: Int,
    inputDepth: Int, inputWidth: Int, inputHeight: Int, outputDepth: Int,
    outputWidth: Int, outputHeight: Int): Unit = {
    val inputData = input.storage().array()
    val fInputData = fInput.storage().array()

    var k = 0
    while (k < nInputPlane * kT * kH * kW) {
      val nip = k / (kT * kH * kW)
      var rest = k % (kT * kH * kW)
      val kt = rest / (kH * kW)
      rest = rest % (kH * kW)
      val kh = rest / kW
      val kw = rest % kW
      var t, x, y, it, ix, iy = 0
      val dstOffset = nip * (kT * kH * kW * outputDepth * outputHeight * outputWidth) +
        kt * (kH * kW * outputDepth * outputHeight * outputWidth) +
        kh * (kW * outputDepth * outputHeight * outputWidth) +
        kw * (outputDepth * outputHeight * outputWidth) + fInput.storageOffset() - 1
      val srcOffset = nip * (inputDepth * inputHeight * inputWidth) + input.storageOffset() - 1

      if (pT > 0 || pH > 0 || pW > 0) {
        t = 0
        while (t < outputDepth) {
          it = t * dT - pT + kt
          var y = 0
          while (y < outputHeight) {
            iy = y * dH - pH + kh
            x = 0
            while (x < outputWidth) {
              ix = x * dW - pW + kw
              if (it < 0 || it >= inputDepth || iy < 0 || iy >= inputHeight ||
                ix < 0 || ix >= inputWidth) {
                fInputData(dstOffset + t * outputHeight * outputWidth + y * outputWidth + x) = 0
              } else {
                fInputData(dstOffset + t * outputHeight * outputWidth + y * outputWidth + x)
                  = inputData(srcOffset + it * inputHeight * inputWidth + iy * inputWidth + ix)
              }
              x += 1
            }
            y += 1
          }
          t += 1
        }
      } else {
        t = 0
        while (t < outputDepth) {
          it = t * dT + kt
          y = 0
          while (y < outputHeight) {
            iy = y * dH + kh
            x = 0
            while (x < outputWidth) {
              ix = x * dW + kw
              fInputData(dstOffset + t * outputHeight * outputWidth + y * outputWidth + x)
                = inputData(srcOffset + it * inputHeight * inputWidth + iy * inputWidth + ix)
              x += 1
            }
            y += 1
          }
          t += 1
        }
      }
      k += 1
    }
  }

  def unfoldedCopyVolFloat(fInput: Tensor[Float], input: Tensor[Float],
    kT: Int, kW: Int, kH: Int,
    dT: Int, dW: Int, dH: Int, pT: Int, pW: Int, pH: Int, nInputPlane: Int,
    inputDepth: Int, inputWidth: Int, inputHeight: Int, outputDepth: Int,
    outputWidth: Int, outputHeight: Int): Unit = {
    val inputData = input.storage().array()
    val fInputData = fInput.storage().array()

    var k = 0
    while (k < nInputPlane * kT * kH * kW) {
      val nip = k / (kT * kH * kW)
      var rest = k % (kT * kH * kW)
      val kt = rest / (kH * kW)
      rest = rest % (kH * kW)
      val kh = rest / kW
      val kw = rest % kW
      var t, x, y, it, ix, iy = 0
      val dstOffset = nip * (kT * kH * kW * outputDepth * outputHeight * outputWidth) +
        kt * (kH * kW * outputDepth * outputHeight * outputWidth) +
        kh * (kW * outputDepth * outputHeight * outputWidth) +
        kw * (outputDepth * outputHeight * outputWidth) + fInput.storageOffset() - 1
      val srcOffset = nip * (inputDepth * inputHeight * inputWidth) + input.storageOffset() - 1

      if (pT > 0 || pH > 0 || pW > 0) {
        t = 0
        while (t < outputDepth) {
          it = t * dT - pT + kt
          var y = 0
          while (y < outputHeight) {
            iy = y * dH - pH + kh
            x = 0
            while (x < outputWidth) {
              ix = x * dW - pW + kw
              if (it < 0 || it >= inputDepth || iy < 0 || iy >= inputHeight ||
                ix < 0 || ix >= inputWidth) {
                fInputData(dstOffset + t * outputHeight * outputWidth + y * outputWidth + x) = 0f
              } else {
                fInputData(dstOffset + t * outputHeight * outputWidth + y * outputWidth + x)
                  = inputData(srcOffset + it * inputHeight * inputWidth + iy * inputWidth + ix)
              }
              x += 1
            }
            y += 1
          }
          t += 1
        }
      } else {
        t = 0
        while (t < outputDepth) {
          it = t * dT + kt
          y = 0
          while (y < outputHeight) {
            iy = y * dH + kh
            x = 0
            while (x < outputWidth) {
              ix = x * dW + kw
              fInputData(dstOffset + t * outputHeight * outputWidth + y * outputWidth + x)
                = inputData(srcOffset + it * inputHeight * inputWidth + iy * inputWidth + ix)
              x += 1
            }
            y += 1
          }
          t += 1
        }
      }
      k += 1
    }
  }

  def unfoldedAccVolDouble(fInput: Tensor[Double], input: Tensor[Double], kT: Int, kW: Int, kH: Int,
    dT: Int, dW: Int, dH: Int, pT: Int, pW: Int, pH: Int, nInputPlane: Int, inputDepth: Int,
    inputWidth: Int, inputHeight: Int,
    outputDepth: Int, outputWidth: Int, outputHeight: Int): Unit = {
    var nip, kt, kw, kh, t, y, x, it, ix, iy = 0
    val inputData = input.storage().array()
    val fInputData = fInput.storage().array()
    nip = 0
    while (nip < nInputPlane) {
      kt = 0
      while (kt < kT) {
        kh = 0
        while (kh < kH) {
          kw = 0
          while (kw < kW) {
            val srcOffset = nip * (kT * kH * kW * outputDepth * outputHeight * outputWidth) +
              kt * (kH * kW * outputDepth * outputHeight * outputWidth) +
              kh * (kW * outputDepth * outputHeight * outputWidth) +
              kw * (outputDepth * outputHeight * outputWidth) + fInput.storageOffset() - 1

            val dstOffset = nip * (inputDepth * inputHeight * inputWidth) +
              input.storageOffset() - 1
            if (pT > 0 || pH > 0 || pW > 0) {
              t = 0
              while (t < outputDepth) {
                it = t * dT - pT + kt
                y = 0
                while (y < outputHeight) {
                  iy = y * dH - pH + kh
                  x = 0
                  while (x < outputWidth) {
                    ix = x * dW - pW + kw
                    if (it < 0 || it >= inputDepth || iy < 0 || iy >= inputHeight ||
                      ix < 0 || ix >= inputWidth) {

                    }
                    else {
                      inputData(dstOffset + it * inputHeight * inputWidth + iy * inputWidth + ix) +=
                          fInputData(srcOffset + t * outputHeight * outputWidth +
                            y * outputWidth + x)
                    }
                    x += 1
                  }
                  y += 1
                }
                t += 1
              }
            }
            else {
              t = 0
              while (t < outputDepth) {
                it = t * dT + kt
                y = 0
                while (y < outputHeight) {
                  iy = y * dH + kh
                  x = 0
                  while (x < outputWidth) {
                    ix = x * dW + kw
                    inputData(dstOffset + it * inputHeight * inputWidth + iy * inputWidth + ix) +=
                      fInputData(srcOffset + t * outputHeight * outputWidth + y * outputWidth + x)
                    x += 1
                  }
                  y += 1
                }
                t += 1
              }
            }
            kw += 1
          }
          kh += 1
        }
        kt += 1
      }
      nip += 1
    }
  }

  def unfoldedAccVolFloat(fInput: Tensor[Float], input: Tensor[Float], kT: Int, kW: Int, kH: Int,
    dT: Int, dW: Int, dH: Int, pT: Int, pW: Int, pH: Int, nInputPlane: Int, inputDepth: Int,
    inputWidth: Int, inputHeight: Int,
    outputDepth: Int, outputWidth: Int, outputHeight: Int): Unit = {
    var nip, kt, kw, kh, t, y, x, it, ix, iy = 0
    val inputData = input.storage().array()
    val fInputData = fInput.storage().array()
    nip = 0
    while (nip < nInputPlane) {
      kt = 0
      while (kt < kT) {
        kh = 0
        while (kh < kH) {
          kw = 0
          while (kw < kW) {
            val srcOffset = nip * (kT * kH * kW * outputDepth * outputHeight * outputWidth) +
              kt * (kH * kW * outputDepth * outputHeight * outputWidth) +
              kh * (kW * outputDepth * outputHeight * outputWidth) +
              kw * (outputDepth * outputHeight * outputWidth) + fInput.storageOffset() - 1

            val dstOffset = nip * (inputDepth * inputHeight * inputWidth) +
              input.storageOffset() - 1
            if (pT > 0 || pH > 0 || pW > 0) {
              t = 0
              while (t < outputDepth) {
                it = t * dT - pT + kt
                y = 0
                while (y < outputHeight) {
                  iy = y * dH - pH + kh
                  x = 0
                  while (x < outputWidth) {
                    ix = x * dW - pW + kw
                    if (it < 0 || it >= inputDepth || iy < 0 || iy >= inputHeight ||
                      ix < 0 || ix >= inputWidth) {

                    }
                    else {
                      inputData(dstOffset + it * inputHeight * inputWidth + iy * inputWidth + ix) +=
                        fInputData(srcOffset + t * outputHeight * outputWidth +
                          y * outputWidth + x)
                    }
                    x += 1
                  }
                  y += 1
                }
                t += 1
              }
            }
            else {
              t = 0
              while (t < outputDepth) {
                it = t * dT + kt
                y = 0
                while (y < outputHeight) {
                  iy = y * dH + kh
                  x = 0
                  while (x < outputWidth) {
                    ix = x * dW + kw
                    inputData(dstOffset + it * inputHeight * inputWidth + iy * inputWidth + ix) +=
                      fInputData(srcOffset + t * outputHeight * outputWidth + y * outputWidth + x)
                    x += 1
                  }
                  y += 1
                }
                t += 1
              }
            }
            kw += 1
          }
          kh += 1
        }
        kt += 1
      }
      nip += 1
    }
  }
}

object Quantize{
  def findMax(src: Array[Float], start: Int, end: Int): Float = {
    src.slice(start, end).max
  }

  def findMin(src: Array[Float], start: Int, end: Int): Float = {
    src.slice(start, end).min
  }

  def quantize(value: Float, max: Float, min: Float): Byte = {
    Math.round(1.0 * value / Math.max(max, min) * Byte.MaxValue).toByte
  }

  def dequantize(byte: Byte, max: Float, min: Float): Float = {
    byte.toFloat / Byte.MaxValue * Math.max(max, min)
  }

  def quantize(src: Array[Float], start: Int, end: Int, dst: ByteBuffer,
    dstOffset: Int): (Float, Float) = {
    val max = Math.abs(findMax(src, start, end))
    val min = Math.abs(findMin(src, start, end))

    for (i <- 0 until end - start) {
      dst.put(dstOffset + i, quantize(src(start + i), max, min))
    }

    (max, min)
  }

  def dequantize(src: Array[Float], start: Int, end: Int, dst: ByteBuffer, dstOffset: Int,
    max: Float, min: Float): Unit = {
    require(src.length >= end, s"you write too much elements")

    for (i <- 0 until end - start) {
      src(start + i) = dequantize(dst.get(dstOffset + i), max, min)
    }
  }

  def quantize(src: Array[Float], start: Int, end: Int, dst: ByteBuffer, dstOffset: Int,
    size: Array[Int]): (Array[Float], Array[Float]) = {
    require(size.length == 2, s"only support 2-dim matrix")
    require(size.product == (end - start), s"number of elements does not match")

    val height = size(0)
    val width = size(1)

    val max = new Array[Float](height)
    val min = new Array[Float](height)

    for (i <- 0 until height) {
      val maxAndMin = quantize(src, start + i * width, start + (i + 1) * width, dst,
        dstOffset + i * width)

      max(i) = maxAndMin._1
      min(i) = maxAndMin._2
    }

    (max, min)
  }

  def dequantize(data: Array[Float], start: Int, end: Int, quantizedData: ByteBuffer, offset: Int,
    max: Array[Float], min: Array[Float], size: Array[Int]): Unit = {
    require(max.length == min.length, s"the number of max doesn't match with the number of min")
    require(size.length == 2, s"only support 2-dim matrix")
    require(max.length == size(0),
      s"the number of max(${max.length}) doesn't match the size(${size(1)})")

    require(size.product == (end - start), s"number of elements does not match")

    val height = size(0)
    val width = size(1)

    for (i <- 0 until height) {
      dequantize(data, start + i * width, start + (i + 1) * width,
        quantizedData, offset + i * width, max(i), min(i))
    }
  }

  private[bigdl] def get2Dim(shape: Array[Int]): Array[Int] = {
    require(shape.length > 1, s"error size dimension, which must be great than 1")
    val first = shape(0)
    val last = shape.slice(1, shape.length).product
    Array(first, last)
  }

  def quantize(input: Tensor[Float], buffer: ByteBuffer,
    offset: Int): (Array[Float], Array[Float]) = {
    val length = input.nElement()

    input.dim() match {
      case 1 =>
        val (max, min) = quantize(input.storage().array(), input.storageOffset() - 1,
          length, buffer, offset)
        (Array(max), Array(min))
      case x if x > 1 =>
        val size = get2Dim(input.size())
        val start = input.storageOffset() - 1
        val end = start + length
        val (max, min) = quantize(input.storage().array(), start, end, buffer, offset, size)
        (max, min)
      case _ => throw new UnsupportedOperationException(s"unsupported input")
    }
  }

  def dequantize(input: Tensor[Float], buffer: ByteBuffer, offset: Int, max: Array[Float],
    min: Array[Float]): Unit = {
    val start = input.storageOffset() - 1
    val end = start + input.nElement()

    input.dim() match {
      case 1 => dequantize(input.storage().array(), start, end, buffer,
        offset, max(0), min(0))
      case x if x > 1 =>
        dequantize(input.storage().array(), start, end, buffer,
          offset, max, min, get2Dim(input.size()))
      case _ => throw new UnsupportedOperationException {
        s"unsupported input dim ${input.dim()}"
      }
    }
  }

  def loss(before: Array[Float], after: Array[Float], start: Int, end: Int): Double = {
    var lossValue = 0.0

    for (i <- start until end) {
      lossValue += (before(i) - after(i))
    }

    lossValue
  }

  def testQuantizeMatrix(): Unit = {
    val src = Array(0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
      0.6f, 0.4f, 0.3f, 0.2f, 0.1f)
    val dst = ByteBuffer.allocate(src.length)
    dst.clear()

    val (max, min) = quantize(src, 0, src.length, dst, 0, Array(2, 5))

    for (i <- src.indices) {
      println(dst.get(i))
    }

    val before = src.clone()
    for (i <- src.indices) {
      src(i) = 0f
    }

    dequantize(src, 0, src.length, dst, 0, max, min, Array(2, 5))
    for (i <- src.indices) {
      println(src(i))
    }
    val after = src.clone()

    println(loss(before, after, 0, src.length))
  }

  def testArray(): Unit = {
    val src = Array[Float](0.6f, 0.4f, -0.3f, 0.2f, 0.1f)

    val dst = ByteBuffer.allocate(src.length)
    dst.clear()

    val (max, min) = quantize(src, 0, src.length, dst, 0)
    println(dst)

    for (i <- src.indices) {
      println(dst.get(i))
    }

    val before = src.clone()
    for (i <- src.indices) {
      src(i) = 0f
    }

    dequantize(src, 0, src.length, dst, 0, max, min)
    for (i <- src.indices) {
      println(src(i))
    }

    val after = src.clone()

    println(loss(before, after, 0, src.length))
  }

  def main(args: Array[String]): Unit = {
    testArray()
    //    testQuantizeMatrix()
  }
}
