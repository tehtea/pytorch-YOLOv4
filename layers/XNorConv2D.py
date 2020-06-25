import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch._six import container_abcs
from itertools import repeat
import math
from scipy import signal
import random
import numpy as np


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_pair = _ntuple(2)


def convert_bool_to_bin_str(bool):
    if bool:
        return '1'
    return '0'


class XNorConv2D(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, bias=False, debug_weight=None):
        """
        Does not support dilation and grouping for now.

        Adding a debug weights parameter so we can add our own weights for easy debugging
        """
        kernel_size = _pair(kernel_size)
        padding = _pair(padding)
        stride = _pair(stride)

        super(XNorConv2D, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.weight = Parameter(torch.Tensor(
            output_channels, input_channels, *kernel_size))
        self.stride = stride
        self.reset_parameters(debug_weight)
        if bias:
            self.bias = Parameter(torch.Tensor(output_channels))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self, debug_weight):
        """
        debug_weight: torch.Tensor instance
        TODO: There is definitely a better way to initialize weights for XNorConv. And this is not the original
            method used by PyTorch's conv2d module.
        """
        if debug_weight is None:
            stdv = 1.0 / math.sqrt(torch.numel(self.weight))
            self.weight.data.uniform_(-stdv, +stdv)
        else:
            if isinstance(debug_weight, torch.Tensor):
                if (debug_weight.size()[-2:] != self.kernel_size):
                    raise Exception(
                        "Debug weight size {} is not equal to provided kernel size {}".format(debug_weight.size()[-2:],
                                                                                              self.kernel_size))
                self.weight.data = debug_weight
            else:
                raise Exception("Debug weight provided is not a Tensor")

    # use tensors instead of vectors for these inputs for faster performance
    # consider bitpacking (or GEMM)
    # for minimum change, just use torch tensors so that GEMM can be used and for better cache locality
    # mkl is much faster than cuda
    # for ternary network, no library support so slower
    # bitpacking etc has to be done ourselves so optimization must be done by oneself too (using openMP and openBLAS) (a lot more work)

    # for training just use normal conv2d forward function (can see github other repos for this, the idea is to use nn.Functional)
    # quantize weights to 64 bits
    # for inference use quantized weights (e.g. ternary_convolute)
    def _bitcount(self, flattened_input, flattened_weight):
        # a mask wouldn't be needed if flattened input is all zeros / unnecessary part are all zeros
        output = bin((~(int(flattened_input, 2)) ^ int(flattened_weight, 2)) & int('1' * len(flattened_input), 2))
        output = output[(output.find('b') + 1):].ljust(len(flattened_input))
        count = 0
        # see if there are popcount functions (probably have in python)
        for i in output:
            if i == '1':
                count += 1
        bitcount_result = 2 * count - len(flattened_input)
        return bitcount_result

    # this whole forward layer will be a C++ extension function
    def inference_forward(self, input, weight, paddingHeight, paddingWidth, strideY, strideX, input_channels,
                          output_channels, kernel_size):
        input = F.pad(input, (paddingHeight, paddingHeight, paddingWidth, paddingWidth), "constant", 1)
        input_channels, input_height, input_width = input.size()
        weight_alpha = torch.zeros([output_channels], dtype=torch.float32)
        for i in range(output_channels):
            weight_alpha[i] = torch.mean(torch.abs(weight[i])).type(torch.float32)

        filter_height, filter_width = kernel_size
        bin_input = input.data >= 0
        bin_weight = weight.data >= 0
        output_height = int((input_height - (filter_height - 1)) / strideY)
        output_width = int((input_width - (filter_width - 1)) / strideX)

        input_K = torch.zeros([output_channels, output_height, output_width],
                              dtype=torch.float32)  # scaling factor for input
        output_map = torch.zeros([output_channels, output_height, output_width], dtype=torch.float32)

        input_patch_size = filter_height * filter_width
        for k in range(output_channels):
            for l in range(input_channels):
                for i in range(0, output_height):
                    for j in range(0, output_width):
                        input_patch = input[l, i * strideY:i * strideY + filter_height,
                                      j * strideX:j * strideX + filter_width]
                        input_patch_binarized = input_patch.data >= 0
                        input_patch_flattened = input_patch_binarized.detach().numpy().flatten()  # flatten the input
                        input_patch_flattened = ''.join(map(convert_bool_to_bin_str,
                                                            input_patch_flattened))  # convert flattened input patch to string

                        flattened_weight = ''.join(
                            map(convert_bool_to_bin_str, bin_weight[k].detach().numpy().flatten()))
                        input_K[k, i, j] += torch.sum(torch.abs(input_patch)) / input_patch_size
                        output_map[k, i, j] += self._bitcount(input_patch_flattened, flattened_weight)
        input_K /= input_channels
        output_map = torch.mul(output_map, input_K)
        for i in range(output_channels):
            output_map[i] *= weight_alpha[i]
        return output_map

    def pad(self, input, padding_height, padding_width):
        input = F.pad(input, (padding_height, padding_height, padding_width, padding_width), "constant", 1)
        return input

    def bin_activ(self, input, weight, kernel_size, output_channels, stride):
        filter_height, filter_width = kernel_size
        stride_x, stride_y = stride
        input_batch_size, input_channels, input_height, input_width = input.size()

        weight_alpha = torch.zeros([output_channels], dtype=torch.float32).to(self.device)
        for output_height_index in range(output_channels):
            weight_alpha[output_height_index] = torch.mean(torch.abs(weight[output_height_index])).type(torch.float32)
        abs_input = torch.abs(input)
        input_K_intermediate = F.avg_pool2d(abs_input, kernel_size, stride=stride, padding=0)
        input_K_one_channel = torch.mean(input_K_intermediate, (1))
        input_K = input_K_one_channel.repeat(1, output_channels, 1, 1)

        input_binarized = input.data >= 0
        input_binarized = input_binarized.float() * 2 - 1
        return {
            "input_K": input_K.to(device=self.device),
            "weight_alpha": weight_alpha.to(device=self.device),
            "input_binarized": input_binarized.to(device=self.device)
        }

    def forward(self, input):
        self.device = input.device
        input = self.pad(input, *self.padding)
        input_binarized = input.data >= 0
        input_binarized = input_binarized.float() * 2 - 1
        input_K = F.avg_pool2d(torch.mean(abs(input), 1), self.kernel_size, stride=self.stride, padding=0)
        input_K = input_K.repeat(1, self.output_channels, 1, 1)
        weight_alpha = self.weight.mean([1, 2, 3]).unsqueeze(1).unsqueeze(2).repeat(input.size()[0], 1, 1, 1)
        if self.training:
            return F.conv2d(input_binarized, self.weight, stride=self.stride, bias=self.bias, padding=0) * \
                   input_K * \
                   weight_alpha
        else:
            return self.inference_forward(input, self.weight, *self.padding, *self.stride, self.input_channels,
                                          self.output_channels, self.kernel_size)


if __name__ == "__main__":
    # test that output is correct
    input_batch_size = 1
    input_channels = 3
    input_height = 4
    input_width = 4
    output_channels = 64
    kernel_height = 3
    kernel_width = 3
    input_numel = input_height * input_width
    kernel_numel = kernel_height * kernel_width
    debug_weight = np.array(
        [[random.choice((-1, 1)) for i in range(kernel_numel)] for k in range(input_channels) for j in
         range(output_channels)]).reshape(
        output_channels, input_channels, kernel_height, kernel_width)
    input_image = np.array(
        [[random.choice((-1, 1)) for i in range(input_numel)] for j in range(input_channels) for k in
         range(input_batch_size)]) \
        .reshape(
        input_batch_size,
        input_channels,
        input_height,
        input_width)
    padding = 1

    output_height = input_height + padding * 2 - (kernel_height - 1)
    output_width = input_width + padding * 2 - (kernel_width - 1)
    expected_output = np.zeros((input_batch_size, output_channels, output_height, output_width))
    for k in range(input_batch_size):
        for i in range(output_channels):
            for j in range(input_channels):
                conv = signal.correlate2d(input_image[k][j], debug_weight[i][j], boundary='fill', fillvalue=1,
                                          mode='same')
                expected_output[k, i] += conv
    print('see expected output shape: ', expected_output.shape)
    crap_module = XNorConv2D(input_channels, output_channels, (kernel_height, kernel_width), padding=padding,
                             debug_weight=torch.Tensor(debug_weight))
    actual_output = crap_module(torch.Tensor(input_image)).detach().numpy().astype(np.float32)
    # print('see expected output: ', expected_output)
    # print('see actual_output: ', actual_output)
    outcome = np.all(expected_output == actual_output)
    if not outcome:
        print("Not equal. See input:")
        print(input_image)
        print(input_image.shape)
        print("see weights subset:")
        print(debug_weight[:3])
        print(debug_weight.shape)
        print("see actual output subset:")
        print(actual_output[:3])
        print("see expected output subset:")
        print(expected_output[:3])
        print("see expected output shape: ", expected_output.shape)
        print("see actual output shape: ", actual_output.shape)
    else:
        print('true')
