from torch.nn import functional as F
import torch
import torch.nn as nn
from utils.anchors import Anchors
import math
from nets.efficientnet import EfficientNet as EffNet
from nets.layers import (Conv2dStaticSamePadding, MaxPool2dStaticSamePadding,
                         MemoryEfficientSwish, Swish)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class ECAWeightModule(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECAWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        #return x * y.expand_as(x)
        return y

class SEWeightModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEWeightModule,self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction ,channels,kernel_size=1,padding=0)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        out = self.avg_pooling(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)
        
        return weight

class Conv2dStaticSamePadding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]
        
        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]
        
        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.conv(x)
        return x
    
class CDilated(nn.Module):
    '''
    This class defines the dilated convolution.
    '''
    def __init__(self, In, Out, kSize, stride=1, d=1):
        super().__init__()
        self.In = In
        self.Out = Out
        self.kSize = kSize
        self.stride = stride
        padding = int((kSize - 1)/2) * d
        self.conv = nn.Conv2d(In, Out, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False, dilation=d)
        '''
        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kSize, int):
            self.kSize = [self.kSize] * 2
        elif len(self.kSize) == 1:
            self.kSize = [self.kSize[0]] * 2
        '''

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        
        h, w = input.shape[-2:]
        
        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kSize[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kSize[0]
        
        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        input = F.pad(input, [left, right, top, bottom])
        '''
        output = self.conv(input)
        return output
    
class AvgPool2dStaticSamePadding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.AvgPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]
        
        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.pool(x)
        return x

class PDAM(nn.Module):
    def __init__(self, nIn, nOut):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        '''
        super().__init__()
        n = int(nOut/4)
        n1 = nOut - 3*n
        b = int(nIn/4)
        b1 = nIn - 3*b
        self.conv1 = nn.Conv2d(in_channels=nIn, out_channels=b1, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=nIn, out_channels=nOut, kernel_size=1, stride=1, bias=False)
        self.c1 = MaxPool2dStaticSamePadding(3,2)
        self.d1 = CDilated(nIn, b, 3, 2, 1) # dilation rate of 2^0
        self.d2 = CDilated(nIn, b, 3, 2, 2) # dilation rate of 2^1
        self.d4 = CDilated(nIn, b, 3, 2, 4) # dilation rate of 2^2
        #self.d8 = CDilated(nIn, n, 3, 2, 8) # dilation rate of 2^3
        self.eca = ECAWeightModule( nOut // 4)
        self.split_channel = nIn//4
        self.softmax = nn.Softmax(dim=1)
        self._bn3 = nn.BatchNorm2d(num_features=nOut, eps=1e-3)

    def forward(self, x):
        batch_size = x.shape[0]
        d0 = self.conv1(x)
        #x3 = self.conv2(x)
        d0 = self.c1(d0)
        d1 = self.d1(x)
        d2 = self.d2(x)
        d3 = self.d4(x)
        #d4 = self.d8(x)

        add2 = d1
        add3 = add2 + d2
        add4 = add3+ d3

        feats = torch.cat((d0,add2,add3,add4),dim=1)
        feats = feats.view(batch_size,4,self.split_channel,feats.shape[2],feats.shape[3])
        
        x1_eca = self.eca(d0)
        x2_eca = self.eca(d1)
        x3_eca = self.eca(d2)
        x4_eca = self.eca(d3)
        x_eca = torch.cat((x1_eca,x2_eca,x3_eca,x4_eca),dim=1)
        attention_vectors = x_eca.view(batch_size,4,self.split_channel,1,1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_eca_weight_fp = feats_weight[:, i, :, :]
            if i ==0:
                out = x_eca_weight_fp
            else:
                out = torch.cat((x_eca_weight_fp,out),1)

        out = self.conv2(out)
        out = self._bn3(out)
        return out


#----------------------------------#
class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels, kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x

class BiFPN(nn.Module):
    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False, attention=True):
        super(BiFPN, self).__init__()
        self.pdam6 = PDAM(nIn = conv_channels[2], nOut = num_channels)
        self.epsilon = epsilon
        self.conv6_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv3_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        self.conv4_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv6_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv7_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p7_upsample = nn.Upsample(scale_factor=4, mode='nearest')
        self.p8_upsample = nn.Upsample(scale_factor=8, mode='nearest')

        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p6_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p7_downsample = MaxPool2dStaticSamePadding(3, 2)


        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        self.first_time = first_time
        if self.first_time:
            self.p5_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p4_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p3_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

            self.p5_to_p6 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
                MaxPool2dStaticSamePadding(3, 2)
            )
            self.p6_to_p7 = nn.Sequential(
                MaxPool2dStaticSamePadding(3, 2)
            )

            self.p4_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p3_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()

        self.attention = attention

    def forward(self, inputs):

        if self.attention:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward_fast_attention(inputs)
        else:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward(inputs)

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward_fast_attention(self, inputs):

        #         C3  64, 64, 40
        #         C4  32, 32, 112
        #         C5  16, 16, 320
        if self.first_time:

            p3, p4, p5 = inputs
            #-------------------------------------------#
            #   C3 64, 64, 40 -> 64, 64, 64
            #-------------------------------------------#
            p3_in = self.p3_down_channel(p3)
            #-------------------------------------------#
            #   C4 32, 32, 112 -> 32, 32, 64
            #                  -> 32, 32, 64
            #-------------------------------------------#
            p4_in_1 = self.p4_down_channel(p4)
            p4_in_2 = self.p4_down_channel_2(p4)
            #-------------------------------------------#
            #   C5 16, 16, 320 -> 16, 16, 64
            #                  -> 16, 16, 64
            #-------------------------------------------#
            p5_in_1 = self.p5_down_channel(p5)
            p5_in_2 = self.p5_down_channel_2(p5)
            #-------------------------------------------#
            #   C5 16, 16, 320 -> 8, 8, 64
            #-------------------------------------------#
            #p6_in = self.p5_to_p6(p5)
            p6_in = self.pdam6(p5)
            #-------------------------------------------#
            #   P6_in 8, 8, 64 -> 4, 4, 64
            #-------------------------------------------#
            p7_in = self.p6_to_p7(p6_in)
            #p7_in = self.pdam6(p6_in)

            p6_w1 = self.p6_w1_relu(self.p6_w1)
            weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
            p6_td= self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))


            p5_w1 = self.p5_w1_relu(self.p5_w1)
            weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
            p5_td= self.conv5_up(self.swish(weight[0] * p5_in_1 + weight[1] * self.p5_upsample(p6_td) ))

            p4_w1 = self.p4_w1_relu(self.p4_w1)
            weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
            p4_td= self.conv4_up(self.swish(weight[0] * p4_in_1 + weight[1] * self.p4_upsample(p5_td) ))

            p3_w1 = self.p3_w1_relu(self.p3_w1)
            weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
            p3_td = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_td)))
            p6_w2 = self.p6_w2_relu(self.p6_w2)
            weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
            p3_out_1 = p3_td
            
            p6_w2 = self.p6_w2_relu(self.p6_w2)
            weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
            p3_out =  p3_out_1 +  p3_in  

            p4_w2 = self.p4_w2_relu(self.p4_w2)
            weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
            p4_out_1 = self.conv4_down(
                self.swish(weight[0] * p4_td+ weight[1] * self.p4_downsample(p3_out) + weight[2]*p4_in_1))
            
            p7_w2 = self.p7_w2_relu(self.p7_w2)
            weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
            p4_out =  p4_out_1 +  p4_in_1 

            p5_w2 = self.p5_w2_relu(self.p5_w2)
            weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
            p5_out_1 = self.conv5_down(
                self.swish( weight[0] * p5_td+ weight[1] * self.p5_downsample(p4_out)  + weight[2]*p5_in_1))
            
            p7_w2 = self.p7_w2_relu(self.p7_w2)
            weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
            p5_out = p5_out_1 +  p5_in_1 

            p6_out = p6_td 
            '''
            p7_w2 = self.p7_w2_relu(self.p7_w2)
            weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
            p6_out = weight[0] *  p6_out + weight[1] * p6_in
            '''
            p7_out = p7_in
            '''
            p7_w2 = self.p7_w2_relu(self.p7_w2)
            weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
            p7_out = weight[0] *  p7_out + weight[1] * p7_in 
            '''
        else:
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

            p6_w1 = self.p6_w1_relu(self.p6_w1)
            weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
            p6_td= self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))

            p5_w1 = self.p5_w1_relu(self.p5_w1)
            weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
            p5_td= self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_td) + weight[2] * self.p7_upsample(p7_in) ))

            p4_w1 = self.p4_w1_relu(self.p4_w1)
            weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
            p4_td= self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_td) ))

            p3_w1 = self.p3_w1_relu(self.p3_w1)
            weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
            p3_td = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_td)))
            p6_w2 = self.p6_w2_relu(self.p6_w2)
            weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
            p3_out_1 = p3_td
            
            p6_w2 = self.p6_w2_relu(self.p6_w2)
            weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
            p3_out = p3_out_1 + p3_in  

            p4_w2 = self.p4_w2_relu(self.p4_w2)
            weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
            p4_out_1 = self.conv4_down(
                self.swish(weight[0] * p4_td+ weight[1] * self.p4_downsample(p3_out) + weight[2]*p4_in))
            
            p7_w2 = self.p7_w2_relu(self.p7_w2)
            weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
            p4_out =  p4_out_1 + p4_in

            p5_w2 = self.p5_w2_relu(self.p5_w2)
            weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
            p5_out_1 = self.conv5_down(
                self.swish( weight[0] * p5_td+ weight[1] * self.p5_downsample(p4_out)  + weight[2]*p5_in))
            
            p7_w2 = self.p7_w2_relu(self.p7_w2)
            weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
            p5_out = p5_out_1 +  p5_in

            p6_out = p6_td 
            '''
            p7_w2 = self.p7_w2_relu(self.p7_w2)
            weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
            p6_out =self.conv5_down(
                self.swish(weight[0] * p6_out_1 + weight[1] * p6_in ))
            '''
            p7_out = p7_in
            '''
            p7_w2 = self.p7_w2_relu(self.p7_w2)
            weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
            p7_out = self.conv5_down(
                self.swish(weight[0] * p7_out_1 + weight[1] * p7_in +  weight[2]*self.p4_downsample(p6_out)))
            '''

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward(self, inputs):
        if self.first_time:
            #pi = 6, 7
            p3, p4, p5 = inputs
            p3_in = self.p3_down_channel(p3)
            p4_in_1 = self.p4_down_channel(p4)
            p4_in_2 = self.p4_down_channel_2(p4)
            p5_in_1 = self.p5_down_channel(p5)
            p5_in_2 = self.p5_down_channel_2(p5)
            #p6_in = self.p5_to_p6(p5)
            p6_in = self.pdam6(p5)
            p7_in = self.p6_to_p7(p6_in)
            #p7_in = self.pdam6(p6_in)

            p6_td= self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))

            p5_td= self.conv5_up(self.swish(p5_in_1 +  self.p5_upsample(p6_td)))

            p4_td= self.conv4_up(self.swish(p4_in_1 +  self.p4_upsample(p5_td)))

            p3_td = self.conv3_up(self.swish( p3_in + self.p3_upsample(p4_td)))

            p3_out = p3_td

            p4_out = self.conv4_down(
                self.swish(p4_in_2 + p4_td+ self.p4_downsample(p3_out) ))

            p5_out = self.conv5_down(
                self.swish( p5_in_2 +  p5_td+ self.p5_downsample(p4_out)  ))

            p6_out = p6_td

            p7_out = p7_in

        else:
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

            p6_td= self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))

            p5_td= self.conv5_up(self.swish(p5_in +  self.p5_upsample(p6_td)))
            
            p4_td= self.conv4_up(self.swish(p4_in +  self.p4_upsample(p5_td)))

            p3_td = self.conv3_up(self.swish( p3_in + self.p3_upsample(p4_td)))

            p3_out = p3_td

            p4_out = self.conv4_down(
                self.swish(p4_in + p4_td+ self.p4_downsample(p3_out)  ))
            
            p5_out = self.conv5_down(
                self.swish( p5_in +  p5_td+ self.p5_downsample(p4_out)  ))
            p6_out = p6_td

            p7_out = p7_in

        return p3_out, p4_out, p5_out, p6_out, p7_out

class BoxNet(nn.Module):
    def __init__(self, in_channels, num_anchors, num_layers, onnx_export=False):
        super(BoxNet, self).__init__()
        self.num_layers = num_layers

        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])

        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in range(5)])
        # 9
        # 4
        self.header = SeparableConvBlock(in_channels, num_anchors * 4, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):

            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], -1, 4)
            
            feats.append(feat)
        feats = torch.cat(feats, dim=1)

        return feats

class ClassNet(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, num_layers, onnx_export=False):
        super(ClassNet, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.conv_list  = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in range(5)])
        # num_anchors = 9
        # num_anchors num_classes
        self.header = SeparableConvBlock(in_channels, num_anchors * num_classes, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], feat.shape[1], feat.shape[2], self.num_anchors, self.num_classes)
            feat = feat.contiguous().view(feat.shape[0], -1, self.num_classes)

            feats.append(feat)
        feats = torch.cat(feats, dim=1)
        feats = feats.sigmoid()

        return feats

class EfficientNet(nn.Module):
    def __init__(self, phi, load_weights=False):
        super(EfficientNet, self).__init__()
        model = EffNet.from_pretrained(f'efficientnet-b{phi}', load_weights)
        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc
        self.model = model

    def forward(self, x):
        x = self.model._conv_stem(x)
        x = self.model._bn0(x)
        x = self.model._swish(x)
        feature_maps = []

        last_x = None
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

            if block._depthwise_conv.stride == [2, 2]:
                feature_maps.append(last_x)
            elif idx == len(self.model._blocks) - 1:
                feature_maps.append(x)
            last_x = x
        del last_x
        return feature_maps[1:]

class EfficientDetBackbone(nn.Module):
    def __init__(self, num_classes=80, phi=0, load_weights=False):
        super(EfficientDetBackbone, self).__init__()

        self.phi = phi

        self.backbone_phi = [0, 1, 2, 3, 4, 5, 6, 6]

        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384]

        #self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8]
        self.fpn_cell_repeats = [1, 1, 1, 1, 1, 1, 1, 1]

        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5]

        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5.]
        num_anchors = 9
        conv_channel_coef = {
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
        }

        #------------------------------------------------------#
        #   P3_out      64,64,64
        #   P4_out      32,32,64
        #   P5_out      16,16,64
        #   P6_out      8,8,64
        #   P7_out      4,4,64
        #------------------------------------------------------#
        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.phi],
                    conv_channel_coef[phi],
                    True if _ == 0 else False,
                    attention=True if phi < 6 else False)
              for _ in range(self.fpn_cell_repeats[phi])])

        self.num_classes = num_classes

        self.regressor = BoxNet(in_channels=self.fpn_num_filters[self.phi], num_anchors=num_anchors,
                                   num_layers=self.box_class_repeats[self.phi])

        self.classifier = ClassNet(in_channels=self.fpn_num_filters[self.phi], num_anchors=num_anchors,
                                     num_classes=num_classes, num_layers=self.box_class_repeats[self.phi])

        self.anchors = Anchors(anchor_scale=self.anchor_scale[phi])

        #-------------------------------------------#
        #         C3  64, 64, 40
        #         C4  32, 32, 112
        #         C5  16, 16, 320
        #-------------------------------------------#
        self.backbone_net = EfficientNet(self.backbone_phi[phi], load_weights)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        _, p3, p4, p5 = self.backbone_net(inputs)

        features = (p3, p4, p5)
        features = self.bifpn(features)

        regression = self.regressor(features)
        classification = self.classifier(features)
        anchors = self.anchors(inputs)
    
        return features, regression, classification, anchors