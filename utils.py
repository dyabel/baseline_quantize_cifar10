import torch
import torch.nn as nn
from torch.nn import functional as F#for Conv2d
from torch.autograd import Function
import numpy as np
from torch.nn.parameter import Parameter
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
bitsW = 8
bitsA = 8
bitsG = 8
bitsE = 8
bitsR = 8
lr = 0.001


def delta(bits):
    result = (2.**(1-bits))
    return result

def shift(x):
    return 2**torch.round(torch.log(x) / torch.log(torch.tensor(2.0)))
#g_vml_cpu not implemented for 'Long log_vml_cpu not implemented for 'Long

def clip(x, bits):
    if bits >= 32:
        step = 0
    else:
        step = delta(bits)
    ceil  = 1 - step
    floor = step - 1
    result = torch.clamp(x, floor, ceil)
    return result

def quant(x, bits):
    if bits >= 32:
        result = x
    else:
        result = torch.round(x/delta(bits))*delta(bits)
        result.requires_grad = True
    return result

def qw(x):
    bits = bitsW
    if bits >= 32:
        result = x
    else:
        result = clip(quant(x,bits),bits)
        result.requires_grad = True
    return result
def qa(x):
    bits = bitsA
    if bits >= 32:
        result = x
    else:
        #result = clip(quant(x,bits),bits)
        x = clip(x,bits)
        y = quant(x,bits).detach()
        
        #result = clip(x,bits)
        #result = quant(x,bits)
        #result.requires_grad = True
    return y


def qg(x) :
    x = x.cuda()
    if bitsG>32:
        return x
    else:
        xmax = torch.max(torch.abs(x))
        x = x/shift(xmax)
        norm = quant(lr * x,bitsR)
        norm_sign = torch.sign(norm)
        norm_abs  = torch.abs(norm)
        norm_int  = torch.floor(norm_abs)
        norm_float = norm_abs - norm_int
        rand_float = torch.rand(x.size()).cuda()
        #print('norm:',norm_sign*(norm_int+0.5*(torch.sign(norm_float-rand_float)+1)))
        norm = norm_sign*(norm_int+0.5*(torch.sign(norm_float-rand_float)+1))
        return norm / delta(bitsG)
        
def qe(x) :
    if bitsE>32:
        return x
    else:
        xmax,_ = torch.max(torch.abs(x),0)
        #x = x/shift(xmax)
        #return clip(quant(x,bitsE),bitsE)
        return quant(x,bitsE)


class QW(Function):
    @staticmethod
    def forward(self, x):
        result = qw(x)
        return result

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output
        return grad_input

class QA(Function):
    @staticmethod
    def forward(self, x):
        #print('#'*20)
        #@print(x[1][1][1][1])
        #print(qa(x)[1][1][1][1])
        result = qa(x)
        #print(result)
        return result

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output
        #print(grad_input)
        return grad_input

class QE(Function):
    @staticmethod
    def forward(self, x):
        return x

    @staticmethod
    def backward(self, grad_output):

        #print(qe(grad_output[1][1][1][1]))
        return qe(grad_output)

class QG(Function):
    @staticmethod
    def forward(self, x):
        return x

    @staticmethod
    def backward(self, grad_output):
        grad_input = qg(grad_output)
        return grad_input
quantizeW = QW().apply
quantizeA = QA().apply
quantizeG = QG().apply
quantizeE = QE().apply





#quantization conv
class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)

        #stride
        #s1 for line; s2 for column
        if isinstance(stride,int) or len(stride) == 1:
            self.s1 = stride
            self.s2 = stride
        elif len(stride) == 2:
            self.s1 = stride[0]
            self.s2 = stride[1]
        #padding
        self.p = padding

        #kernel size
        if isinstance(kernel_size,int) or len(kernel_size) == 1:
            self.k1 = kernel_size
            self.k2 = kernel_size
        elif len(kernel_size) == 2:
            self.k1 = kernel_size[0]
            self.k2 = kernel_size[1]

        #output feature size
        self.output_H = None
        self.output_W = None

        self.input_shape = None


    def conv2d_forward(self, input, weight):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        self.input_shape = input.shape
        #print(self.input_shape)
        #print(self.weight[1][1][1][1])
        #print('quantizew:',quantizeW(self.weight))
        #print('quantizeA:',quantizeA(input))
        #print(self.conv2d_forward(quantizeA(input),quantizeW(self.weight)))
        return quantizeG(self.conv2d_forward(input,quantizeW(self.weight)))
        #return self.conv2d_forward(input,quantizeW(self.weight))
        #return self.conv2d_forward(input,self.weight)

class MyConv(nn.Module):
    def __init__(self, in_channels,out_channels,kernel_size=3,padding=0):
        super(MyConv,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,padding)
        self.kernel_size = kernel_size
    def forward(self,x):
        self.conv.weight =quantizeW(self.conv.weight)
        x = quantizeA(x)
        x = self.conv(x)
        return x 

class ReLU(nn.ReLU):
    def __init__(self,inplace = True):
        super(ReLU,self).__init__(inplace)
        self.inplace = inplace
    def forward(self, input):
        x =  F.relu(input,inplace = self.inplace)
        #x = quantizeE(x)
        x = quantizeA(x)
        return x


    


