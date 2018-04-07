import torch
from torch.autograd import Variable
from numpy.fft import rfft2, irfft2


class BadFFTFunction(torch.autograd.Function):

    def forward(self, input):
        numpy_input = input.numpy()
        result = abs(rfft2(numpy_input))
        return input.new(result)

    def backward(self, grad_output):
        numpy_go = grad_output.numpy()
        result = irfft2(numpy_go)
        return grad_output.new(result)

# since this layer does not have any parameters, we can
# simply declare this as a function, rather than as an nn.Module class

def incorrect_fft(input):
    return BadFFTFunction()(input)


input = Variable(torch.randn(8, 8), requires_grad=True)

result = incorrect_fft(input)
print(result.data)

result.backward(torch.randn(result.size()))
print(input.grad)
