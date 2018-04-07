import torch
import torch.nn as nn
from torch.autograd import Variable
from scipy.signal import convolve2d, correlate2d


class ScipyConv2dFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, filter):
        result = correlate2d(input.numpy(), filter.numpy(), mode='valid')
        ctx.save_for_backward(input, filter)
        return input.new(result)

    @staticmethod
    def backward(ctx, grad_output):
        input, filter = ctx.saved_tensors
        grad_output = grad_output.data
        grad_input = convolve2d(grad_output.numpy(), filter.t().numpy(), mode='full')
        grad_filter = convolve2d(input.numpy(), grad_output.numpy(), mode='valid')

        return Variable(grad_output.new(grad_input)), \
            Variable(grad_output.new(grad_filter))


class ScipyConv2d(torch.nn.modules.module.Module):

    def __init__(self, kh, kw):
        super(ScipyConv2d, self).__init__()
        self.filter = nn.parameter.Parameter(torch.randn(kh, kw))

    def forward(self, input):
        return ScipyConv2dFunction.apply(input, self.filter)


module = ScipyConv2d(3, 3)
print(list(module.parameters()))
input = Variable(torch.randn(10, 10), requires_grad=True)
output = module(input)
print(output)
output.backward(torch.randn(8, 8))
print(input.grad)
