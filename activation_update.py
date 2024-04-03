"""Scaling function to define custom backward pass if needed.
"""
import torch


class MultiplicationFactor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, scale):
        ctx.save_for_backward(inputs, scale)
        return inputs * scale

    @staticmethod
    def backward(ctx, grad_output):
        inputs, scale = ctx.saved_tensors
        return grad_output * scale, grad_output * inputs
