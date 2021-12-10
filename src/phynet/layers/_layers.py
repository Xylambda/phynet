import torch
import torch.nn as nn


class CustomLinear(nn.Module):
    """
    Reimplementation of `torch.nn.Linear` layer.

    Parameters
    ----------
    in_features : int
        Number of input units.
    out_features : int
        Number of output units.
    bias : bool, optional, default: False
        Whether to use the bias term (True) or not (False).

    Attributes
    ----------
    weights : torch.Tensor
        Optimizable weights matrix.
    bias : torch.Tensor
        Obtimizable biases vector.
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool=False):
        super(CustomLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        _weight = torch.empty(self.in_features, self.out_features)
        self.weight = nn.Parameter(_weight)
        
        if bias:
            _bias = torch.empty(self.out_features)
            self.bias = nn.Parameter(_bias)
        else:
            self.register_parameter('bias', None)

        # initialize with Kaiming init to reduce convergence problems
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, bias=self.bias)

        """
        self._chech_shape(x)

        out = torch.matmul(x, self.weight.t())

        if self.bias is not None:
            out = out + self.bias

        return out
        """

    def _chech_shape(self, x: torch.Tensor):
        if x.shape[0] != self.in_features:
            raise ValueError(f"Wrong input shape. Expected {self.in_features}")

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=torch.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / torch.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
