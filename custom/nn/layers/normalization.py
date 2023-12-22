import mlx.core as mx
from mlx.nn.layers.base import Module

class LayerNormBasselCorrected(Module):
    r"""Applies layer normalization [1] on the inputs with Bessel's Correction used by default like PyTorch.

    Computes

    .. math::

        y = \frac{x - E[x]}{\sqrt{Var[x]} + \epsilon} \gamma + \beta,

    where :math:`\gamma` and :math:`\beta` are learned per feature dimension
    parameters initialized at 1 and 0 respectively.

    Var[x] would by default apply Bessel's Correction.

    [1]: https://arxiv.org/abs/1607.06450

    Args:
        dims (int): The feature dimension of the input to normalize over
        eps (float): A small additive constant for numerical stability
        affine (bool): If True learn an affine transform to apply after the
            normalization
        correction (bool): 
    """

    def __init__(self, dims: int, eps: float = 1e-5, affine: bool = True, correction: bool = True):
        super().__init__()
        if affine:
            self.bias = mx.zeros((dims,))
            self.weight = mx.ones((dims,))
        self.eps = eps
        self.dims = dims
        self.correction = correction

    def _extra_repr(self):
        return f"{self.dims}, eps={self.eps}, affine={'weight' in self}"

    def __call__(self, x):
        means = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True, ddof=int(self.correction))
        x = (x - means) * mx.rsqrt(var + self.eps)
        return (self.weight * x + self.bias) if "weight" in self else x


class LayerNormTorchAlike(Module):
    r"""Applies layer normalization [1] on the inputs in PyTorch's style.
    MLX's official LayerNorm has a different behavior with PyTorch's.

    Computes

    .. math::

        y = \frac{x - E[x]}{\sqrt{Var[x]} + \epsilon} \gamma + \beta,

    where :math:`\gamma` and :math:`\beta` are learned per feature dimension
    parameters initialized at 1 and 0 respectively.

    Var[x] would by default apply Bessel's Correction.

    [1]: https://arxiv.org/abs/1607.06450

    Args:
        dims (int): The feature dimension of the input to normalize over
        eps (float): A small additive constant for numerical stability
        affine (bool): If True learn an affine transform to apply after the
            normalization
        correction (bool): 
    """

    def __init__(self, dims: int, eps: float = 1e-5, affine: bool = True, correction: bool = True):
        super().__init__()
        if affine:
            self.bias = mx.zeros((dims,))
            self.weight = mx.ones((dims,))
        self.eps = eps
        self.dims = dims
        self.correction = correction

    def _extra_repr(self):
        return f"{self.dims}, eps={self.eps}, affine={'weight' in self}"

    def __call__(self, x):
        # Calculate the mean of all elements;
        # i.e. the means for each element $\mathbb{E}[X]$
        mean = x.mean(axis=-1, keepdims=True)
        # Calculate the squared mean of all elements;
        # i.e. the means for each element $\mathbb{E}[X^2]$
        mean_x2 = (x ** 2).mean(axis=-1, keepdims=True)
        # Variance of all element $Var[X] = \mathbb{E}[X^2] - \mathbb{E}[X]^2$
        var = mean_x2 - mean ** 2

        # Normalize $$\hat{X} = \frac{X - \mathbb{E}[X]}{\sqrt{Var[X] + \epsilon}}$$
        x_norm = (x - mean) / mx.sqrt(var + self.eps)
        # Scale and shift $$\text{LN}(x) = \gamma \hat{X} + \beta$$        
        x_norm = self.weight * x_norm + self.bias
        return x_norm
