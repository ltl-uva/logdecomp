import torch
from .lu import LogDomainLU
from .lu import BatchLogDomainLU


class LogDetExp(torch.autograd.Function):

    @classmethod
    def forward(cls, ctx, X):
        ctx.save_for_backward(X.detach())
        ctx.lu = LogDomainLU(X)
        z = ctx.lu.logdet()
        return X.new_tensor(z)  # just a scalar, we're ok with a copy here.

    @classmethod
    def backward(cls, ctx, dz):
        X, = ctx.saved_tensors
        Y_np = ctx.lu.inv()
        Y = (torch.from_numpy(Y)
                  .to(dtype=X.dtype, device=X.device))
        return Y.T * torch.exp(X) * dz


class BatchLogDetExp(torch.autograd.Function):

    @classmethod
    def forward(cls, ctx, X, lengths):
        ctx.save_for_backward(X.detach())
        ctx.lu = BatchLogDomainLU(X, lengths)
        logdet_np = ctx.lu.logdet()
        logdet = (torch.from_numpy(logdet_np)
                       .to(dtype=X.dtype, device=X.device))
        return logdet

    @classmethod
    def backward(cls, ctx, dz):
        # dz is a vector
        X, = ctx.saved_tensors
        Y_np = ctx.lu.inv()
        Y = (torch.from_numpy(Y_np)
                  .to(dtype=X.dtype, device=X.device))
        Yt = Y.transpose(-2, -1)
        dzu = dz.unsqueeze(-1).unsqueeze(-1)
        return Yt * torch.exp(X) * dzu, None


class InvExp(torch.autograd.Function):
    @classmethod
    def forward(cls, ctx, X):
        Y_np = LogDomainLU(X).inv()
        Y = (torch.from_numpy(Y_np)
                  .to(dtype=X.dtype, device=X.device))
        ctx.save_for_backward(X.detach(), Y)
        return Y

    @classmethod
    def backward(cls, ctx, dZ):
        X, Y = ctx.saved_tensors
        return (-Y.T @ dZ @ Y.T) * torch.exp(X)


class BatchInvExp(torch.autograd.Function):
    @classmethod
    def forward(cls, ctx, X, lengths):
        Y_np = BatchLogDomainLU(X, lengths).inv()
        Y = (torch.from_numpy(Y_np)
                  .to(dtype=X.dtype, device=X.device))
        ctx.save_for_backward(X.detach(), Y)
        return Y

    @classmethod
    def backward(cls, ctx, dZ):
        X, Y = ctx.saved_tensors
        Yt = Y.transpose(-2, -1)
        return (-Yt @ dZ @ Yt) * torch.exp(X), None


def logdetexp(X, lengths=None):
    """Stable computation of log |det exp(X)| (exp/log elementwise).

    Equivalent of X.exp().slogdet().logabsdet.

    Uses Eigen LU decomposition with full pivoting on top of a custom
    log-domain float64 datatype, for full computation in log domain.

    Parameters:
    -----------

    X: torch Tensor, shape = [d, d] or [b, d, d]
        The matrix or batch of matrices to factor.

    lengths: list[Int], shape = [b]
        If batched, lengths[i] is the number of rows and columns of X[i].
        Required if X.ndim != 2.
    """

    if X.ndim == 2:
        return LogDetExp.apply(X)
    elif X.ndim == 3:
        if isinstance(lengths, torch.Tensor):
            lengths = lengths.tolist()
        return BatchLogDetExp.apply(X, lengths)
    else:
        raise ValueError("X.shape must be [d, d] or [b, d, d].")


def invexp(X, lengths=None):
    """Stable computation of exp(X)^-1 (exp elementwise).

    Equivalent of X.exp().inverse().

    Uses Eigen LU decomposition with full pivoting on top of a custom
    log-domain float64 datatype, for full computation in log domain.

    Parameters:
    -----------

    X: torch Tensor, shape = [d, d] or [b, d, d]
        The matrix or batch of matrices to factor.

    lengths: list[Int], shape = [b]
        If batched, lengths[i] is the number of rows and columns of X[i].
        Required if X.ndim != 2.
    """

    if X.ndim == 2:
        return InvExp.apply(X)
    elif X.ndim == 3:
        if isinstance(lengths, torch.Tensor):
            lengths = lengths.tolist()
        return BatchInvExp.apply(X, lengths)
    else:
        raise ValueError("X.shape must be [d, d] or [b, d, d].")
