import torch
from .lu import LogDomainLU
from .lu import BatchLogDomainLU


def _from_numpy_as(src_np, trg):
    return torch.from_numpy(src_np).to(dtype=trg.dtype, device=trg.device)


def _get_sign(sign):
    return torch.as_tensor(sign, dtype=bool, device='cpu')


def _apply_sign(expX, sign):
    expX[sign.expand_as(expX)] *= -1
    return expX


def _make_batch_lu(X, lengths, sign):
    return BatchLogDomainLU(X.numpy(), lengths, sign.numpy())


def _get_all(X, lengths, sign):
    X = X.detach()
    if isinstance(lengths, torch.Tensor):
        lengths = lengths.tolist()
    sign = _get_sign(sign)
    return X, sign, _make_batch_lu(X, lengths, sign)


class LogDetExp(torch.autograd.Function):

    @classmethod
    def forward(cls, ctx, X):
        X = X.detach()
        ctx.save_for_backward(X)
        ctx.lu = LogDomainLU(X.numpy())
        z = ctx.lu.logdet()
        return X.new_tensor(z)  # just a scalar, we're ok with a copy here.

    @classmethod
    def backward(cls, ctx, dz):
        X, = ctx.saved_tensors
        Y_np = ctx.lu.inv()
        Y = _from_numpy_as(Y_np, X)
        return Y.T * torch.exp(X) * dz


class BatchLogDetExp(torch.autograd.Function):

    @classmethod
    def forward(cls, ctx, X, lengths, sign):
        X, sign, ctx.lu = _get_all(X, lengths, sign)
        ctx.save_for_backward(X, sign)
        logdet_np = ctx.lu.logdet()
        logdet = _from_numpy_as(logdet_np, X)
        return logdet

    @classmethod
    def backward(cls, ctx, dz):
        X, sign = ctx.saved_tensors
        expX = _apply_sign(torch.exp(X), sign)
        Y_np = ctx.lu.inv()
        Y = _from_numpy_as(Y_np, X)
        Yt = Y.transpose(-2, -1)
        dzu = dz.unsqueeze(-1).unsqueeze(-1)
        return Yt * expX * dzu, None, None


class InvExp(torch.autograd.Function):
    @classmethod
    def forward(cls, ctx, X):
        X.detach()
        Y_np = LogDomainLU(X.numpy()).inv()
        Y = _from_numpy_as(Y_np, X)
        ctx.save_for_backward(X.detach(), Y)
        return Y

    @classmethod
    def backward(cls, ctx, dZ):
        X, Y = ctx.saved_tensors
        return (-Y.T @ dZ @ Y.T) * torch.exp(X)


class BatchInvExp(torch.autograd.Function):
    @classmethod
    def forward(cls, ctx, X, lengths, sign):
        X, sign, lu = _get_all(X, lengths, sign)
        Y_np = lu.inv()
        Y = _from_numpy_as(Y_np, X)
        ctx.save_for_backward(X, Y, sign)
        return Y

    @classmethod
    def backward(cls, ctx, dZ):
        X, Y, sign = ctx.saved_tensors
        Yt = Y.transpose(-2, -1)
        expX = _apply_sign(torch.exp(X), sign)
        return (-Yt @ dZ @ Yt) * expX, None, None


class BatchLogDetAndInvExp(torch.autograd.Function):
    @classmethod
    def forward(cls, ctx, X, lengths, sign):
        X, sign, lu = _get_all(X, lengths, sign)
        logdet_np = lu.logdet()
        Y_np = lu.inv()
        logdet = _from_numpy_as(logdet_np, X)
        Y = _from_numpy_as(Y_np, X)

        ctx.save_for_backward(X, Y, sign)
        return logdet, Y

    @classmethod
    def backward(cls, ctx, dl_dlogdet, dl_dY):
        X, Y, sign = ctx.saved_tensors
        Yt = Y.transpose(-2, -1)

        expX = _apply_sign(torch.exp(X), sign)

        d_via_logdet = Yt * expX * dl_dlogdet
        d_via_Y =  (-Yt @ dl_dY @ Yt) * expX

        return d_via_logdet + d_via_Y, None, None


def logdetexp(X, lengths=None, sign=False):
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

    sign: bool or torch Tensor, shape [0], [d, d] or [b, d, d]
        If sign[k, i, j] = True (or broadcastable), then the matrix factored has
        value -exp(A[k, i, j]) at that position, rather than +exp.
        Only supported in batch mode.
    """

    if X.ndim == 2:
        if sign != False:
            raise NotImplementedError("sign only implemented in batched case")
        return LogDetExp.apply(X)
    elif X.ndim == 3:
        if isinstance(lengths, torch.Tensor):
            lengths = lengths.squeeze().tolist()
        return BatchLogDetExp.apply(X, lengths, sign)
    else:
        raise ValueError("X.shape must be [d, d] or [b, d, d].")


def invexp(X, lengths=None, sign=False):
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

    sign: bool or torch Tensor, shape [0], [d, d] or [b, d, d]
        If sign[k, i, j] = True (or broadcastable), then the matrix factored has
        value -exp(A[k, i, j]) at that position, rather than +exp.
        Only supported in batch mode.
    """

    if X.ndim == 2:
        if sign != False:
            raise NotImplementedError("sign only implemented in batched case")
        return InvExp.apply(X)
    elif X.ndim == 3:
        return BatchInvExp.apply(X, lengths, sign)
    else:
        raise ValueError("X.shape must be [d, d] or [b, d, d].")


def logdet_and_inv_exp(X, lengths, sign):
    """Stable computation of log|S*exp(X)| and (S*exp(X))^-1 (exp elementwise).

    Uses Eigen LU decomposition with full pivoting on top of a custom
    log-domain float64 datatype, for full computation in log domain.

    Batch mode only for now.

    Parameters:
    -----------

    X: torch Tensor, shape = [b, d, d]
        The matrix or batch of matrices to factor.

    lengths: list[Int], shape = [b]
        lengths[i] is the number of rows and columns of X[i].

    sign: bool or torch Tensor, shape [0], [d, d] or [b, d, d]
        If sign[k, i, j] = True (or broadcastable), then the matrix factored has
        value -exp(A[k, i, j]) at that position, rather than +exp.
        Only supported in batch mode.
    """
    return BatchLogDetAndInvExp.apply(X, lengths, sign)
