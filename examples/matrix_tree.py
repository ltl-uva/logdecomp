import torch

NINF = float('-inf')

from logdecomp.functions import logdet_and_inv_exp


def logz_and_marginals_mtt(X, lengths):
    """Non-projective dependency parsing log Z and marginals.

    Uses the matrix-tree theorem.

    Based on torch-struct by Sasha Rush.
    """

    batch, N, N = X.shape

    # mask out everything outside of the lengths. -- TODO might be done for us already.
    ix = torch.arange(N, device=X.device).expand(batch, N)
    lengths_t = (torch.as_tensor(lengths, device=X.device)
                      .unsqueeze(1))
    ix = ix < lengths_t
    det_offset = torch.diag_embed((~ix).float())
    ix = ix.unsqueeze(2).expand(-1, -1, N)
    mask = torch.transpose(ix, 1, 2) * ix
    mask = mask.float()
    mask[mask == 0] = NINF
    mask[mask == 1] = 0

    X = X + mask

    Xmax = X.max(dim=-1)[0].max(dim=-1)[0]
    X = X - Xmax.unsqueeze(-1).unsqueeze(-1)

    eye = torch.eye(N, device=X.device)
    log_lap = X.masked_fill(eye != 0, NINF)
    log_diag = torch.logsumexp(log_lap, dim=1)

    torch.diagonal(log_lap, dim1=-2, dim2=-1)[...] = log_diag

    # set root scores to first row
    log_lap[:, 0] = torch.diagonal(X, 0, -2, -1)

    # sign bit: which coordinates of the laplacian have negative sign?
    sign = eye.to(dtype=torch.bool).clone()
    sign[0, :] = 1
    sign = ~sign

    logdet, inv = logdet_and_inv_exp(log_lap, lengths, sign)
    logdet += lengths_t.squeeze() * Xmax

    factor = (
        torch.diagonal(inv, 0, -2, -1)
        .unsqueeze(2)
        .expand_as(X)
        .transpose(1, 2)
    )
    term1 = X.exp().mul(factor).clone()
    term2 = X.exp().mul(inv.transpose(1, 2)).clone()
    term1[:, :, 0] = 0
    term2[:, 0] = 0
    output = term1 - term2
    roots_output = (
        torch.diagonal(X, 0, -2, -1)
        .exp()
        .mul(inv.transpose(1, 2)[:, 0])
    )
    output = output + torch.diag_embed(roots_output, 0, -2, -1)
    return logdet, output


def main():

    torch.manual_seed(42)
    X = torch.randn(2, 4, 4, requires_grad=True)
    lengths = [3, 4]
    logdet, mu = logz_and_marginals_mtt(X, lengths)

    from torch_struct import deptree_nonproj, deptree_part

    logdet_exp = deptree_part(X, False, lengths, eps=0)
    mu_exp = deptree_nonproj(X, False, lengths, eps=0)

    print(logdet, logdet_exp)

    print(mu)
    print(mu_exp)


if __name__ == '__main__':
    main()

