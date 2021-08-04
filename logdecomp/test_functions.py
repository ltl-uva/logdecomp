import torch
import pytest

from logdecomp import logdetexp, invexp


# initialize seed in all tests
@pytest.fixture(autouse=True)
def random():
    torch.manual_seed(42)


def test_logdetexp_forward():
    X = torch.randn(20, 20)
    expected = X.double().exp().slogdet().logabsdet.float()
    obtained = logdetexp(X)
    assert torch.allclose(expected, obtained)


def test_invexp_forward():
    X = torch.randn(20, 20)
    expected = X.double().exp().inverse().float()
    obtained = invexp(X)
    assert torch.allclose(expected, obtained)


def test_batch_logdetexp_forward():
    b, d = 10, 20
    X = torch.randn(b, d, d)
    lengths = torch.randint(2, d, (b,))

    expected = torch.tensor([
        X[k, :lengths[k], :lengths[k]].double().exp().slogdet().logabsdet
        for k in range(b)]).float()

    obtained = logdetexp(X, lengths)

    assert torch.allclose(expected, obtained)


def test_batch_invexp_forward():
    b, d = 10, 20
    X = torch.randn(b, d, d)
    lengths = torch.randint(2, d, (b,))

    obtained = invexp(X, lengths)

    for k in range(b):
        expected = (X[k, :lengths[k], :lengths[k]]
                    .double()
                    .exp()
                    .inverse()
                    .float())

        assert torch.allclose(expected, obtained[k, :lengths[k], :lengths[k]])


def test_batch_logdetexp_backward():
    b, d = 10, 20

    X = torch.randn(b, d, d, requires_grad=True)
    lengths = torch.randint(2, d, (b,))
    dloss = torch.randn(b)

    logdet_expected = torch.cat([
        (X[k, :lengths[k], :lengths[k]]
            .double()
            .exp()
            .slogdet()
            .logabsdet
            .unsqueeze(0))
        for k in range(b)]).float()

    logdet_obtained = logdetexp(X, lengths)

    grad_expected, = torch.autograd.grad(logdet_expected, X, dloss)
    grad_obtained, = torch.autograd.grad(logdet_obtained, X, dloss)

    assert torch.allclose(grad_expected, grad_obtained)


def test_batch_invexp_backward():
    b, d = 10, 5

    X = torch.randn(b, d, d, requires_grad=True)
    lengths = torch.randint(2, d, (b,))
    dloss = torch.randn(b, d, d)

    inv_expected = torch.stack([
        torch.nn.functional.pad(
            X[k, :lengths[k], :lengths[k]]
                .double()
                .exp()
                .inverse(),
            (0, d - lengths[k], 0, d - lengths[k]),
            "constant",
            0)
        for k in range(b)]).float()

    inv_obtained = invexp(X, lengths)

    grad_expected, = torch.autograd.grad(inv_expected, X, dloss)
    grad_obtained, = torch.autograd.grad(inv_obtained, X, dloss)

    # this is numerically sketchier. Fails for many seeds.
    assert torch.allclose(grad_expected, grad_obtained)


def test_batch_invexp_sign():

    # test is a bit numerically unstable but seems correct

    b, d = 2, 5
    A = torch.randn(b, d, d)
    lengths = [d] * b

    # test 1d sign
    A_neg = -torch.abs(A)
    X = torch.log(-A_neg)
    sign = True

    inv_expected = A_neg.double().inverse()
    inv_obtained = invexp(X, lengths, sign).double()
    assert torch.allclose(inv_expected, inv_obtained)

    # test 3d sign (taken from A)
    sign = A < 0
    X = torch.log(torch.abs(A))
    inv_expected = A.double().inverse()
    inv_obtained = invexp(X, lengths, sign).double()
    assert torch.allclose(inv_expected, inv_obtained)

    # test 2d sign
    sign = torch.randn(d, d) > 0
    A_same_sign = torch.abs(A)
    A_same_sign[:, sign] *= -1

    inv_expected = A_same_sign.double().inverse()
    # same X as above, but diff. signs
    inv_obtained = invexp(X, lengths, sign).double()
    assert torch.allclose(inv_expected, inv_obtained)


