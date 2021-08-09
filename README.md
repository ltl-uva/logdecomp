![tests workflow](https://github.com/ltl-uva/logdecomp/actions/workflows/build_wheels.yml/badge.svg)

# logdecomp: stable inverse and logdet in log domain.

Library for computing `inv(A)` (matrix inverse) and `log(abs(det(A)))` (signed log-determinant)
and their gradients, for matrices `X` of the form `a_ii = s_ij * exp(x_ij)`.

By Vlad Niculae `@vene` // licensed under BSD 2-clause.

## Usage example

```python
In [1]: import torch

In [2]: import logdecomp

In [3]: X = torch.randn(3, 3)

In [4]: logdecomp.logdetexp(X)
Out[4]: tensor(1.0835)

In [5]: X.exp().slogdet().logabsdet
Out[5]: tensor(1.0835)
```

## Installation

```bash
pip install logdecomp
```


## Building from source

Make sure you have Eigen installed. If it's in a non-standard directory, set
`EIGEN_DIR`.

```bash
pip install .
```

## Acknowledgements

Powered by Eigen FullPivLU decomposition with a custom log-domain datatype
originally by Chris Dyer [(gist)](https://gist.github.com/redpony/2400470)
with some modifications by André Martins and myself.


