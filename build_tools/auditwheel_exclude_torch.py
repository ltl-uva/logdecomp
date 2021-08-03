"""Monkey-patch auditwheel policies to allow torch libraries.

Idea simplified from the insipration source:
<https://github.com/DIPlib/diplib/blob/bca5333236f42e02046db8684af61173f200086e/tools/travis/auditwheel>
"""

import re
import sys

from auditwheel.main import main
from auditwheel.policy import _POLICIES as POLICIES

for p in POLICIES:
    if p['name'].startswith('manylinux'):
        p['lib_whitelist'].extend(['libtorch.so', 'libc10.so', 'libtorch_cpu.so', 'libtorch_python.so'])

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(main())
