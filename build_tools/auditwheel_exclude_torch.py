# -*- coding: utf-8 -*-
import re
import sys

from auditwheel.main import main
from auditwheel.policy import _POLICIES as POLICIES

# libjvm is loaded dynamically; do not include it
for p in POLICIES:
    if p['name'].startswith('manylinux'):
        p['lib_whitelist'].extend(['libtorch.so', 'libc10.so', 'libtorch_cpu.so', 'libtorch_python.so'])

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(main())
