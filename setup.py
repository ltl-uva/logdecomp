import os
import re
import warnings
from setuptools import setup
import sys
import setuptools

from torch.utils import cpp_extension


__version__ = '0.1'


# thanks to https://github.com/dfm/transit/blob/master/setup.py
def find_eigen():
    """
    Find the location of the Eigen 3 include directory. This will return
    ``None`` on failure.
    """
    # List the standard locations including a user supplied hint.
    search_dirs = []
    env_var = os.getenv("EIGEN_DIR")
    if env_var:
        search_dirs.append(env_var)
    search_dirs += [
        "/usr/local/include/eigen3",
        "/usr/local/homebrew/include/eigen3",
        "/opt/local/var/macports/software/eigen3",
        "/opt/local/include/eigen3",
        "/usr/include/eigen3",
        "/usr/include/local",
        "/usr/include",
    ]

    # Loop over search paths and check for the existence of the Eigen/Dense
    # header.
    for d in search_dirs:
        path = os.path.join(d, "Eigen", "Dense")
        if os.path.exists(path):
            # Determine the version.
            vf = os.path.join(d, "Eigen", "src", "Core", "util", "Macros.h")
            if not os.path.exists(vf):
                continue
            src = open(vf, "r").read()
            v1 = re.findall("#define EIGEN_WORLD_VERSION (.+)", src)
            v2 = re.findall("#define EIGEN_MAJOR_VERSION (.+)", src)
            v3 = re.findall("#define EIGEN_MINOR_VERSION (.+)", src)
            if not len(v1) or not len(v2) or not len(v3):
                continue
            v = "{0}.{1}.{2}".format(v1[0], v2[0], v3[0])
            print("Found Eigen version {0} in: {1}".format(v, d))
            return d
    warnings.warn("Could not find eigen. Please set EIGEN_DIR.")
    return None


extensions = [
    cpp_extension.CppExtension('logdecomp.lu',
              ["logdecomp/lu.cpp"],
              language='c++',
              include_dirs=[find_eigen()],
              extra_compile_args=['-std=c++17'],
    ),
]

setup(name='logdecomp',
      version=__version__,
      author="Vlad Niculae",
      ext_modules=extensions,
      setup_requires=['pybind11>=2.5.0'],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      zip_safe=False
)