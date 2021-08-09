import os
import re
import warnings
from setuptools import setup
import sys
import setuptools

from pybind11.setup_helpers import Pybind11Extension, WIN


__version__ = '0.2'


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
        "C:\ProgramData\chocolatey\lib\eigen\include"
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


CPP_VER = '/std:c++17' if WIN else '-std=c++17'


extensions = [
    Pybind11Extension('logdecomp.lu',
                      ["logdecomp/lu.cpp"],
                      include_dirs=[find_eigen()],
                      extra_compile_args=[CPP_VER]
    )
]


setup(name='logdecomp',
      version=__version__,
      author="Vlad Niculae",
      author_email="vlad@vene.ro",
      url="https://github.com/ltl-uva/logdecomp/",
      ext_modules=extensions,
      install_requires=["torch>=1.8.1", "numpy"],
      packages=['logdecomp'],
      setup_requires=['pybind11>=2.5.0'],
      zip_safe=False
)
