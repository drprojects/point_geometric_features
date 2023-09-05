# ----------------------------------------------------------------------#
#       distutils setup script for compiling pgeof extensions          #
# ----------------------------------------------------------------------#
""" 
Compilation command: python setup.py build_ext

Loic Landrieu 2023
"""

from distutils.core import setup, Extension
from distutils.command.build import build
from distutils.ccompiler import new_compiler
import numpy
import shutil  # for rmtree, os.rmdir can only remove _empty_ directory
import os
import re

###  targets and compile options  ###
name = "pgeof"

include_dirs = [numpy.get_include()]  # find the Numpy headers

# compilation and linkage options
# MIN_OPS_PER_THREAD roughly controls parallelization, see doc in README.md
if os.name == 'nt':  # windows
    extra_compile_args = ["/DPGEOF_WINDOWS"]
    extra_link_args = []
elif os.name == 'posix':  # linux
    extra_compile_args = ["-std=c++11", "-fopenmp"]
    extra_link_args = ["-lgomp"]
else:
    raise NotImplementedError('OS not yet supported.')


###  auxiliary functions  ###
class build_class(build):
    def initialize_options(self):
        build.initialize_options(self)
        self.build_lib = "bin"

    def run(self):
        build_path = self.build_lib


def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))


###  preprocessing  ###
# ensure right working directory
tmp_work_dir = os.path.realpath(os.curdir)
os.chdir(os.path.realpath(os.path.dirname(__file__)))

if not os.path.exists("bin"):
    os.mkdir("bin")

# remove previously compiled lib
purge("bin/", "pgeof")

###  compilation  ###
mod = Extension(
    name,
    # list source files
    ["./pgeof_cpy.cpp"],
    include_dirs=include_dirs,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args)

setup(name=name, ext_modules=[mod], cmdclass=dict(build=build_class))

###  postprocessing  ###
try:
    shutil.rmtree("build")  # remove temporary compilation products
except FileNotFoundError:
    pass

os.chdir(tmp_work_dir)  # get back to initial working directory
