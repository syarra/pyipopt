#!/usr/bin/env python
""" PYLEVMAR, Python bindings to LEVMAR
Levenberg-Marquardt algorithm for (constrained) least-squares problems
"""
DOCLINES = __doc__.split("\n")

# build with: $ python setup.py build_ext --inplace
# clean with: # python setup.py clean --all
# see:
# http://www.scipy.org/Documentation/numpy_distutils
# http://docs.cython.org/docs/tutorial.html


import os
from distutils.core import setup, Extension
from distutils.core import Command
from numpy.distutils.misc_util import get_numpy_include_dirs

# ADAPT THIS TO FIT YOUR SYSTEM

extra_compile_args = ['-g']
include_dirs = [get_numpy_include_dirs()[0],'pyipopt/src','/home/MGI/syarre/Env/pyOpt-IPOPT/programs/Ipopt-3.9.2/build/include/coin']
library_dirs = ['pyipopt/src','/home/MGI/syarre/Env/pyOpt-IPOPT/lib','/home/MGI/syarre/Env/pyOpt-IPOPT/programs/Ipopt-3.9.2/build/lib/coin','/home/MGI/syarre/Env/pyOpt-IPOPT/programs/Ipopt-3.9.2/build/lib/coin/ThirdParty']
libraries = ['m','coinhsl','coinmetis','lapack','ipopt']

# PACKAGE INFORMATION
CLASSIFIERS = """\
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved
Programming Language :: C
Programming Language :: Python
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Linux
"""

NAME                = 'pyipopt'
MAINTAINER          = "Sebastian F. Walter"
MAINTAINER_EMAIL    = "sebastian.walter@gmail.com"
DESCRIPTION         = DOCLINES[0]
LONG_DESCRIPTION    = "\n".join(DOCLINES[2:])
URL                 = "http://github.com/b45ch1/pyipopt"
DOWNLOAD_URL        = "http://github.com/b45ch1/pyipopt"
LICENSE             = 'GPL'
CLASSIFIERS         = filter(None, CLASSIFIERS.split('\n'))
AUTHOR              = "Eric You Xu, Sebastian F. Walter"
AUTHOR_EMAIL        = "sebastian.walter@gmail.com"
PLATFORMS           = ["Linux"]
MAJOR               = 0
MINOR               = 1
MICRO               = 0
ISRELEASED          = False
VERSION             = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

# IT IS USUALLY NOT NECESSARY TO CHANGE ANTHING BELOW THIS POINT
# override default setup.py help output
import sys
if len(sys.argv) == 1:
    print """

    You didn't enter what to do!

    Options:
    1: build the extension with
    python setup.py build_ext --inplace

    2: remove generated files with
    python setup.py clean --all


    Remark: This is an override of the default behaviour of the distutils setup.
    """
    exit()

class clean(Command):
    """
    This class is used in numpy.distutils.core.setup.
    When $python setup.py clean is called, an instance of this class is created and then it's run method is called.
    """

    description = "Clean everything"
    user_options = [("all","a","the same")]

    def initialize_options(self):
        self.all = None

    def finalize_options(self):
        pass

    def run(self):
        import os
        os.system("rm -rf build")
        os.system("rm *.pyc")


def fullsplit(path, result=None):
    """
    Split a pathname into components (the opposite of os.path.join) in a
    platform-neutral way.
    """
    if result is None:
        result = []
    head, tail = os.path.split(path)
    if head == '':
        return [tail] + result
    if head == path:
        return result
    return fullsplit(head, [tail] + result)

# find all files that should be included
packages, data_files = [], []
for dirpath, dirnames, filenames in os.walk('pyipopt'):
    # Ignore dirnames that start with '.'
    for i, dirname in enumerate(dirnames):
        if dirname.startswith('.'): del dirnames[i]
    if '__init__.py' in filenames:
        packages.append('.'.join(fullsplit(dirpath)))
    elif filenames:
        data_files.append([dirpath, [os.path.join(dirpath, f) for f in filenames]])

options_dict = {}
options_dict.update({
'name':NAME,
'version':VERSION,
'description' :DESCRIPTION,
'long_description' : LONG_DESCRIPTION,
'license':LICENSE,
'author':AUTHOR,
'platforms':PLATFORMS,
'author_email': AUTHOR_EMAIL,
'url':URL,
'packages' :packages,
'ext_package' : 'pyipopt',
'ext_modules': [Extension('_pyipopt', ['pyipopt/src/callback.c', 'pyipopt/src/pyipopt.c' ],
                include_dirs = include_dirs,
                library_dirs = library_dirs,
                runtime_library_dirs = library_dirs,
                libraries = libraries),
],

'cmdclass' : {'clean':clean}
})

setup(**options_dict)
                                         
