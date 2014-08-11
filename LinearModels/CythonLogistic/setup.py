from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
  name = 'Multilogistic Regression',
  include_dirs = [np.get_include()], 
  ext_modules = cythonize("multiLogisticRegression.pyx"),
)