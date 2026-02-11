"""Functions to produce JSA/moments for Frequency Conversion with variable domain configs """

from itertools import product
import numpy as np
from scipy.linalg import expm

# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=consider-using-enumerate