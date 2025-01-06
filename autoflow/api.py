import numpy as np
from .nodes import *

def variable(init_val, name = None):
    """
    defines a node in the computational graph representing a variable

    Parameters:
    ----------
    init_val: np.ndarray | Number
        the initial value of the variable
    name: String
        the name of the variable node
    """
    return Variable.create_node(init_val, name)

def constant(value, name = None):
    """
    defines a node in the computational graph representing a constant

    Parameters:
    ----------
    value: np.ndarray | Number
        initial value of the constant
    name: String
        the name of the constant node
    """
    return Constant.create_node(value, name)