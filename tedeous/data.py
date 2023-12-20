"""module for working with inerface for initialize grid, conditions and equation"""

from typing import List, Union
import torch
import numpy as np
import sys
import os

from device import device_type, check_device

device = device_type()

def tensor_dtype(dtype: str):
    """convert tensor to dtype format

    Args:
        dtype (str): dtype

    Returns:
        dtype: torch.dtype
    """
    if dtype == 'float32':
        dtype = torch.float32
    elif dtype == 'float64':
        dtype = torch.float64
    elif dtype == 'float16':
        dtype = torch.float16

    return dtype

class Domain():
    """class for grid building
    """
    def __init__(self, type='uniform'):
        self.type = type
        self._variable_dict = {}
    
    def variable(
            self,
            variable_name: str,
            variable_set: Union[List, torch.Tensor],
            n_points: Union[None, int],
            dtype: str = 'float32') -> None:
        """ determine varibles for grid building.

        Args:
            varible_name (str): varible name.
            spatial_variable (List): [start, stop] list for spatial variable.
            n_points (int): number of points in discretization for variable.
            dtype (str, optional): dtype of result vector. Defaults to 'float32'.

        """
        dtype = tensor_dtype(dtype)

        if isinstance(variable_set, torch.Tensor):
            variable_tensor = check_device(variable_tensor)
            variable_tensor = variable_set.to(dtype)
            self.variable_dict[variable_name] = variable_tensor
        else:
            if self.type == 'uniform':
                n_points = n_points + 1
                start, end = variable_set
                variable_tensor = torch.linspace(start, end, n_points, dtype=dtype)
                self.variable_dict[variable_name] = variable_tensor
    
    def build(self, mode: str) -> torch.Tensor:
        """ building the grid for algorithm

        Args:
            mode (str): mode for equation solution, *mat, autograd, NN*

        Returns:
            torch.Tensor: resulting grid.
        """
        var_lst = list(self._variable_dict.values())
        if mode in ('autograd', 'NN'):
            if len(self._variable_dict) == 1:
                grid = var_lst[0].reshape(-1, 1).to(device)
            else:
                grid = torch.cartesian_prod(*var_lst).to(device)
        else:
            grid = np.meshgrid(*var_lst, indexing='ij')
            grid = torch.tensor(np.array(grid)).to(device)

        return grid

class Conditions():
    """class for adding the conditions: initial, boundary, and data.
    """
    def __init__(self):
        self.conditions_lst = []

    def _bnd_value_check(
            self,
            bnd,
            value = None):
        """ check device and tensor format for inputs.

        Args:
            bnd: variants of bnd for all conditions
            value:conditions value. Defaults to None.

        Returns:
            tuple(bnd, value): checked bnd and value
        """

        if isinstance(bnd, torch.Tensor):
            bnd = check_device(bnd)

        elif isinstance(bnd, list):
            for i, val in enumerate(bnd):
                if isinstance(val, torch.Tensor):
                    bnd[i] = check_device(val)

        if isinstance(value, torch.Tensor):
            value = check_device(value)

        if isinstance(value, (float, int)):
            value = torch.tensor([value])
            value = check_device(value)

        return bnd, value

    def dirichlet(
            self,
            bnd: Union[torch.Tensor, dict],
            value: Union[callable, torch.Tensor, float],
            var: int = 0):
        """ determine dirichlet boundary condition.

        Args:
            bnd (Union[torch.Tensor, dict]): boundary points can be torch.Tensor
            or dict with keys as coordinates names and values as coordinates values or list(start, end)
            value (Union[callable, torch.Tensor, float]): values at the boundary (bnd)
            var (int, optional): variable for system case, for single equation is 0. Defaults to 0.
        """
        bnd, value = self._bnd_value_check(bnd, value=value)
        
        self.conditions_lst.append({'bnd': bnd,
                                    'operator': None,
                                    'value': value,
                                    'var': var,
                                    'type': 'dirichlet'})

    def operator(self,
                 bnd: Union[torch.Tensor, dict],
                 operator: dict,
                 value: Union[callable, torch.Tensor, float]):
        """ determine operator boundary condition

        Args:
            bnd (Union[torch.Tensor, dict]): boundary points can be torch.Tensor
            or dict with keys as coordinates names and values as coordinates values or list(start, end)
            operator (dict): dictionary with opertor terms: {'operator name':{coeff, term, pow, var}}
            value (Union[callable, torch.Tensor, float]): value on the boundary (bnd).
        """
        bnd, value = self._bnd_value_check(bnd, value=value)

        try:
            var = operator[operator.keys()[0]]['var']
        except:
            var = 0

        self.conditions_lst.append({'bnd': bnd,
                                    'operator': operator,
                                    'value': value,
                                    'var': var,
                                    'type': 'operator'})

    def periodic(self,
                 bnd: Union[List[torch.Tensor], List[dict]],
                 operator: dict = None,
                 var: int = 0):
        """Periodic can be: periodic dirichlet (example u(x,t)=u(-x,t))
        if form with bnd and var for system case.
        or periodic operator (example du(x,t)/dx=du(-x,t)/dx)
        in form with bnd and operator.
        Parameter 'bnd' is list: [b_coord1:torch.Tensor, b_coord2:torch.Tensor,..] or
        bnd = [{'x': 1, 't': [0,1]},{'x': -1, 't':[0,1]}]

        Args:
            bnd (Union[List[torch.Tensor], List[dict]]): list with dicionaries or torch.Tensors
            operator (dict, optional): operator dict. Defaults to None.
            var (int, optional): variable for system case and periodic dirichlet. Defaults to 0.
        """
        bnd, value = self._bnd_value_check(bnd)

        if operator is None:
            self.conditions_lst.append({'bnd': bnd,
                                        'operator': operator,
                                        'value': value,
                                        'var': var,
                                        'type': 'periodic'})
        else:
            try:
                var = operator[operator.keys()[0]]['var']
            except:
                var = 0
            self.conditions_lst.append({'bnd': bnd,
                                        'operator': operator,
                                        'value': value,
                                        'var': var,
                                        'type': 'periodic'})

    def data(
        self,
        bnd: Union[torch.Tensor, dict],
        operator: Union[dict, None],
        value: torch.Tensor,
        var: int = 0):
        """ conditions for available solution data

        Args:
            bnd (Union[torch.Tensor, dict]): boundary points can be torch.Tensor
            or dict with keys as coordinates names and values as coordinates values or list(start, end)
            operator (Union[dict, None]): dictionary with opertor terms: {'operator name':{coeff, term, pow, var}}
            value (Union[torch.Tensor, float]): values at the boundary (bnd)
            var (int, optional): variable for system case and periodic dirichlet. Defaults to 0.
        """

        bnd, value = self._bnd_value_check(bnd, value=value)
        
        self.conditions_lst.append({'bnd': bnd,
                                    'operator': operator,
                                    'value': value,
                                    'var': var,
                                    'type': 'data'})

    def _build_one_bnd(self, cond, grid, var_lst):
        dtype = grid.dtype
        if isinstance(cond['bnd'], torch.Tensor):
            cond['bnd'] = cond['bnd'].to(dtype)
        elif isinstance(cond['bnd'], dict):
            
            


                    


class Equation():
    """class for adding eqution.
    """
    def __init__(self):
        self.equation_lst = []
    
    def equation(self, eq: dict):
        """ add equation

        Args:
            eq (dict): equation in operator form.
        """
        self.equation_lst.append(eq)

# domain = Domain()

# domain.variable('x', [0,1], 10)
# domain.variable('t', [1,2], 5)

# grid = domain.build('NN')

# boundaries = Conditions()

# bop= {
#         'du/dx':
#             {
#                 'coeff': 1,
#                 'du/dx': [0],
#                 'pow': 1,
#                 'var': 0
#             }
# }

# # boundaries.operator({'x':0, 't': [0,1]}, operator=bop, value=5)
# boundaries.periodic([{'x':0, 't':[0,1]}, {'x':1, 't':[0,1]}], bop)

# print(boundaries.conditions_lst)