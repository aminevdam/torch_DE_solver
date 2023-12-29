"""module for working with inerface for initialize grid, conditions and equation"""

from typing import List, Union
import torch
import numpy as np
import sys
import os

from tedeous.device import check_device
from tedeous.input_preprocessing import EquationMixin


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
        self.variable_dict = {}
    
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
        var_lst = list(self.variable_dict.values())
        if mode in ('autograd', 'NN'):
            if len(self.variable_dict) == 1:
                grid = check_device(var_lst[0].reshape(-1, 1))
            else:
                grid = check_device(torch.cartesian_prod(*var_lst))
        else:
            grid = np.meshgrid(*var_lst, indexing='ij')
            grid = check_device(torch.tensor(np.array(grid)))

        return grid


class Conditions():
    """class for adding the conditions: initial, boundary, and data.
    """
    def __init__(self):
        self.conditions_lst = []

    def dirichlet(
            self,
            bnd: dict,
            value: Union[callable, torch.Tensor, float],
            var: int = 0):
        """ determine dirichlet boundary condition.

        Args:
            bnd (Union[torch.Tensor, dict]): boundary points can be torch.Tensor
            or dict with keys as coordinates names and values as coordinates values or list(start, end)
            value (Union[callable, torch.Tensor, float]): values at the boundary (bnd)
            var (int, optional): variable for system case, for single equation is 0. Defaults to 0.
        """

        self.conditions_lst.append({'bnd': bnd,
                                    'bop': None,
                                    'bval': value,
                                    'var': var,
                                    'type': 'dirichlet'})

    def operator(self,
                 bnd: dict,
                 operator: dict,
                 value: Union[callable, torch.Tensor, float]):
        """ determine operator boundary condition

        Args:
            bnd (Union[torch.Tensor, dict]): boundary points can be torch.Tensor
            or dict with keys as coordinates names and values as coordinates values or list(start, end)
            operator (dict): dictionary with opertor terms: {'operator name':{coeff, term, pow, var}}
            value (Union[callable, torch.Tensor, float]): value on the boundary (bnd).
        """
        try:
            var = operator[operator.keys()[0]]['var']
        except:
            var = 0
        operator = EquationMixin.equation_unify(operator)
        self.conditions_lst.append({'bnd': bnd,
                                    'bop': operator,
                                    'bval': value,
                                    'var': var,
                                    'type': 'operator'})

    def periodic(self,
                 bnd: List[dict],
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
        value = torch.tensor([0.])
        if operator is None:
            self.conditions_lst.append({'bnd': bnd,
                                        'bop': operator,
                                        'bval': value,
                                        'var': var,
                                        'type': 'periodic'})
        else:
            try:
                var = operator[operator.keys()[0]]['var']
            except:
                var = 0
            operator = EquationMixin.equation_unify(operator)
            self.conditions_lst.append({'bnd': bnd,
                                        'bop': operator,
                                        'bval': value,
                                        'var': var,
                                        'type': 'periodic'})

    def data(
        self,
        bnd: dict,
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
        if operator is not None:
            operator = EquationMixin.equation_unify(operator)
        self.conditions_lst.append({'bnd': bnd,
                                    'bop': operator,
                                    'bval': value,
                                    'var': var,
                                    'type': 'data'})

    def _bnd_grid(self, bnd, variable_dict, dtype):
        dtype = variable_dict[list(variable_dict.keys())[0]].dtype
        var_lst = []
        for var in variable_dict.keys():
            if isinstance(bnd[var], torch.Tensor):
                var_lst.append(check_device(bnd[var]).to(dtype))
            elif isinstance(bnd[var], (float, int)):
                var_lst.append(check_device(torch.tensor([bnd[var]])).to(dtype))
            elif isinstance(bnd[var], list):
                lower_bnd = bnd[var][0]
                upper_bnd = bnd[var][1]
                grid_var = variable_dict[var]
                bnd_var = grid_var[(grid_var >= lower_bnd) & (grid_var <= upper_bnd)]
                var_lst.append(check_device(bnd_var).to(dtype))
        bnd_grid = torch.cartesian_prod(*var_lst).to(dtype)
        return bnd_grid

    def build(self, variable_dict):
        if self.conditions_lst == []:
            return None

        try:
            dtype = variable_dict[list(variable_dict.keys())[0]].dtype
        except:
            dtype = variable_dict[list(variable_dict.keys())[0]][0].dtype # if periodic

        for cond in self.conditions_lst:
            if cond['type'] == 'periodic':
                cond_lst = []
                for bnd in cond['bnd']:
                    cond_lst.append(self._bnd_grid(bnd, variable_dict, dtype))
                cond['bnd'] = cond_lst
            else:
                cond['bnd'] = self._bnd_grid(cond['bnd'], variable_dict, dtype)
            
            if isinstance(cond['bval'], torch.Tensor):
                cond['bval'] = check_device(cond['bval']).to(dtype)
            elif isinstance(cond['bval'], (float, int)):
                cond['bval'] = check_device(
                    torch.ones_like(cond['bnd'][:,0])*cond['bval']).to(dtype)
            elif isinstance(cond['bval'], callable):
                cond['bval'] = check_device(cond['bval'](cond['bnd'])).to(dtype)
        
        return self.conditions_lst


class Equation():
    """class for adding eqution.
    """
    def __init__(self):
        self.equation_lst = []
    
    def add(self, eq: dict):
        """ add equation

        Args:
            eq (dict): equation in operator form.
        """
        self.equation_lst.append(eq)
