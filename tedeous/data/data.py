"""module for working with initial data as: x,y,..t and available experiments data"""

from typing import List, Union
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

from device import device_type, check_device

device = device_type()

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
        if dtype == 'float32':
            dtype = torch.float32
        elif dtype == 'float64':
            dtype = torch.float64
        elif dtype == 'float16':
            dtype = torch.float16

        if isinstance(variable_set, torch.Tensor):
            variable_tensor = variable_set.to(dtype)
            self._variable_dict[variable_name] = variable_tensor
        else:
            if self.type == 'uniform':
                n_points = n_points + 1
                start, end = variable_set
                variable_tensor = torch.linspace(start, end, n_points, dtype=dtype)
                self._variable_dict[variable_name] = variable_tensor
    
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
    def __init__(self):
        self.conditions_lst = []

    def bnd_value_check(self, bnd, value=None):

        if isinstance(bnd, torch.Tensor):
            bnd = check_device(bnd)

        elif isinstance(bnd, list):
            for i, value in enumerate(bnd):
                if isinstance(value, torch.Tensor):
                    bnd[i] = check_device(value)

        if isinstance(value, torch.Tensor):
            value = check_device(value)

        if isinstance(value, float):
            value = torch.Tensor([value])
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
        bnd, value = self.bnd_value_check(bnd, value=value)
        
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
        bnd, value = self.bnd_value_check(bnd, value=value)

        try:
            var = operator[operator.keys()[0]]['var']
        except:
            var = 0

        self.conditions_lst.append({'bnd': bnd,
                                    'operator': None,
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
            var (int, optional): variable for system case and periodic dirichlet. Defaults to None.
        """
        bnd, value = self.bnd_value_check(bnd)

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

    def data(self, bnd, operator, value, var):
        pass

dirichlet({'x': 0., 't': [0,1]})