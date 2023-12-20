import torch
from typing import Union, List
from data import Domain, Conditions, Equation

class Model():
    """class for preprocessing"""
    def __init__(
            self,
            net: Union[torch.nn.Module, torch.Tensor],
            domain: Domain,
            equation: Equation,
            conditions: Conditions):
        """
        Args:
            net (Union[torch.nn.Module, torch.Tensor]): neural network or torch.Tensor for mode *mat*
            grid (Domain): object of class Domain
            equation (Equation): object of class Equation
            conditions (Conditions): object of class Conditions
        """
        self.net = net
        self.domain = domain
        self.equation = equation
        self.conditions = conditions
    
    def compile(
            self,
            mode: str = 'autograd',
            problem: str = 'forward',
            loss: str = 'l2',
            h: float = None,
            derivative_points: int = None):
        