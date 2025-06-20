from functools import cached_property

from bioptim import Bounds, HolonomicConstraintsList, HolonomicConstraintsFcn
from casadi import MX, SX, vertcat, if_else, nlpsol, DM, Function
import numpy as np

from .pianist_holonomic import HolonomicPianist


class HolonomicPianistWithSpring(HolonomicPianist):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.spring = dict()

    def add_spring(self, function: callable, min_value: float, max_value: float):
        """
        Add a spring to the model

        Parameters
        ----------
        spring: dict
            The spring to add to the model
        """
        self.spring = {
            "function": function,
            "min_value": min_value,
            "max_value": max_value,
        }

    def compute_spring_force(self, q: np.array, qdot: np.array) -> np.array:
        """
        Compute the spring force

        Parameters
        ----------
        q: np.array
            The generalized coordinates
        qdot: np.array
            The generalized velocities

        Returns
        -------
        np.array
            The spring force
        """
        q_spring = q[-1]
        return self.spring["function"](q_spring)
