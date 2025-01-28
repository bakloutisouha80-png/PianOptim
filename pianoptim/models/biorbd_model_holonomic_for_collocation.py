from typing import Callable, Any, override

import biorbd_casadi as biorbd
from bioptim import HolonomicBiorbdModel, ParameterList


class HolonomicBiorbdModelForCollocation(HolonomicBiorbdModel):
    """
    This class allows to define a biorbd model with custom holonomic constraints.
    """

    def __init__(
        self,
        bio_model: str | biorbd.Model,
        parameters: ParameterList = None,
    ):
        super().__init__(bio_model, parameters=parameters)
