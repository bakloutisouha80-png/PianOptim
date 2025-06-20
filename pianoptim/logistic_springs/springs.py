from functools import partial
import pandas as pd
from .utils import model_exponential_decay, model_cubic

import os

local_path = os.path.dirname(os.path.abspath(__file__))

IDENTIFICATION_RESULTS = pd.read_csv(local_path + "/identification_results_45.csv", index_col=0).to_dict()["value"]

SPRING_FUNCTION_EXPONENTIAL_DECAY = partial(
    model_exponential_decay,
    params=[IDENTIFICATION_RESULTS["A"], IDENTIFICATION_RESULTS["k"], IDENTIFICATION_RESULTS["C"]],
)
SPRING_FUNCTION_CUBIC_INCREASE = partial(
    model_cubic,
    params=[
        IDENTIFICATION_RESULTS["a"],
        IDENTIFICATION_RESULTS["b"],
        IDENTIFICATION_RESULTS["c"],
        IDENTIFICATION_RESULTS["d"],
    ],
)
