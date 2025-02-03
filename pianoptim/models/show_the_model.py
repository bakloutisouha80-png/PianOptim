import numpy as np
from pyorerun import LiveModelAnimation

model_path = "pianist_and_key.bioMod"
lma = LiveModelAnimation(model_path)
# lma.q = np.array(
#     [
#         [0.00792227],
#         [-0.13197982],
#         [0.24771835],
#         [-0.20400101],
#         [0.26070489],
#         [-0.37418803],
#         [0.288201],
#         [-0.04343295],
#         [-0.21240923],
#         [-0.00484653],
#         [-0.06921613],
#         [-0.02205693],
#         [0],
#     ]
# ).squeeze()
lma.rerun("the_pianist")
