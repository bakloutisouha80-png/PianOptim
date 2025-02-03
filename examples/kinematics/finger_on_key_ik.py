from pianoptim.models.pianist_holonomic import HolonomicPianist
from pyorerun import PhaseRerun, BiorbdModel
import numpy as np
from casadi import MX, nlpsol, vertcat

model_path = "../../pianoptim/models/pianist_and_key.bioMod"
model = HolonomicPianist(model_path)

q = MX.sym("q", model.nb_q, 1)

target = model.marker(model.marker_names.index("Key1_Top"), None)(q, model.parameters)
finger = model.marker(model.marker_names.index("finger_marker"), None)(q, model.parameters)
mcp = model.marker(model.marker_names.index("MCP_marker"), None)(q, model.parameters)

g = finger - target
g = vertcat(g, target[0] - mcp[0])

segment_names = tuple([s.name().to_string() for s in model.model.segments()])
idx_arm = segment_names.index("RightUpperArm")
H_arm = model.homogeneous_matrices_in_global(idx_arm)(q, model.parameters)
idx_pelvis = segment_names.index("Pelvis")
H_pelvis = model.homogeneous_matrices_in_global(idx_pelvis)(q, model.parameters)

f = 0
f += 100 * (q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2 + q[4] ** 2)
f += 10 * (q[10] ** 2 + q[11] ** 2)
f += q[6] ** 2 + q[7] ** 2

f += 0.001 * (q[5] + 30 * np.pi / 180) ** 2
f += 0.001 * (q[8] + 105 * np.pi / 180) ** 2  #

q_ranges = model.ranges_from_model("q")
q_min = [q_ranges[i].min() for i in range(model.nb_q)]
q_max = [q_ranges[i].max() for i in range(model.nb_q)]

s = nlpsol(
    "sol",
    "ipopt",
    {"x": q, "f": f, "g": g},
)
output = s(x0=np.zeros(model.nb_q), lbg=np.zeros(4), ubg=np.zeros(4), lbx=q_min, ubx=q_max)
q0 = np.array(output["x"])


g = vertcat(g, target[1] - mcp[1])

f = 0
for i in [6, 7, 8, 9, 10]:
    f += (q[i] - q0[i]) ** 2

s = nlpsol(
    "sol",
    "ipopt",
    {"x": q, "f": f, "g": g},
)
output = s(x0=q0, lbg=np.zeros(5), ubg=np.zeros(5), lbx=q_min, ubx=q_max)

q1 = np.array(output["x"])
q = np.concatenate((q0, q1), axis=1)

prr = PhaseRerun(t_span=np.array([0, 1]))
prr_model = BiorbdModel(model_path)
prr.add_animated_model(prr_model, q)
prr.rerun()

print(q)
