"""
This is a multiphase problem where the pianist is playing a note on the piano.
The play style is a press play where the finger is placed on the key and the key is pressed.

So this is a three-phase problem:
- Phase 0: The finger is placed on the key and goes down
- Phase 1: The key is actively pressed into the bed
- Phase 2: The key is released, from the bed and the finger is lifted up to the top position
"""

from bioptim import (
    ObjectiveList,
    DynamicsList,
    BoundsList,
    InitialGuessList,
    CostType,
    Node,
    OptimalControlProgram,
    ObjectiveFcn,
    ConstraintFcn,
    ConstraintList,
    OdeSolver,
    Solver,
    BiMappingList,
    TimeAlignment,
    SolutionMerge,
    OnlineOptim,
    MultinodeConstraintList, BiorbdModel,
    PhaseTransitionList,
)
from bioptim import DynamicsFcn

from pianoptim.models.pianist_holonomic import HolonomicPianist
from pianoptim.models.constant import (
    FINGER_TIP_ON_KEY_RELAXED, KEY_TOP_PRESSED, KEY_TOP_UNPRESSED, ELEVATED_FINGER_TIP)
from pianoptim.utils.custom_functions import (
    custom_func_track_markers,
    custom_func_track_markers_velocity,
    custom_contraint_lambdas,
)
from pianoptim.utils.custom_transitions import custom_phase_transition_algebraic_post

import numpy as np

from pianoptim.utils.torque_derivative_holonomic_driven import (
    configure_holonomic_torque_derivative_driven_with_qv,
    holonomic_torque_derivative_driven_with_qv,
    constraint_holonomic,
    constraint_holonomic_end,
)


def prepare_ocp(
    model_path: str,
    n_shootings: tuple[int, ...],
    min_phase_times: tuple[float, ...],
    max_phase_times: tuple[float, ...],
    ode_solver: OdeSolver,
) -> OptimalControlProgram:

    dynamics = DynamicsList()
    x_bounds = BoundsList()
    x_init = InitialGuessList()
    a_bounds = BoundsList()
    a_init = InitialGuessList()
    u_bounds = BoundsList()
    u_init = InitialGuessList()
    objective_functions = ObjectiveList()
    constraints = ConstraintList()
    multinode_constraints = MultinodeConstraintList()
    phase_transitions = PhaseTransitionList()

    # Load and constraints the dynamic model
    models = (
        HolonomicPianist(model_path),
        HolonomicPianist(model_path),
        HolonomicPianist(model_path),
        BiorbdModel("../../pianoptim/models/pianist.bioMod"),
    )
    first_model = models[0]

    u_variable_bimapping = BiMappingList()
    u_variable_bimapping.add(
        "q", to_second=[0, 1, 2, 3, 4, None, None, 5, 6, 7, 8, 9, None], to_first=[0, 1, 2, 3, 4, 7, 8, 9, 10, 11]
    )
    u_variable_bimapping.add(
        "qdot", to_second=[0, 1, 2, 3, 4, None, None, 5, 6, 7, 8, 9, None], to_first=[0, 1, 2, 3, 4, 7, 8, 9, 10, 11]
    )

    v_variable_bimapping = BiMappingList()
    v_variable_bimapping.add(
        "q", to_second=[None, None, None, None, None, 0, 1, None, None, None, None, None, 3], to_first=[5, 6, 12]
    )

    dof_mapping = BiMappingList()
    for p in range(3):
        dof_mapping.add(
            "tau",
            to_second=[i for i in range(first_model.nb_q - 1)] + [None],
            to_first=[i for i in range(first_model.nb_q - 1)],
            phase=p,
        )
        dof_mapping.add(
            "taudot",
            to_second=[i for i in range(first_model.nb_q - 1)] + [None],
            to_first=[i for i in range(first_model.nb_q - 1)],
            phase=p,
        )

    q = FINGER_TIP_ON_KEY_RELAXED
    qu = q[[0, 1, 2, 3, 4, 7, 8, 9, 10, 11]]
    qv = q[[5, 6, 12]]

    for i in range(3):
        dynamics.add(
            configure_holonomic_torque_derivative_driven_with_qv,
            dynamic_function=holonomic_torque_derivative_driven_with_qv,
            custom_q_v_init=qv,
            phase=i,
        )
        # Path Constraints
        constraints.add(
            constraint_holonomic,
            node=Node.ALL_SHOOTING,
            phase=i,
        )
        constraints.add(
            constraint_holonomic_end,
            node=Node.END,
            phase=i,
        )
    # for i in range(1, 3):
    #     multinode_constraints.add(
    #         algebraic_continuity,
    #         nodes=(Node.END, Node.START),
    #         nodes_phase=(i - 1, i),
    #     )

    for p in range(3):
        x_bounds.add("q_u", bounds=models[0].bounds_from_ranges("q", u_variable_bimapping), phase=p)
        x_bounds.add("qdot_u", bounds=models[0].bounds_from_ranges("qdot", u_variable_bimapping), phase=p)

        a_bounds.add("q_v", bounds=models[0].bounds_from_ranges("q", v_variable_bimapping), phase=p)

        x_init.add("q_u", qu, phase=p)

        x_init.add("qdot_u", [0] * (models[0].nb_q - 3), phase=p)

        a_init.add("q_v", qv, phase=p)

        x_bounds.add(
            "tau",
            min_bound=[-40] * (models[0].nb_tau - 1),
            max_bound=[40] * (models[0].nb_tau - 1),
            phase=p,
        )

        x_init.add("tau", [0] * (models[0].nb_tau - 1), phase=p)

        u_bounds.add(
            "taudot", min_bound=[-10000] * (models[0].nb_tau - 1), max_bound=[10000] * (models[0].nb_tau - 1), phase=p
        )

        u_init.add("taudot", [0] * (models[0].nb_tau - 1), phase=p)

    # Objective Functions
    elbow_wrist_idx = [8, 10]
    no_elbow_wrist_idx = [i for i in range(12) if i not in elbow_wrist_idx]

    for p in range(3):
        # reduce the torque variation on all joints except elbow and wrist
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
            key="taudot",
            phase=p,
            weight=1,
            index=no_elbow_wrist_idx,
        )
        # reduce the torque on all joints
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="tau", phase=p, weight=0.1)
        # dont generate transverse forces along medial-lateral axis
        objective_functions.add(
            custom_contraint_lambdas,
            custom_type=ObjectiveFcn.Lagrange,
            index=[2],
            phase=p,
            weight=0.1,
            custom_qv_init=qv,
        )
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot_u", phase=1, weight=0.001)

    constraints.add(
        custom_func_track_markers,
        phase=0,
        node=Node.START,
        marker="contact_finger",
        target=KEY_TOP_UNPRESSED,
        custom_qv_init=qv,
    )
    for i in range(0, 3):
        constraints.add(
            custom_contraint_lambdas,
            phase=i,
            node=Node.ALL,
            custom_qv_init=qv,
            min_bound=[-20, -20, -20],
            max_bound=[20, 20, 20],
        )

    constraints.add(
        custom_func_track_markers,
        phase=0,
        node=Node.INTERMEDIATES,
        marker="contact_finger",
        custom_qv_init=qv,
        min_bound=KEY_TOP_PRESSED,
        max_bound=KEY_TOP_UNPRESSED,
    )

    constraints.add(
        custom_func_track_markers,
        phase=1,
        node=Node.START,
        marker="contact_finger",
        # target=np.tile(BED_POSE, (1, n_shootings[1] + 1)),
        target=KEY_TOP_PRESSED,
        custom_qv_init=qv,
        # min_bound=np.tile(-slack, (1, n_shootings[1] + 1)),
        # max_bound=np.tile(+slack, (1, n_shootings[1] + 1)),
    )
    constraints.add(
        custom_func_track_markers_velocity,
        phase=1,
        node=Node.ALL,
        marker="contact_finger",
        custom_qv_init=qv,
    )

    constraints.add(
        custom_func_track_markers,
        phase=1,
        node=Node.END,
        marker="contact_finger",
        # target=np.tile(BED_POSE, (1, n_shootings[1] + 1)),
        target=KEY_TOP_PRESSED,
        custom_qv_init=qv,
        # min_bound=np.tile(-slack, (1, n_shootings[1] + 1)),
        # max_bound=np.tile(+slack, (1, n_shootings[1] + 1)),
    )

    constraints.add(
        custom_func_track_markers,
        phase=2,
        node=Node.ALL_SHOOTING,
        marker="contact_finger",
        custom_qv_init=qv,
        min_bound=KEY_TOP_PRESSED,
        max_bound=KEY_TOP_UNPRESSED,
    )

    constraints.add(
        custom_func_track_markers,
        phase=2,
        node=Node.END,
        marker="contact_finger",
        target=KEY_TOP_UNPRESSED,
        custom_qv_init=qv,
    )

    phase_times = [(min_t + max_t) / 2 for min_t, max_t in zip(min_phase_times, max_phase_times)]

    dynamics.add(
        DynamicsFcn.TORQUE_DERIVATIVE_DRIVEN,
        phase=3,
    )

    # fake mapping to make the OCP not crash
    dof_mapping.add(
        "tau",
        to_second=[i for i in range(models[-1].nb_q)],
        to_first=[i for i in range(models[-1].nb_q)],
        phase=3,
    )
    dof_mapping.add(
        "taudot",
        to_second=[i for i in range(models[-1].nb_q)],
        to_first=[i for i in range(models[-1].nb_q)],
        phase=3,
    )

    x_bounds.add("q", bounds=models[3].bounds_from_ranges("q"), phase=3)
    x_bounds.add("qdot", bounds=models[3].bounds_from_ranges("qdot"), phase=3)
    x_init.add("q", q[:-1], phase=3)
    x_init.add("qdot", [0] * (models[3].nb_q), phase=3)

    x_bounds.add(
        "tau",
        min_bound=[-40] * (models[3].nb_tau),
        max_bound=[40] * (models[3].nb_tau),
        phase=3,
    )
    x_init.add("tau", [0] * (models[3].nb_tau), phase=3)

    u_bounds.add(
        "taudot",
        min_bound=[-10000] * (models[3].nb_tau),
        max_bound=[10000] * (models[3].nb_tau),
        phase=3,
    )
    u_init.add("taudot", [0] * (models[3].nb_tau), phase=3)

    phase_transitions.add(
        custom_phase_transition_algebraic_post,
        phase_pre_idx=2,
    )

    # extra objectives
    p = 3
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
        key="taudot",
        phase=p,
        weight=1,
        index=no_elbow_wrist_idx,
    )
    # reduce the torque on all joints
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="tau", phase=3, weight=0.1)

    shoulder_non_flexion_dof = [7, 6]
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", phase=3, weight=1, index=shoulder_non_flexion_dof
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=3, weight=0.1,
                            index=shoulder_non_flexion_dof
    )

    constraints.add(
        ConstraintFcn.TRACK_MARKERS,
        phase=3,
        node=Node.ALL_SHOOTING,
        marker_index="contact_finger",
        min_bound=KEY_TOP_UNPRESSED,  # ;qke sure bound only x direction todo tomorrow
        max_bound=ELEVATED_FINGER_TIP,
        index=0
    )

    constraints.add(
        ConstraintFcn.TRACK_MARKERS,
        phase=3,
        node=Node.END,
        marker_index="contact_finger",
        target=ELEVATED_FINGER_TIP,
    )

    # The first and last frames are at rest
    x_bounds[0]["qdot_u"][:, 0] = 0
    x_bounds[-1]["qdot"].min[:, -1] = -0.1
    x_bounds[-1]["qdot"].max[:, -1] = 0.1


    # Prepare the optimal control program
    ocp = OptimalControlProgram(
        bio_model=models,
        dynamics=dynamics,
        n_shooting=n_shootings,
        phase_time=phase_times,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        a_bounds=a_bounds,
        x_init=x_init,
        u_init=u_init,
        a_init=a_init,
        objective_functions=objective_functions,
        constraints=constraints,
        ode_solver=ode_solver,
        use_sx=False,
        n_threads=32,
        variable_mappings=dof_mapping,
        phase_transitions=phase_transitions,
        multinode_constraints=multinode_constraints,
    )

    return ocp, qv


def main():
    model_path = "../../pianoptim/models/pianist_and_key.bioMod"
    # n_shooting = (15, 15, 15)
    # ode_solver = OdeSolver.COLLOCATION(polynomial_degree=3)
    # ode_solver = OdeSolver.RK2(n_integration_steps=5)
    # ode_solver = OdeSolver.RK4(n_integration_steps=5)

    n_shooting = (3, 3, 3, 30)
    ode_solver = [
        OdeSolver.COLLOCATION(polynomial_degree=9),
        OdeSolver.COLLOCATION(polynomial_degree=9),
        OdeSolver.COLLOCATION(polynomial_degree=9),
        OdeSolver.COLLOCATION(polynomial_degree=3),
    ]

    min_phase_time = (0.04, 0.045, 0.05, 0.25)
    max_phase_time = (0.05, 0.055, 0.06, 0.35)

    ocp, qv = prepare_ocp(
        model_path=model_path,
        n_shootings=n_shooting,
        min_phase_times=min_phase_time,
        max_phase_times=max_phase_time,
        ode_solver=ode_solver,
    )
    ocp.add_plot_penalty(CostType.OBJECTIVES)

    solv = Solver.IPOPT(
        # online_optim=OnlineOptim.MULTIPROCESS_SERVER,
        # online_optim=OnlineOptim.DEFAULT,
        show_options={"show_bounds": True, "automatically_organize": False},
    )
    solv.set_maximum_iterations(500)
    solv.set_linear_solver("ma57")
    sol = ocp.solve(solv)

    print(sol.real_time_to_optimize)

    from pyorerun import BiorbdModel as PyorerunBiorbdModel, MultiPhaseRerun

    pyomodel = PyorerunBiorbdModel(model_path)
    stepwise_time = sol.stepwise_time(to_merge=SolutionMerge.NODES, time_alignment=TimeAlignment.STATES)
    stepwise_states = sol.stepwise_states(to_merge=SolutionMerge.NODES)
    stepwise_astates = sol.decision_algebraic_states(to_merge=SolutionMerge.NODES)

    q = [np.zeros((pyomodel.nb_q, len(stepwise_time[phase]))) for phase in range(3)]
    for phase in range(3):
        q_u = stepwise_states[phase]["q_u"]
        q_v = stepwise_astates[phase]["q_v"]
        q[phase] = ocp.nlp[phase].model.state_from_partition(q_u, q_v).toarray()

    mprr = MultiPhaseRerun()
    mprr.add_phase(t_span=stepwise_time[0], phase=0)
    mprr.add_phase(t_span=stepwise_time[1], phase=1)
    mprr.add_phase(t_span=stepwise_time[2], phase=2)
    mprr.add_phase(t_span=stepwise_time[3], phase=3)

    mprr.add_animated_model(pyomodel, q[0], phase=0)
    mprr.add_animated_model(pyomodel, q[1], phase=1)
    mprr.add_animated_model(pyomodel, q[2], phase=2)

    # add 1 row of zeros under last phase q
    qtemp = stepwise_states[3]["q"]
    nb_steps = len(stepwise_time[3])
    q.append(np.vstack((qtemp, np.zeros((1, nb_steps)))))
    mprr.add_animated_model(pyomodel, q[3], phase=3)

    mprr.rerun()
    sol.print_cost()

    import datetime

    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    sol.graphs(show_bounds=True, save_name=f"../../results/press_play_torque_derivative_driven_{date}.png")


if __name__ == "__main__":
    main()
