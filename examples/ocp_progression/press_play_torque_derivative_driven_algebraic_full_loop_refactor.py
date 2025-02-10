"""
This is a multiphase problem where the pianist is playing a note on the piano.
The play style is a press play where the finger is placed on the key and the key is pressed.

So this is a three-phase problem:
- Phase 0: The finger is place on the key and waits to go down
- Phase 1: The finger is placed on the key and goes down
- Phase 2: The key is actively pressed into the bed
- Phase 3: The key is released, from the bed and the finger is lifted up to the top position
- Phase 4: The finger is lifted up to the top position
- Phase 5: The finger is replaced to the key ready to play again
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
from pianoptim.utils.custom_transitions import (custom_phase_transition_algebraic_post,
                                                custom_phase_transition_algebraic_pre,
                                                transition_algebraic_pre_with_collision)

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
    objective_functions = ObjectiveList()
    constraints = ConstraintList()
    multinode_constraints = MultinodeConstraintList()
    phase_transitions = PhaseTransitionList()

    # Load and constraints the dynamic model
    models = (
        HolonomicPianist(model_path),
        HolonomicPianist(model_path),
        HolonomicPianist(model_path),
        HolonomicPianist(model_path),
        BiorbdModel("../../pianoptim/models/pianist.bioMod"),
        BiorbdModel("../../pianoptim/models/pianist.bioMod"),
    )
    first_model = models[0]

    u_to_second = [0, 1, 2, 3, 4, None, None, 5, 6, 7, 8, 9, None]
    u_to_first = [0, 1, 2, 3, 4, 7, 8, 9, 10, 11]

    v_to_second = [None, None, None, None, None, 0, 1, None, None, None, None, None, 3]
    v_to_first = [5, 6, 12]

    u_variable_bimapping = BiMappingList()
    u_variable_bimapping.add(
        "q", to_second=u_to_second, to_first=u_to_first
    )
    u_variable_bimapping.add(
        "qdot", to_second=u_to_second, to_first=u_to_first
    )


    v_variable_bimapping = BiMappingList()
    v_variable_bimapping.add(
        "q", to_second=v_to_second, to_first=v_to_first
    )

    NB_TAU = first_model.nb_tau - 1

    tau_to_second = [i for i in range(NB_TAU)] + [None]
    tau_to_first = [i for i in range(NB_TAU)]

    x_bounds = BoundsList()
    x_init = InitialGuessList()
    a_bounds = BoundsList()
    a_init = InitialGuessList()
    u_bounds = BoundsList()
    u_init = InitialGuessList()

    dof_mapping = BiMappingList()
    for p in range(4):
        dof_mapping.add(
            "tau",
            to_second=tau_to_second,
            to_first=tau_to_first,
            phase=p,
        )
        dof_mapping.add(
            "taudot",
            to_second=tau_to_second,
            to_first=tau_to_first,
            phase=p,
        )

    q = FINGER_TIP_ON_KEY_RELAXED
    qu = q[first_model.independent_joint_index]
    qv = q[first_model.dependent_joint_index]

    for p in range(4):
        dynamics.add(
            configure_holonomic_torque_derivative_driven_with_qv,
            dynamic_function=holonomic_torque_derivative_driven_with_qv,
            custom_q_v_init=qv,
            phase=p,
        )
        # Path Constraints
        constraints.add(
            constraint_holonomic,
            node=Node.ALL_SHOOTING,
            phase=p,
        )
        constraints.add(
            constraint_holonomic_end,
            node=Node.END,
            phase=p,
        )

    for p in range(4, 6):
        dynamics.add(
            DynamicsFcn.TORQUE_DERIVATIVE_DRIVEN,
            phase=p,
        )

    taudot_max, taudot_min = 5000, -5000

    for p in range(4):
        x_bounds.add("q_u", bounds=models[0].bounds_from_ranges("q", u_variable_bimapping), phase=p)
        x_bounds.add("qdot_u", bounds=models[0].bounds_from_ranges("qdot", u_variable_bimapping), phase=p)

        a_bounds.add("q_v", bounds=models[0].bounds_from_ranges("q", v_variable_bimapping), phase=p)

        x_init.add("q_u", qu, phase=p)
        x_init.add("qdot_u", [0] * models[0].nb_independent_joints, phase=p)

        a_init.add("q_v", qv, phase=p)

        x_bounds.add(
            "tau",
            min_bound=[-40] * NB_TAU,
            max_bound=[40] * NB_TAU,
            phase=p,
        )
        x_init.add("tau", [0] * NB_TAU, phase=p)

        u_bounds.add(
            "taudot", min_bound=[-taudot_min] * NB_TAU, max_bound=[taudot_max] * NB_TAU, phase=p
        )

        u_init.add("taudot", [0] * NB_TAU, phase=p)

    #  GUIDING THE KEY HEIGHT
    a_bounds[0]["q_v"].min[-1, :] = 0.0025
    a_bounds[0]["q_v"].max[-1, :] = -0.0025
    # Reducing the key bounds
    # this should go from 0 to -0.01,
    # but I let a bit of slack to avoid numerical issues
    a_bounds[1]["q_v"].min[-1, 0] = 0.0025
    a_bounds[1]["q_v"].max[-1, 0] = -0.0025

    a_bounds[1]["q_v"].min[-1, 1] = -0.020
    a_bounds[1]["q_v"].max[-1, 1] = 0.01

    a_bounds[1]["q_v"].min[-1, 2] = -0.01025
    a_bounds[1]["q_v"].max[-1, 2] = -0.00975

    a_bounds[2]["q_v"].min[-1, :] = -0.01025
    a_bounds[2]["q_v"].max[-1, :] = -0.00975

    a_bounds[3]["q_v"].min[-1, 0] = -0.01025
    a_bounds[3]["q_v"].max[-1, 0] = -0.00975

    a_bounds[3]["q_v"].min[-1, 1:] = -0.020
    a_bounds[3]["q_v"].max[-1, 1:] = 0.01

    for p in range(4, 6):

        # mapping that map nothing to make the OCP not crash
        to_second = [i for i in range(models[-1].nb_q)]
        to_first = [i for i in range(models[-1].nb_q)]
        dof_mapping.add(
            "tau",
            to_second=to_second,
            to_first=to_first,
            phase=p,
        )
        dof_mapping.add(
            "taudot",
            to_second=to_second,
            to_first=to_first,
            phase=p,
        )

        x_bounds.add("q", bounds=models[p].bounds_from_ranges("q"), phase=p)
        x_bounds.add("qdot", bounds=models[p].bounds_from_ranges("qdot"), phase=p)
        x_init.add("q", q[:-1], phase=p)
        x_init.add("qdot", [0] * (models[p].nb_q), phase=p)

        x_bounds.add(
            "tau",
            min_bound=[-40] * NB_TAU,
            max_bound=[40] * NB_TAU,
            phase=p,
        )
        x_init.add("tau", [0] * NB_TAU, phase=p)

        u_bounds.add(
            "taudot",
            min_bound=[-taudot_min] * NB_TAU,
            max_bound=[taudot_max] * NB_TAU,
            phase=p,
        )
        u_init.add("taudot", [0] * NB_TAU, phase=p)

    # Objective Functions
    elbow_wrist_idx = [8, 10]
    shoulder_non_flexion_dof = [7, 6]
    no_elbow_wrist_idx = [i for i in range(first_model.nb_q) if i not in elbow_wrist_idx]

    objective_functions.add(
        custom_contraint_lambdas,
        custom_type=ObjectiveFcn.Lagrange,
        index=[0, 2],
        phase=0,
        weight=0.1,
        custom_qv_init=qv,
    )
    for p in range(4):
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

    # Trying to help with no speed of joint to garanty no speed of the key
    # But as side effects
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot_u", phase=0, weight=0.001)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot_u", phase=2, weight=0.001)

    for p in range(4, 6):
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
            key="taudot",
            phase=p,
            weight=1,
            index=no_elbow_wrist_idx,
        )

        # reduce the torque on all joints
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="tau", phase=p, weight=0.1)

        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", phase=p, weight=1, index=shoulder_non_flexion_dof
        )
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", phase=p, weight=0.1,
                                index=shoulder_non_flexion_dof
        )

        constraints.add(
            ConstraintFcn.TRACK_MARKERS,
            phase=p,
            node=Node.ALL_SHOOTING,
            marker_index="contact_finger",
            min_bound=KEY_TOP_UNPRESSED,
            max_bound=ELEVATED_FINGER_TIP,
            index=0  # make sure bound only x direction
        )

    constraints.add(
        custom_func_track_markers,
        phase=0,
        node=Node.ALL_SHOOTING,
        marker="contact_finger",
        target=KEY_TOP_UNPRESSED,
        custom_qv_init=qv,
    )

    constraints.add(
        custom_func_track_markers_velocity,
        phase=0,
        node=Node.ALL,
        marker="contact_finger",
        custom_qv_init=qv,
    )

    constraints.add(
        custom_func_track_markers,
        phase=1,
        node=Node.START,
        marker="contact_finger",
        target=KEY_TOP_UNPRESSED,
        custom_qv_init=qv,
    )
    for i in range(0, 4):
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
        phase=1,
        node=Node.INTERMEDIATES,
        marker="contact_finger",
        custom_qv_init=qv,
        min_bound=KEY_TOP_PRESSED,
        max_bound=KEY_TOP_UNPRESSED,
    )

    constraints.add(
        custom_func_track_markers,
        phase=2,
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
        phase=2,
        node=Node.ALL,
        marker="contact_finger",
        custom_qv_init=qv,
    )

    constraints.add(
        custom_func_track_markers,
        phase=2,
        node=Node.END,
        marker="contact_finger",
        target=KEY_TOP_PRESSED,
        custom_qv_init=qv,
    )

    constraints.add(
        custom_func_track_markers,
        phase=3,
        node=Node.ALL_SHOOTING,
        marker="contact_finger",
        custom_qv_init=qv,
        min_bound=KEY_TOP_PRESSED,
        max_bound=KEY_TOP_UNPRESSED,
    )

    constraints.add(
        custom_func_track_markers,
        phase=3,
        node=Node.END,
        marker="contact_finger",
        target=KEY_TOP_UNPRESSED,
        custom_qv_init=qv,
    )



    phase_transitions.add(
        custom_phase_transition_algebraic_post,
        phase_pre_idx=3,
    )



    constraints.add(
        ConstraintFcn.TRACK_MARKERS,
        phase=4,
        node=Node.END,
        marker_index="contact_finger",
        target=ELEVATED_FINGER_TIP,
    )

    constraints.add(
        ConstraintFcn.TRACK_MARKERS,
        phase=5,
        node=Node.END,
        marker_index="contact_finger",
        target=KEY_TOP_UNPRESSED,
    )

    # The first and last frames are at rest
    x_bounds[0]["qdot_u"][:, 0] = 0
    x_bounds[-1]["qdot"].min[:, -1] = -0.1
    x_bounds[-1]["qdot"].max[:, -1] = 0.1

    multinode_constraints.add(
        # custom_phase_transition_algebraic_pre,
        transition_algebraic_pre_with_collision,
        nodes_phase=(5,0),
        nodes=(Node.END, Node.START),
    )

    # Prepare the optimal control program
    ocp = OptimalControlProgram(
        bio_model=models,
        dynamics=dynamics,
        n_shooting=n_shootings,
        phase_time=[(min_t + max_t) / 2 for min_t, max_t in zip(min_phase_times, max_phase_times)],
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

    n_shooting = (15, 3, 3, 3, 30, 3)
    ode_solver = [
        OdeSolver.COLLOCATION(polynomial_degree=6),
        OdeSolver.COLLOCATION(polynomial_degree=9),
        OdeSolver.COLLOCATION(polynomial_degree=9),
        OdeSolver.COLLOCATION(polynomial_degree=9),
        OdeSolver.COLLOCATION(polynomial_degree=3),
        OdeSolver.COLLOCATION(polynomial_degree=9),
    ]

    min_phase_time = (0.3, 0.04, 0.045, 0.05, 0.225, 0.05)
    max_phase_time = (0.3, 0.05, 0.055, 0.06, 0.275, 0.05)

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
    solv.set_maximum_iterations(10000)
    solv.set_linear_solver("ma57")
    sol = ocp.solve(solv)

    print(sol.real_time_to_optimize)

    from pyorerun import BiorbdModel as PyorerunBiorbdModel, MultiPhaseRerun

    pyomodel = PyorerunBiorbdModel(model_path)
    stepwise_time = sol.stepwise_time(to_merge=SolutionMerge.NODES, time_alignment=TimeAlignment.STATES)
    stepwise_states = sol.stepwise_states(to_merge=SolutionMerge.NODES)
    stepwise_astates = sol.decision_algebraic_states(to_merge=SolutionMerge.NODES)

    q = [np.zeros((pyomodel.nb_q, len(stepwise_time[phase]))) for phase in range(4)]
    for phase in range(4):
        q_u = stepwise_states[phase]["q_u"]
        q_v = stepwise_astates[phase]["q_v"]
        q[phase] = ocp.nlp[phase].model.state_from_partition(q_u, q_v).toarray()

    mprr = MultiPhaseRerun()
    mprr.add_phase(t_span=stepwise_time[0], phase=0)
    mprr.add_phase(t_span=stepwise_time[1], phase=1)
    mprr.add_phase(t_span=stepwise_time[2], phase=2)
    mprr.add_phase(t_span=stepwise_time[3], phase=3)
    mprr.add_phase(t_span=stepwise_time[4], phase=4)
    mprr.add_phase(t_span=stepwise_time[5], phase=5)

    mprr.add_animated_model(pyomodel, q[0], phase=0)
    mprr.add_animated_model(pyomodel, q[1], phase=1)
    mprr.add_animated_model(pyomodel, q[2], phase=2)
    mprr.add_animated_model(pyomodel, q[3], phase=3)

    # add 1 row of zeros under last phase q
    for p in range(4, 6):
        qtemp = stepwise_states[p]["q"]
        nb_steps = len(stepwise_time[p])
        q.append(np.vstack((qtemp, np.zeros((1, nb_steps)))))
        mprr.add_animated_model(pyomodel, q[p], phase=p)

    mprr.rerun()
    sol.print_cost()

    import datetime

    has_converged = sol.status == 0
    folder = "converged" if has_converged else "not_converged"
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    sol.graphs(
        show_bounds=True,
        save_name=f"../../results/{folder}/press_play_torque_derivative_driven_{date}.png"
    )


if __name__ == "__main__":
    main()
