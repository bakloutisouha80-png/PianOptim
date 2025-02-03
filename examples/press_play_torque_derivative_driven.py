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
    DynamicsFcn,
    BoundsList,
    InitialGuessList,
    CostType,
    Node,
    OptimalControlProgram,
    ObjectiveFcn,
    ConstraintList,
    OdeSolver,
    Solver,
    PlotType,
    BiMappingList,
    OnlineOptim,
    TimeAlignment,
    SolutionMerge,
    ConstraintFcn,
)

from pianoptim.models.pianist_holonomic import HolonomicPianist
from pianoptim.utils.custom_functions import (
    custom_func_track_markers,
    custom_func_track_markers_velocity,
    custom_contraint_lambdas,
)
from pianoptim.utils.dynamics import (
    holonomic_torque_driven_custom_qv_init,
    configure_holonomic_torque_driven,
)


import numpy as np

from pianoptim.utils.torque_derivative_holonomic_driven import (
    configure_holonomic_torque_derivative_driven,
    holonomic_torque_derivative_driven_custom_qv_init,
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
    u_bounds = BoundsList()
    u_init = InitialGuessList()
    objective_functions = ObjectiveList()
    constraints = ConstraintList()

    # Load and constraints the dynamic model
    models = (HolonomicPianist(model_path), HolonomicPianist(model_path), HolonomicPianist(model_path))
    first_model = models[0]

    variable_bimapping = BiMappingList()
    variable_bimapping.add(
        "q", to_second=[0, 1, 2, 3, 4, None, None, 5, 6, 7, 8, 9], to_first=[0, 1, 2, 3, 4, 7, 8, 9, 10, 11]
    )
    variable_bimapping.add(
        "qdot", to_second=[0, 1, 2, 3, 4, None, None, 5, 6, 7, 8, 9], to_first=[0, 1, 2, 3, 4, 7, 8, 9, 10, 11]
    )

    dof_mapping = BiMappingList()
    dof_mapping.add(
        "tau",
        to_second=[i for i in range(first_model.nb_q - 1)] + [None],
        to_first=[i for i in range(first_model.nb_q - 1)],
    )
    dof_mapping.add(
        "taudot",
        to_second=[i for i in range(first_model.nb_q - 1)] + [None],
        to_first=[i for i in range(first_model.nb_q - 1)],
    )

    q = first_model.q_hand_on_keyboard
    qu = q[[0, 1, 2, 3, 4, 7, 8, 9, 10, 11]]
    qv = q[[5, 6, 12]]

    dynamics.add(
        configure_holonomic_torque_derivative_driven,
        dynamic_function=holonomic_torque_derivative_driven_custom_qv_init,
        custom_q_v_init=qv,
        phase=0,
    )
    dynamics.add(
        configure_holonomic_torque_derivative_driven,
        dynamic_function=holonomic_torque_driven_custom_qv_init,
        custom_q_v_init=qv,
        phase=1,
    )
    dynamics.add(
        configure_holonomic_torque_derivative_driven,
        dynamic_function=holonomic_torque_driven_custom_qv_init,
        custom_q_v_init=qv,
        phase=2,
    )

    x_bounds.add("q_u", bounds=models[0].bounds_from_ranges("q", variable_bimapping), phase=0)
    x_bounds.add("qdot_u", bounds=models[0].bounds_from_ranges("qdot", variable_bimapping), phase=0)
    x_bounds.add("q_u", bounds=models[1].bounds_from_ranges("q", variable_bimapping), phase=1)
    x_bounds.add("qdot_u", bounds=models[1].bounds_from_ranges("qdot", variable_bimapping), phase=1)
    x_bounds.add("q_u", bounds=models[2].bounds_from_ranges("q", variable_bimapping), phase=2)
    x_bounds.add("qdot_u", bounds=models[2].bounds_from_ranges("qdot", variable_bimapping), phase=2)

    x_init.add("q_u", qu, phase=0)
    x_init.add("q_u", qu, phase=1)
    x_init.add("q_u", qu, phase=2)

    x_init.add("qdot_u", [0] * (models[0].nb_q - 3), phase=0)
    x_init.add("qdot_u", [0] * (models[1].nb_q - 3), phase=1)
    x_init.add("qdot_u", [0] * (models[2].nb_q - 3), phase=2)

    x_bounds.add("tau", min_bound=[-40] * (models[0].nb_tau - 1), max_bound=[40] * (models[0].nb_tau - 1), phase=0)
    x_bounds.add("tau", min_bound=[-40] * (models[1].nb_tau - 1), max_bound=[40] * (models[1].nb_tau - 1), phase=1)
    x_bounds.add("tau", min_bound=[-40] * (models[2].nb_tau - 1), max_bound=[40] * (models[2].nb_tau - 1), phase=2)

    x_init.add("tau", [0] * (models[0].nb_tau - 1), phase=0)
    x_init.add("tau", [0] * (models[1].nb_tau - 1), phase=1)
    x_init.add("tau", [0] * (models[2].nb_tau - 1), phase=2)

    u_bounds.add(
        "taudot", min_bound=[-10000] * (models[0].nb_tau - 1), max_bound=[10000] * (models[0].nb_tau - 1), phase=0
    )
    u_bounds.add(
        "taudot", min_bound=[-10000] * (models[1].nb_tau - 1), max_bound=[10000] * (models[1].nb_tau - 1), phase=1
    )
    u_bounds.add(
        "taudot", min_bound=[-10000] * (models[2].nb_tau - 1), max_bound=[10000] * (models[2].nb_tau - 1), phase=2
    )

    u_init.add("taudot", [0] * (models[0].nb_tau - 1), phase=0)
    u_init.add("taudot", [0] * (models[1].nb_tau - 1), phase=1)
    u_init.add("taudot", [0] * (models[2].nb_tau - 1), phase=2)

    # Objective Functions
    #objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="tau", phase=0, weight=1)
    #objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="tau", phase=1, weight=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot_u", phase=1, weight=0.001)
    #objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="tau", phase=2, weight=1)

    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="taudot", phase=0, weight=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="taudot", phase=1, weight=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="taudot", phase=2, weight=1)

    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="tau", phase=0, weight=0.1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="tau", phase=1, weight=0.1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="tau", phase=2, weight=0.1)

    for p in range(3):
        objective_functions.add(
            custom_contraint_lambdas,
            custom_type=ObjectiveFcn.Lagrange,
            index=[2],
            phase=p,
            weight=0.1,
            custom_qv_init=qv,
        )

    # The first and last frames are at rest
    x_bounds[0]["qdot_u"][:, 0] = 0
    x_bounds[-1]["qdot_u"].min[:, -1] = -0.01
    x_bounds[-1]["qdot_u"].max[:, -1] = 0.01

    # Start and end with the finger on the key at top position without any velocity
    START_POSE = np.array([[-0.16104043, -0.5356114, 0.124]]).T
    BED_POSE = np.array([[-0.16104043, -0.5356114, 0.114]]).T

    constraints.add(
        custom_func_track_markers,
        phase=0,
        node=Node.START,
        marker="contact_finger",
        target=START_POSE,
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
        min_bound=BED_POSE,
        max_bound=START_POSE,
    )

    slack = np.array([(0.001, 0.001, 0.001)]).T
    constraints.add(
        custom_func_track_markers,
        phase=1,
        node=Node.START,
        marker="contact_finger",
        # target=np.tile(BED_POSE, (1, n_shootings[1] + 1)),
        target=BED_POSE,
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
        target=BED_POSE,
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
        min_bound=BED_POSE,
        max_bound=START_POSE,
    )

    constraints.add(
        custom_func_track_markers,
        phase=2,
        node=Node.END,
        marker="contact_finger",
        target=START_POSE,
        custom_qv_init=qv,
    )

    phase_times = [(min_t + max_t) / 2 for min_t, max_t in zip(min_phase_times, max_phase_times)]

    # Prepare the optimal control program
    ocp = OptimalControlProgram(
        bio_model=models,
        dynamics=dynamics,
        n_shooting=n_shootings,
        phase_time=phase_times,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=objective_functions,
        constraints=constraints,
        ode_solver=ode_solver,
        use_sx=False,
        n_threads=32,
        variable_mappings=dof_mapping,
    )

    # # Add a graph that shows the finger height
    # for i in range(3):
    #     ocp.add_plot(
    #         "Finger height",
    #         lambda t0, phases_dt, node_idx, x, u, p, a, d: models[i].compute_marker_from_dm(
    #             x[: models[i].nb_q], "contact_finger"
    #         )[2, :],
    #         phase=i,
    #         plot_type=PlotType.INTEGRATED,
    #     )

    return ocp, qv


def main():
    model_path = "../pianoptim/models/pianist_and_key.bioMod"
    n_shooting = (20, 20, 20)
    min_phase_time = (0.05, 0.05, 0.05)
    max_phase_time = (0.10, 0.10, 0.10)
    # ode_solver = OdeSolver.RK2(n_integration_steps=5)
    # ode_solver = OdeSolver.RK4(n_integration_steps=5)
    ode_solver = OdeSolver.COLLOCATION(polynomial_degree=3)
    #
    ocp, qv = prepare_ocp(
        model_path=model_path,
        n_shootings=n_shooting,
        min_phase_times=min_phase_time,
        max_phase_times=max_phase_time,
        ode_solver=ode_solver,
    )
    ocp.add_plot_penalty(CostType.ALL)

    solv = Solver.IPOPT(
        # online_optim=OnlineOptim.MULTIPROCESS_SERVER,
        # online_optim=OnlineOptim.DEFAULT,
        # show_options={"show_bounds": True, "automatically_organize": False},
    )
    solv.set_maximum_iterations(5000)
    solv.set_linear_solver("ma57")
    sol = ocp.solve(solv)

    print(sol.real_time_to_optimize)

    from pyorerun import BiorbdModel as PyorerunBiorbdModel, MultiPhaseRerun

    pyomodel = PyorerunBiorbdModel(model_path)
    stepwise_time = sol.stepwise_time(to_merge=SolutionMerge.NODES, time_alignment=TimeAlignment.STATES)
    stepwise_states = sol.stepwise_states(to_merge=SolutionMerge.NODES)

    q = [np.zeros((pyomodel.nb_q, len(stepwise_time[phase]))) for phase in range(3)]
    for phase in range(3):
        q_u = stepwise_states[phase]["q_u"]
        for i, q_u in enumerate(q_u.T):
            q[phase][:, i] = sol.ocp.nlp[phase].model.compute_q()(q_u, qv).toarray().squeeze()

    mprr = MultiPhaseRerun()
    mprr.add_phase(t_span=stepwise_time[0], phase=0)
    mprr.add_phase(t_span=stepwise_time[1], phase=1)
    mprr.add_phase(t_span=stepwise_time[2], phase=2)

    mprr.add_animated_model(pyomodel, q[0], phase=0)
    mprr.add_animated_model(pyomodel, q[1], phase=1)
    mprr.add_animated_model(pyomodel, q[2], phase=2)

    mprr.rerun()
    sol.print_cost()

    import datetime
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    sol.graphs(show_bounds=True, save_name=f"../results/press_play_torque_derivative_driven_{date}.png")


if __name__ == "__main__":
    main()
