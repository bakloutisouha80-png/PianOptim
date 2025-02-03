"""
TODO:
    - TORQUE DERIVATIVE DRIVEN
    - limited newton iterations
    - single shooting on qv to initialize the newton descent
    - COLLOCATION with lamda and qv as variables
"""

from bioptim import (
    ObjectiveList,
    DynamicsList,
    BoundsList,
    InitialGuessList,
    CostType,
    Node,
    OptimalControlProgram,
    DynamicsFcn,
    ObjectiveFcn,
    ConstraintList,
    OdeSolver,
    Solver,
    PlotType,
    BiMappingList,
    OnlineOptim,
    TimeAlignment,
    SolutionMerge,
    MultinodeConstraintList,
)

from pianoptim.models.pianist_holonomic import HolonomicPianist
from pianoptim.utils.custom_functions import custom_func_superimpose_markers, constraint_qv_init
from pianoptim.utils.dynamics import (
    holonomic_torque_driven_free_qv,
    configure_holonomic_torque_driven_free_qv,
    holonomic_torque_driven_with_qv,
)


import numpy as np


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
    model = HolonomicPianist(model_path)
    variable_bimapping = BiMappingList()
    variable_bimapping.add(
        "q", to_second=[0, 1, 2, 3, 4, None, None, 5, 6, 7, 8, 9], to_first=[0, 1, 2, 3, 4, 7, 8, 9, 10, 11]
    )
    variable_bimapping.add(
        "qdot", to_second=[0, 1, 2, 3, 4, None, None, 5, 6, 7, 8, 9], to_first=[0, 1, 2, 3, 4, 7, 8, 9, 10, 11]
    )

    dof_mapping = BiMappingList()
    dof_mapping.add(
        "tau", to_second=[i for i in range(model.nb_q - 1)] + [None], to_first=[i for i in range(model.nb_q - 1)]
    )

    # dynamics.add(DynamicsFcn.HOLONOMIC_TORQUE_DRIVEN, phase=0)

    dynamics.add(
        configure_holonomic_torque_driven_free_qv,
        dynamic_function=holonomic_torque_driven_with_qv,
        phase=0,
    )
    # multinode_constraints = MultinodeConstraintList()
    # for i in range(n_shootings - 1):
    #     multinode_constraints.add(
    #         constraint_qv_init,
    #         nodes_phase=(0, 0),
    #         nodes=(i, i + 1),
    #     )
    q = model.q_hand_on_keyboard
    qu = q[[0, 1, 2, 3, 4, 7, 8, 9, 10, 11]]
    qv = q[[5, 6, 12]]
    x_bounds.add("q_u", bounds=model.bounds_from_ranges("q", variable_bimapping), phase=0)
    # x_bounds.add("q_v", bounds=model.bounds_from_ranges("q", variable_bimapping), phase=0)
    x_bounds.add("qdot_u", bounds=model.bounds_from_ranges("qdot", variable_bimapping), phase=0)
    x_init.add("q_u", qu, phase=0)
    # x_init.add("q_v", qv, phase=0)
    x_init.add("qdot_u", [0] * (model.nb_q - 3), phase=0)

    u_bounds.add("tau", min_bound=[-40] * (model.nb_tau - 1), max_bound=[40] * (model.nb_tau - 1), phase=0)
    u_init.add("tau", [0] * (model.nb_tau - 1), phase=0)

    # Minimization to convexify the problem
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=0, weight=0.001)

    # The first and last frames are at rest
    x_bounds["qdot_u"][:, 0] = 0
    x_bounds["qdot_u"][:, -1] = 0

    # Start and end with the finger on the key at top position without any velocity
    constraints.add(
        custom_func_superimpose_markers,
        phase=0,
        node=Node.START,
        first_marker="finger_marker",
        second_marker="Key1_Top",
    )

    constraints.add(
        custom_func_superimpose_markers,
        phase=0,
        node=Node.END,
        first_marker="finger_marker",
        second_marker="Key1_Top",
        axes=2,
    )

    # Prepare the optimal control program
    ocp = OptimalControlProgram(
        bio_model=model,
        dynamics=dynamics,
        n_shooting=n_shootings,
        phase_time=(min_phase_times + max_phase_times) / 2,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        objective_functions=objective_functions,
        constraints=constraints,
        # multinode_constraints=multinode_constraints,
        ode_solver=ode_solver,
        use_sx=False,
        n_threads=8,
        variable_mappings=dof_mapping,
    )

    # Add a graph that shows the finger height
    ocp.add_plot(
        "Finger height",
        lambda t0, phases_dt, node_idx, x, u, p, a, d: model.compute_marker_from_dm(x[: model.nb_q], "finger_marker")[
            2, :
        ],
        phase=0,
        plot_type=PlotType.INTEGRATED,
    )

    return ocp


def main():
    model_path = "../../pianoptim/models/pianist_and_key.bioMod"
    n_shooting = 20
    min_phase_time = 0.05
    max_phase_time = 0.10
    # ode_solver = OdeSolver.RK1(n_integration_steps=5)
    # ode_solver = OdeSolver.RK4(n_integration_steps=5)
    ode_solver = OdeSolver.COLLOCATION(polynomial_degree=3)
    #
    ocp = prepare_ocp(
        model_path=model_path,
        n_shootings=n_shooting,
        min_phase_times=min_phase_time,
        max_phase_times=max_phase_time,
        ode_solver=ode_solver,
    )
    ocp.add_plot_penalty(CostType.ALL)

    solv = Solver.IPOPT(
        # online_optim=OnlineOptim.MULTIPROCESS_SERVER,
        online_optim=OnlineOptim.DEFAULT,
        show_options={"show_bounds": True, "automatically_organize": False},
    )
    solv.set_maximum_iterations(500)
    solv.set_linear_solver("ma57")
    sol = ocp.solve(solv)

    print(sol.real_time_to_optimize)
    print(sol.decision_algebraic_states(to_merge=SolutionMerge.NODES)["q_v"])

    from pyorerun import BiorbdModel as PyorerunBiorbdModel, PhaseRerun

    pyomodel = PyorerunBiorbdModel(model_path)
    stepwise_time = sol.stepwise_time(to_merge=SolutionMerge.NODES, time_alignment=TimeAlignment.STATES)
    stepwise_q_u = sol.stepwise_states(to_merge=SolutionMerge.NODES)["q_u"]

    q = np.zeros((pyomodel.nb_q, len(stepwise_time)))
    for i, q_u in enumerate(stepwise_q_u.T):
        q[:, i] = sol.ocp.nlp[0].model.compute_q()(q_u, np.zeros(3)).toarray().squeeze()

    rerun = PhaseRerun(stepwise_time)
    rerun.add_animated_model(pyomodel, q)
    rerun.rerun("animation")

    sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()
