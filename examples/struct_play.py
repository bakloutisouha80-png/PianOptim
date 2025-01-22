from enum import Enum, auto

from bioptim import (
    ObjectiveList,
    DynamicsList,
    ConstraintFcn,
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
    Axis,
    OnlineOptim,
    PlotType,
)

from pianoptim.utils.pianist import Pianist
from pianoptim.utils.dynamics import PianistDyanmics


class ZeroPosition(Enum):
    Q = auto()
    MARKER = auto()


PREP_PHASE = 0
DOWN_PHASE = 1
HOLD_PHASE = 2
UP_PHASE = 3
END_PHASE = 4

def prepare_ocp(
    model_path: str,
    n_shootings: tuple[int, ...],
    min_phase_times: tuple[float, ...],
    max_phase_times: tuple[float, ...],
    block_trunk: bool,
    zero_position: ZeroPosition,
    ode_solver: OdeSolver,
) -> OptimalControlProgram:

    n_phases = 5

    pianist_models: list[Pianist] = []
    dynamics = DynamicsList()
    x_bounds = BoundsList()
    x_init = InitialGuessList()
    u_bounds = BoundsList()
    objective_functions = ObjectiveList()
    constraints = ConstraintList()

    # Load and constraints the dynamic model
    for phase in range(n_phases):
        model = Pianist(model_path)
        pianist_models.append(model)

        if phase in (DOWN_PHASE, HOLD_PHASE, UP_PHASE):
            dynamics.add(PianistDyanmics.configure_forward_dynamics_with_external_forces, phase=phase)
        else:
            dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase=phase)

        x_bounds.add("q", bounds=model.bounds_from_ranges("q"), phase=phase)
        x_init.add("q", model.q_hand_on_keyboard, phase=phase)

        x_bounds.add("qdot", bounds=model.bounds_from_ranges("qdot"), phase=phase)
        u_bounds.add("tau", bounds=model.joint_torque_bounds, phase=phase)

        # Minimization to convexify the problem
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", phase=phase, weight=0.001)

        # Some of the phases should be done as fast as possible
        if min_phase_times[phase] != max_phase_times[phase]:
            objective_functions.add(
                ObjectiveFcn.Mayer.MINIMIZE_TIME,
                phase=phase,
                min_bound=min_phase_times[phase],
                max_bound=max_phase_times[phase],
                weight=100,
            )

    # The first and last frames are at rest
    x_bounds[0]["qdot"][:, 0] = 0
    x_bounds[-1]["qdot"][:, -1] = 0

    # The trunk should not move if requested
    if block_trunk:
        for phase in range(n_phases):
            x_bounds[phase]["qdot"][pianist_models[phase].trunk_dof, :] = 0

    # Start with the finger on the key
    if zero_position == ZeroPosition.MARKER:
        constraints.add(
            ConstraintFcn.SUPERIMPOSE_MARKERS,
            phase=PREP_PHASE,
            node=Node.START,
            first_marker="finger_marker",
            second_marker="Key1_Top",
        )
    elif zero_position == ZeroPosition.Q:
        x_bounds[PREP_PHASE]["q"][:, 0] = pianist_models[PREP_PHASE].q_hand_on_keyboard
    else:
        raise ValueError("Invalid initial position")

    # The finger should be at the top of the key at the end of the first phase and not moving in the horizontal plane
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        phase=PREP_PHASE,
        node=Node.END,
        first_marker="finger_marker",
        second_marker="Key1_Top",
    )
    constraints.add(
        ConstraintFcn.TRACK_MARKERS_VELOCITY,
        phase=PREP_PHASE,
        node=Node.END,
        axes=[Axis.X, Axis.Y],
        marker_index="finger_marker",
    )

    # The key should be fully pressed during the hold phase
    objective_functions.add(
        ObjectiveFcn.Lagrange.SUPERIMPOSE_MARKERS,
        phase=HOLD_PHASE,
        first_marker="finger_marker",
        second_marker="key1_base",
        axes=Axis.Z,
        quadratic=False,
        weight=-1000,
    )

    # The key should be fully lifted at the end of the up phase (press_phase)
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        phase=UP_PHASE,
        node=Node.END,
        axes=Axis.Z,
        first_marker="finger_marker",
        second_marker="Key1_Top",
    )

    # The body should be at its final position at the end of the last phase
    if zero_position == ZeroPosition.MARKER:
        constraints.add(
            ConstraintFcn.SUPERIMPOSE_MARKERS,
            phase=END_PHASE,
            node=Node.END,
            first_marker="finger_marker",
            second_marker="key1_above",
        )
    elif zero_position == ZeroPosition.Q:
        x_bounds[END_PHASE]["q"][:, -1] = model.q_hand_above_keyboard
    else:
        raise ValueError("Invalid final position")

    # Prepare the optimal control program
    ocp = OptimalControlProgram(
        bio_model=pianist_models,
        dynamics=dynamics,
        n_shooting=n_shootings,
        phase_time=[(n + m) / 2 for n, m in zip(min_phase_times, max_phase_times)],
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        objective_functions=objective_functions,
        constraints=constraints,
        ode_solver=ode_solver,
        use_sx=False,
        n_threads=6,
    )

    # Add a graph that shows the finger height
    for phase in range(n_phases):
        ocp.add_plot(
            "Finger height",
            lambda t0, phases_dt, node_idx, x, u, p, a, d: pianist_models[phase].compute_marker_from_dm(
                x[: pianist_models[phase].nb_q], "finger_marker"
            )[2, :],
            phase=phase,
            plot_type=PlotType.INTEGRATED,
        )

    return ocp


def main():
    model_path = "../pianoptim/models/pianist.bioMod"
    block_trunk = False
    n_shooting = (20, 20, 50, 20, 20)
    min_phase_time = (0.05, 0.01, 0.05, 0.01, 0.05)
    max_phase_time = (0.10, 0.05, 0.05, 0.05, 0.10)
    zero_position = ZeroPosition.MARKER
    ode_solver = OdeSolver.RK4(n_integration_steps=5)

    ocp = prepare_ocp(
        model_path=model_path,
        block_trunk=block_trunk,
        n_shootings=n_shooting,
        min_phase_times=min_phase_time,
        max_phase_times=max_phase_time,
        zero_position=zero_position,
        ode_solver=ode_solver,
    )
    ocp.add_plot_penalty(CostType.ALL)

    solv = Solver.IPOPT(
        # online_optim=OnlineOptim.MULTIPROCESS_SERVER,
        online_optim=OnlineOptim.DEFAULT,
        show_options={"show_bounds": True, "automatically_organize": False},
    )
    solv.set_maximum_iterations(0)  # TODO This should not be necessary
    # solv.set_linear_solver("ma57")

    sol = ocp.solve(solv)
    # sol.graphs()
    sol.animate(viewer="pyorerun")


if __name__ == "__main__":
    main()
