import numpy as np

from bioptim import PenaltyController, ControlType, DynamicsFunctions
from bioptim.limits.penalty import PenaltyFunctionAbstract
from casadi import horzcat, DM


def custom_func_track_markers(
    controller: PenaltyController,
    marker: str | int,
    axes: list[int] = [0, 1, 2],
    custom_qv_init: np.ndarray = None,
):
    """
    Minimize the distance between two markers
    By default this function is quadratic, meaning that it minimizes distance between them.

    Parameters
    ----------
    controller: PenaltyController
        The penalty node elements
    marker: str | int
        The name or index of the marker
    axes: list[int]
        The axes to track
    """

    first_marker_idx = controller.model.marker_index(marker) if isinstance(marker, str) else marker
    PenaltyFunctionAbstract._check_idx("marker", [first_marker_idx], controller.model.nb_markers)

    qu = controller.states["q_u"].mapping.to_second.map(controller.states["q_u"].cx)
    qv_init = DM.zeros(controller.model.nb_dependent_joints) if custom_qv_init is None else DM(custom_qv_init)
    q = controller.model.compute_q()(qu, qv_init)

    diff_markers = controller.model.marker(first_marker_idx)(q, controller.parameters.cx)

    return diff_markers[axes]


def custom_func_superimpose_markers(
    controller: PenaltyController,
    first_marker: str | int,
    second_marker: str | int,
    axes: list[int] = [0, 1, 2],
):
    """
    Minimize the distance between two markers
    By default this function is quadratic, meaning that it minimizes distance between them.

    Parameters
    ----------
    controller: PenaltyController
        The penalty node elements
    first_marker: str | int
        The name or index of one of the two markers
    second_marker: str | int
        The name or index of one of the two markers
    """

    first_marker_idx = controller.model.marker_index(first_marker) if isinstance(first_marker, str) else first_marker
    second_marker_idx = (
        controller.model.marker_index(second_marker) if isinstance(second_marker, str) else second_marker
    )
    PenaltyFunctionAbstract._check_idx("marker", [first_marker_idx, second_marker_idx], controller.model.nb_markers)
    qu = controller.states["q_u"].mapping.to_second.map(controller.states["q_u"].cx)
    qv_init = DM.zeros(controller.model.nb_dependent_joints)
    q = controller.model.compute_q()(qu, qv_init)

    diff_markers = controller.model.marker(second_marker_idx)(q, controller.parameters.cx) - controller.model.marker(
        first_marker_idx
    )(q, controller.parameters.cx)

    return diff_markers[axes]


def constraint_qv_init(
    controllers: list[PenaltyController],
):
    """
    Minimize the distance between two markers
    By default this function is quadratic, meaning that it minimizes distance between them.

    Parameters
    ----------
    controller: PenaltyController
        The penalty node elements
    """

    if controllers[0].control_type in (
        ControlType.CONSTANT,
        ControlType.CONSTANT_WITH_LAST_NODE,
    ):
        u = controllers[0].controls.cx_start
    elif controllers[0].control_type == ControlType.LINEAR_CONTINUOUS:
        # TODO: For cx_end take the previous node
        u = horzcat(controllers[0].controls.cx_start, controllers[0].controls.cx_end)
    else:
        raise NotImplementedError(f"Dynamics with {controllers[0].control_type} is not implemented yet")

    t_span = controllers[0].t_span.cx
    states_end_interval = controllers[0].integrate(
        t_span=t_span,
        x0=controllers[0].states.cx_start,
        u=u,
        p=controllers[0].parameters.cx_start,
        a=controllers[0].algebraic_states.cx_start,
        d=controllers[0].numerical_timeseries.cx,
    )["xf"]

    qv_init_control = controllers[0].controls["q_v_init"].mapping.to_first.map(controllers[0].controls["q_v_init"].cx)

    qu_end = states_end_interval[0 : controllers[0].model.nb_independent_joints]
    qv_end = controllers[0].model.compute_q_v()(qu_end, qv_init_control)

    qv_start = controllers[1].controls["q_v_init"].mapping.to_second.map(controllers[1].controls["q_v_init"].cx)

    return qv_end - qv_start
