from typing import Any

import numpy as np

from bioptim import PenaltyController, ControlType, DynamicsFunctions
from bioptim.limits.penalty import PenaltyFunctionAbstract
from casadi import horzcat, DM, vertcat, MX


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
    if "q_v" in controller.algebraic_states:
        qv = controller.algebraic_states["q_v"].cx
        q = controller.model.state_from_partition(qu, qv)
    else:
        qv_init = DM.zeros(controller.model.nb_dependent_joints) if custom_qv_init is None else DM(custom_qv_init)
        q = controller.model.compute_q()(qu, qv_init)

    diff_markers = controller.model.marker(first_marker_idx)(q, controller.parameters.cx)

    return diff_markers[axes]


def custom_func_track_markers_velocity(
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
    qdotu = controller.states["qdot_u"].mapping.to_second.map(controller.states["qdot_u"].cx)

    if "q_v" in controller.algebraic_states:
        qv = controller.algebraic_states["q_v"].cx
        q = controller.model.state_from_partition(qu, qv)
    else:
        qv_init = DM.zeros(controller.model.nb_dependent_joints) if custom_qv_init is None else DM(custom_qv_init)
        q = controller.model.compute_q()(qu, qv_init)

    qdot = controller.model.compute_qdot()(q, qdotu)

    marker_vel = controller.model.marker_velocity(first_marker_idx)(q, qdot, controller.parameters.cx)

    return marker_vel[axes]


def custom_func_superimpose_markers(
    controller: PenaltyController,
    first_marker: str | int,
    second_marker: str | int,
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
    if "q_v" in controller.algebraic_states:
        qv = controller.algebraic_states["q_v"].cx
        q = controller.model.state_from_partition(qu, qv)
    else:
        qv_init = DM.zeros(controller.model.nb_dependent_joints) if custom_qv_init is None else DM(custom_qv_init)
        q = controller.model.compute_q()(qu, qv_init)

    diff_markers = controller.model.marker(second_marker_idx)(q, controller.parameters.cx) - controller.model.marker(
        first_marker_idx
    )(q, controller.parameters.cx)

    return diff_markers[axes]


def constraint_qv_init(
    controllers: list[PenaltyController],
):
    """
    The initial q_v of the second phase should be the final q_v of the first phase

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


def minimize_qdot_v_intermediates(
    controllers: PenaltyController,
    dof_idx: int = None,
):
    """
    Minimize the qdot_v from q_u, qdot_u, q_v


    Parameters
    ----------
    controller: PenaltyController
        The penalty node elements
    """
    if dof_idx is None:
        dof_idx = [i for i in range(controllers.model.nb_independent_joints)]

    q_u = controllers.states["q_u"]
    q_u_complete = q_u.mapping.to_second.map(q_u.cx)
    qdot_u = controllers.states["qdot_u"]
    qdot_u_complete = qdot_u.mapping.to_second.map(qdot_u.cx)

    q_v = controllers.algebraic_states["q_v"]
    q_v_complete = q_v.mapping.to_second.map(q_v.cx)

    q = controllers.model.state_from_partition(q_u_complete, q_v_complete)
    qdot_v = controllers.model.compute_qdot_v()(q, qdot_u_complete)[dof_idx]

    for q_u_cx, q_v_cx, qdot_u_cx in zip(q_u.cx_intermediates_list, q_v.cx_intermediates_list, qdot_u.cx_intermediates_list):
        q_u_complete = q_u.mapping.to_second.map(q_u_cx)
        qdot_u_complete = qdot_u.mapping.to_second.map(qdot_u_cx)
        q_v_complete = q_v.mapping.to_second.map(q_v_cx)
        q = controllers.model.state_from_partition(q_u_complete, q_v_complete)
        qdot_v = vertcat(qdot_v, controllers.model.compute_qdot_v()(q, qdot_u_complete)[dof_idx])

    return qdot_v


def custom_contraint_lambdas(controller: PenaltyController, custom_qv_init: np.ndarray = None) -> MX:
    """The model can only pull on the legs, not push"""
    # Recuperer les q
    model = controller.model

    q_u = controller.states["q_u"].cx
    qdot_u = controller.states["qdot_u"].cx

    tau = controller.controls["tau"] if "tau" in controller.controls else controller.states["tau"]
    tau_complete = tau.mapping.to_second.map(tau.cx)

    # Calculer lambdas
    if "q_v" in controller.algebraic_states:
        q_v = controller.algebraic_states["q_v"].cx
        q = model.state_from_partition(q_u, q_v)
        qdot = model.compute_qdot()(q, qdot_u)
        qddot_u = model.partitioned_forward_dynamics_with_qv()(q_u, q_v, qdot_u, tau_complete)
        qddot = model.compute_qddot()(q, qdot, qddot_u)
        lambdas = model._compute_the_lagrangian_multipliers()(q, qdot, qddot, tau_complete)
    else:
        lambdas = model.compute_the_lagrangian_multipliers()(q_u, qdot_u, custom_qv_init, tau_complete)

    return lambdas


def custom_contraint_lambdas_shear(controller: PenaltyController, bio_model: Any) -> MX:
    """
    Relaxed friction cone, the model can push a little bit
    lagrange_1**2 < lagrange_0**2
    """
    q_u = controller.states["q_u"].cx
    qdot_u = controller.states["qdot_u"].cx
    tau = controller.controls["tau"].cx if "tau" in controller.controls else controller.states["tau"].cx
    pelvis_mx = MX.zeros(3)
    new_tau = vertcat(pelvis_mx, tau)

    # Calculer lambdas
    lambdas = bio_model.compute_the_lagrangian_multipliers(q_u, qdot_u, new_tau)
    lagrange_0 = lambdas[0]
    lagrange_1 = lambdas[1]

    return lagrange_0**2 - lagrange_1**2


# def add_constraint_tucking_friction_cone(biomodel_holonomic, constraints):
#     # "relaxed friction cone"
#     constraints.add(
#         custom_contraint_lambdas_shear,
#         node=Node.ALL_SHOOTING,
#         bio_model=biomodel_holonomic,
#         max_bound=np.inf,
#         min_bound=0,
#         phase=2,
#     )
#     # The model can only pull on the legs, not push
#     constraints.add(
#         custom_contraint_lambdas_normal,
#         node=Node.ALL_SHOOTING,
#         bio_model=biomodel_holonomic,
#         max_bound=-0.1,
#         min_bound=-np.inf,
#         phase=2,
#     )
#     return constraints
