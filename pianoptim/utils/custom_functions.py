from bioptim import PenaltyController
from bioptim.limits.penalty import PenaltyFunctionAbstract


def custom_func_track_markers(
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
    q = controller.model.compute_q()(qu, controller.parameters.cx)

    diff_markers = controller.model.marker(second_marker_idx)(q, controller.parameters.cx) - controller.model.marker(
        first_marker_idx
    )(q, controller.parameters.cx)

    return diff_markers[axes]
