from functools import cached_property

from bioptim import Bounds, HolonomicConstraintsList, HolonomicConstraintsFcn
from casadi import MX, SX, vertcat, if_else, nlpsol, DM, Function
import numpy as np

from pianoptim.models.biorbd_model_holonomic_for_collocation import HolonomicBiorbdModelForCollocation


class HolonomicPianist(HolonomicBiorbdModelForCollocation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        holonomic_constraints = HolonomicConstraintsList()
        # Phase 2 (Tucked phase):
        holonomic_constraints.add(
            "holonomic_constraints",
            HolonomicConstraintsFcn.superimpose_markers,
            biorbd_model=self,
            marker_1="contact_finger",
            marker_2="Key1_Top_in_Key1",
            index=slice(0, 3),
            local_frame_index=self.nb_segments - 1,
        )

        self.set_holonomic_configuration(
            constraints_list=holonomic_constraints,
            independent_joint_index=sorted([3, 4, 2, 1, 7, 8, 9, 10, 11, 0]),
            dependent_joint_index=sorted([12, 6, 5]),
        )

    @property
    def trunk_dof(self) -> list[int]:
        """
        Returns the index of the degrees of freedom of the trunk
        """
        return [0, 1, 2, 3]

    @property
    def hand_dof(self) -> list[int]:
        """
        Returns the index of the degrees of freedom of the hand
        """
        return [10, 11]

    @cached_property
    def q_hand_on_keyboard(self) -> np.array:
        """
        This runs the inverse kinematics to get the body position for the hand over the keyboard
        """
        q = MX.sym("q", self.nb_q, 1)

        target = self.marker(self.marker_names.index("Key1_Top"), None)(q, self.parameters)
        finger = self.marker(self.marker_names.index("finger_marker"), None)(q, self.parameters)

        s = nlpsol("sol", "ipopt", {"x": q, "g": finger - target}, {"ipopt.hessian_approximation": "limited-memory"})
        return np.array(s(x0=np.zeros(self.nb_q), lbg=np.zeros(3), ubg=np.zeros(3))["x"])[:, 0]

    @cached_property
    def q_hand_above_keyboard(self) -> np.array:
        """
        This runs the inverse kinematics to get the body position for the hand over the keyboard
        """
        q = MX.sym("q", self.nb_q, 1)

        target = self.marker(self.marker_names.index("key1_above"), None)(q, self.parameters)
        finger = self.marker(self.marker_names.index("finger_marker"), None)(q, self.parameters)

        s = nlpsol("sol", "ipopt", {"x": q, "g": finger - target}, {"ipopt.hessian_approximation": "limited-memory"})
        return np.array(s(x0=np.zeros(self.nb_q), lbg=np.zeros(3), ubg=np.zeros(3))["x"])[:, 0]

    @property
    def joint_torque_bounds(self) -> Bounds:
        return Bounds(min_bound=[-40] * self.nb_tau, max_bound=[40] * self.nb_tau, key="tau")

    def compute_marker_from_dm(self, q: DM, marker_name: str, zero_name: str | None = None) -> DM:
        """
        Compute the position of a marker given the generalized coordinates

        Parameters
        ----------
        q: DM
            The generalized coordinates of the system
        marker_name: str
            The name of the marker to compute
        zero_name: str | None
            The name of the marker to substract to the marker

        Returns
        -------
        The position of the marker
        """
        q_sym = MX.sym("q", self.nb_q, 1)
        marker = self.marker(self.marker_names.index(marker_name), None)(q_sym, self.parameters)
        if zero_name is not None:
            # zero = self.marker(q_sym, self.marker_names.index(zero_name))
            zero = self.marker(self.marker_names.index(zero_name), None)(q_sym, self.parameters)
            marker = marker - zero
        func = Function("marker", [q_sym], [marker])
        return func(q)

    def compute_key_reaction_forces(self, q: MX | SX) -> MX | SX:
        """
        Compute the external forces based on the position of the finger. The force is an exponential function based on the
        depth of the finger in the key. The force is null if the finger is over the key, the force slowly increases
        during the course of the movement as the finger presses the key, and finally increases drastically as it reaches
        the bottom of the key.

        Parameters
        ----------
        q: MX | SX
            The generalized coordinates of the system

        Returns
        -------
        The external forces in the tuple[MX | SX] format
        """

        finger = self.marker(self.marker_names.index("finger_marker"), None)(q, self.parameters)
        key_top = self.marker(self.marker_names.index("Key1_Top"), None)(q, self.parameters)
        key_bottom = self.marker(self.marker_names.index("key1_base"), None)(q, self.parameters)

        finger_penetration = key_top[2] - finger[2]
        max_penetration = key_top[2] - key_bottom[2]
        force_at_bed = 1
        force_increate_rate = 5e4
        max_force = 30

        x = 0  # This is done via contact
        y = 0  # This is done via contact
        # z = force_at_bed * np.exp(force_increate_rate * (finger_penetration - max_penetration))
        # Temporary until we get actual data
        z = if_else(finger_penetration < 0, 0, 10)  # finger_penetration / max_penetration * max_force)
        px = finger[0]
        py = finger[1]
        pz = finger[2]
        return vertcat(x, y, z, px, py, pz)

    def compute_key_reaction_forces_dm(self, q: DM) -> DM:
        """
        Interface to compute_key_reaction_forces for DM
        """

        q_sym = MX.sym("q", self.nb_q, 1)
        func = Function("forces", [q_sym], [self.compute_key_reaction_forces(q_sym)])
        return func(q)

    def normalized_friction_force(self, q: MX | SX, qdot: MX | SX, tau: MX | SX, mu: float) -> MX | SX:
        """
        Compute the friction force on the key

        Parameters
        ----------
        q: MX | SX
            The generalized coordinates of the system
        qdot: MX | SX
            The generalized velocities of the system
        tau: MX | SX
            The generalized forces of the system
        mu: float
            The friction coefficient

        Returns
        -------
        The friction force in the tuple[MX | SX] format
        """
        forces = self.compute_key_reaction_forces(q)
        forces[2] += 0.00001  # To avoid division by 0

        normalized_contact = self.contact_forces()(q, qdot, tau, forces[2], self.parameters)
        normalized_horizontal_forces_squared = normalized_contact[0] ** 2 + normalized_contact[1] ** 2

        # The horizontal forces are in the cone of friction if this returns [-1, 1]
        return normalized_horizontal_forces_squared / mu**2

    def normalized_friction_force_dm(self, q: DM, qdot: DM, tau: DM, mu: float) -> DM:
        """
        Interface to normalize_friction_force_on_key for DM
        """
        q_sym = MX.sym("q", self.nb_q, 1)
        qdot_sym = MX.sym("qdot", self.nb_q, 1)
        tau_sym = MX.sym("tau", self.nb_tau, 1)
        func = Function(
            "forces", [q_sym, qdot_sym, tau_sym], [self.normalized_friction_force(q_sym, qdot_sym, tau_sym, mu)]
        )
        return func(q, qdot, tau)
