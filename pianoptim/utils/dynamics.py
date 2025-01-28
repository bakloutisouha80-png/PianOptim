from bioptim import (
    OptimalControlProgram,
    NonLinearProgram,
    ConfigureProblem,
    DynamicsEvaluation,
    DynamicsFunctions,
    PenaltyController,
)
from casadi import MX, SX, vertcat

from ..models.pianist import Pianist


class PianistDyanmics:
    @staticmethod
    def configure_forward_dynamics_with_external_forces(
        ocp: OptimalControlProgram, nlp: NonLinearProgram, numerical_data_timeseries=None
    ):
        """
        Tell the program which variables are states and controls.
        The user is expected to use the ConfigureProblem.configure_xxx functions.

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        """

        ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)
        ConfigureProblem.configure_dynamics_function(ocp, nlp, PianistDyanmics.forward_dynamics_with_external_forces)

        ConfigureProblem.configure_contact_function(ocp, nlp, DynamicsFunctions.forces_from_torque_driven)

    @staticmethod
    def forward_dynamics_with_external_forces(
        time: MX | SX,
        states: MX | SX,
        controls: MX | SX,
        parameters: MX | SX,
        algebraic_states: MX | SX,
        numerical_timeseries: MX | SX,
        nlp: NonLinearProgram,
    ) -> DynamicsEvaluation:
        """
        The custom dynamics function that provides the derivative of the states: dxdt = f(x, u, p)

        Parameters
        ----------
        time: MX | SX
            The time of the system
        states: MX | SX
            The state of the system
        controls: MX | SX
            The controls of the system
        parameters: MX | SX
            The parameters acting on the system
        algebraic_states: MX | SX
            The algebraic states of the system
        nlp: NonLinearProgram
            A reference to the phase
        my_additional_factor: int
            An example of an extra parameter sent by the user

        Returns
        -------
        The derivative of the states in the tuple[MX | SX] format
        """

        model: Pianist = nlp.model

        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tau = DynamicsFunctions.get(nlp.controls["tau"], controls)
        translational_force = model.compute_key_reaction_forces(q)

        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        ddq = model.forward_dynamics(with_contact=True)(q, qdot, tau, translational_force[2], parameters)

        return DynamicsEvaluation(dxdt=vertcat(dq, ddq), defects=None)

    @staticmethod
    def normalized_friction_force(
        controller: PenaltyController,
        mu: float,
    ):
        """
        Add a constraint of static friction at contact points constraining for small tangential forces.
        This function make the assumption that normal_force is always positive
        That is mu*normal_force = tangential_force. To prevent from using a square root, the previous
        equation is squared

        Parameters
        ----------
        constraint: Constraint
            The actual constraint to declare
        controller: PenaltyController
            The penalty node elements
        """
        model: Pianist = controller.get_nlp.model
        return model.normalized_friction_force(
            controller.states["q"].cx_start,
            controller.states["qdot"].cx_start,
            controller.controls["tau"].cx_start,
            mu=mu,
        )


def configure_holonomic_torque_driven_free_qv(
    ocp: OptimalControlProgram, nlp: NonLinearProgram, numerical_data_timeseries=None
):
    """
    Tell the program which variables are states and controls.
    The user is expected to use the ConfigureProblem.configure_xxx functions.

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    """

    name = "q_u"
    names_u = [nlp.model.name_dof[i] for i in nlp.model.independent_joint_index]
    ConfigureProblem.configure_new_variable(
        name,
        names_u,
        ocp,
        nlp,
        True,
        False,
        False,
    )

    name = "q_v"
    names_v = [nlp.model.name_dof[i] for i in nlp.model.dependent_joint_index]
    ConfigureProblem.configure_new_variable(
        name,
        names_v,
        ocp,
        nlp,
        False,
        False,
        False,
        as_algebraic_states=True,
    )

    name = "qdot_u"
    names_qdot = ConfigureProblem._get_kinematics_based_names(nlp, "qdot")
    names_udot = [names_qdot[i] for i in nlp.model.independent_joint_index]
    ConfigureProblem.configure_new_variable(
        name,
        names_udot,
        ocp,
        nlp,
        True,
        False,
        False,
        # NOTE: not ready for phase mapping yet as it is based on dofnames of the class BioModel
        # see _set_kinematic_phase_mapping method
        # axes_idx=ConfigureProblem._apply_phase_mapping(ocp, nlp, name),
    )

    ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)

    # extra plots
    ConfigureProblem.configure_qv(ocp, nlp, nlp.model.compute_q_v)
    ConfigureProblem.configure_qdotv(ocp, nlp, nlp.model._compute_qdot_v)
    ConfigureProblem.configure_lagrange_multipliers_function(ocp, nlp, nlp.model.compute_the_lagrangian_multipliers)

    ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.holonomic_torque_driven)


def holonomic_torque_driven_free_qv(
    time,
    states,
    controls,
    parameters,
    algebraic_states,
    numerical_timeseries,
    nlp,
) -> DynamicsEvaluation:
    """
    The custom dynamics function that provides the derivative of the states: dxdt = f(t, x, u, p, a, d)

    Parameters
    ----------
    time: MX.sym | SX.sym
        The time of the system
    states: MX.sym | SX.sym
        The state of the system
    controls: MX.sym | SX.sym
        The controls of the system
    parameters: MX.sym | SX.sym
        The parameters acting on the system
    algebraic_states: MX.sym | SX.sym
        The algebraic states of the system
    numerical_timeseries: MX.sym | SX.sym
        The numerical timeseries of the system
    nlp: NonLinearProgram
        A reference to the phase

    Returns
    -------
    The derivative of the states in the tuple[MX | SX] format
    """

    q_u = DynamicsFunctions.get(nlp.states["q_u"], states)
    qdot_u = DynamicsFunctions.get(nlp.states["qdot_u"], states)
    tau = DynamicsFunctions.get(nlp.controls["tau"], controls)
    q_v_init = DynamicsFunctions.get(nlp.controls["q_v_init"], controls)
    qddot_u = nlp.model.partitioned_forward_dynamics()(q_u, qdot_u, q_v_init, tau)

    return DynamicsEvaluation(dxdt=vertcat(qdot_u, qddot_u), defects=None)


def holonomic_torque_driven_with_qv(
    time,
    states,
    controls,
    parameters,
    algebraic_states,
    numerical_timeseries,
    nlp,
) -> DynamicsEvaluation:
    """
    The custom dynamics function that provides the derivative of the states: dxdt = f(t, x, u, p, a, d)

    Parameters
    ----------
    time: MX.sym | SX.sym
        The time of the system
    states: MX.sym | SX.sym
        The state of the system
    controls: MX.sym | SX.sym
        The controls of the system
    parameters: MX.sym | SX.sym
        The parameters acting on the system
    algebraic_states: MX.sym | SX.sym
        The algebraic states of the system
    numerical_timeseries: MX.sym | SX.sym
        The numerical timeseries of the system
    nlp: NonLinearProgram
        A reference to the phase

    Returns
    -------
    The derivative of the states in the tuple[MX | SX] format
    """

    q_u = DynamicsFunctions.get(nlp.states["q_u"], states)
    qdot_u = DynamicsFunctions.get(nlp.states["qdot_u"], states)
    tau = DynamicsFunctions.get(nlp.controls["tau"], controls)
    q_v = DynamicsFunctions.get(nlp.algebraic_states["q_v"], algebraic_states)
    qddot_u = nlp.model.partitioned_forward_dynamics_with_qv()(q_u, q_v, qdot_u, tau)

    return DynamicsEvaluation(dxdt=vertcat(qdot_u, qddot_u), defects=None)


def holonomic_torque_driven_custom_qv_init(
    time,
    states,
    controls,
    parameters,
    algebraic_states,
    numerical_timeseries,
    nlp,
    custom_q_v_init,
) -> DynamicsEvaluation:
    """
    The custom dynamics function that provides the derivative of the states: dxdt = f(t, x, u, p, a, d)

    Parameters
    ----------
    time: MX.sym | SX.sym
        The time of the system
    states: MX.sym | SX.sym
        The state of the system
    controls: MX.sym | SX.sym
        The controls of the system
    parameters: MX.sym | SX.sym
        The parameters acting on the system
    algebraic_states: MX.sym | SX.sym
        The algebraic states of the system
    numerical_timeseries: MX.sym | SX.sym
        The numerical timeseries of the system
    nlp: NonLinearProgram
        A reference to the phase
    custom_q_v_init: np.ndarray
        The initial guess for each newton descent

    Returns
    -------
    The derivative of the states in the tuple[MX | SX] format
    """

    q_u = DynamicsFunctions.get(nlp.states["q_u"], states)
    qdot_u = DynamicsFunctions.get(nlp.states["qdot_u"], states)
    tau = DynamicsFunctions.get(nlp.controls["tau"], controls)
    qddot_u = nlp.model.partitioned_forward_dynamics()(q_u, qdot_u, custom_q_v_init, tau)

    return DynamicsEvaluation(dxdt=vertcat(qdot_u, qddot_u), defects=None)


def configure_holonomic_torque_driven(
    ocp: OptimalControlProgram,
    nlp: NonLinearProgram,
    custom_q_v_init=None,
    numerical_data_timeseries=None,
):
    """
    Tell the program which variables are states and controls.
    The user is expected to use the ConfigureProblem.configure_xxx functions.

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    """

    name = "q_u"
    names_u = [nlp.model.name_dof[i] for i in nlp.model.independent_joint_index]
    ConfigureProblem.configure_new_variable(
        name,
        names_u,
        ocp,
        nlp,
        True,
        False,
        False,
    )

    name = "q_v"
    names_v = [nlp.model.name_dof[i] for i in nlp.model.dependent_joint_index]
    ConfigureProblem.configure_new_variable(
        name,
        names_v,
        ocp,
        nlp,
        False,
        False,
        False,
        as_algebraic_states=True,
    )

    name = "qdot_u"
    names_qdot = ConfigureProblem._get_kinematics_based_names(nlp, "qdot")
    names_udot = [names_qdot[i] for i in nlp.model.independent_joint_index]
    ConfigureProblem.configure_new_variable(
        name,
        names_udot,
        ocp,
        nlp,
        True,
        False,
        False,
        # NOTE: not ready for phase mapping yet as it is based on dofnames of the class BioModel
        # see _set_kinematic_phase_mapping method
        # axes_idx=ConfigureProblem._apply_phase_mapping(ocp, nlp, name),
    )

    ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)

    # extra plots
    ConfigureProblem.configure_qv(ocp, nlp, nlp.model.compute_q_v)
    ConfigureProblem.configure_qdotv(ocp, nlp, nlp.model._compute_qdot_v)
    ConfigureProblem.configure_lagrange_multipliers_function(ocp, nlp, nlp.model.compute_the_lagrangian_multipliers)

    ConfigureProblem.configure_dynamics_function(
        ocp, nlp, holonomic_torque_driven_custom_qv_init, custom_q_v_init=custom_q_v_init
    )
