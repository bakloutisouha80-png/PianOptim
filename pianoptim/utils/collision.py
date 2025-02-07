from casadi import solve


def collision_impact(model, q, qdot_minus, e=0):
    """
    Compute the post-collision generalized velocity qdot_plus using CasADi symbolic expressions:

    Arguments:
    ----------
    model :  Your model or system object that exposes:
               - model.massMatrixInverse(q) -> CasADi matrix H_inv
               - constraint_jacobian(q)     -> CasADi matrix G
    q     :  CasADi vector (or DM) of generalized positions
    qdot_minus : CasADi vector (or DM) of pre-collision generalized velocities
    e     :  scalar (coefficient of restitution in [0,1]);
             can be a float or a CasADi symbolic variable

    Returns:
    --------
    qdot_plus : CasADi expression (symbolic or DM) for the post-collision velocity
    """

    # 1) Inverse of the mass matrix M(q).
    M_inv = model.model.massMatrixInverse(q).to_mx()  # shape: (n x n)

    # 2) Constraint Jacobian
    J = model.constraint_jacobian(q)  # shape: (m x n)
    J_T = J.T  # shape: (n x m)

    # 3) Delassus matrix S = J * M^{-1} * J^T
    #    shape: (m x m)
    delassus = J @ M_inv @ J_T

    # 4) Compute the impulse: Lambda = -S^{-1} * (e + 1)*J*qdot_minus
    #    (Use casadi.solve(...) rather than S^{-1} for better numeric stability)
    rhs = (e + 1.0) * (J @ qdot_minus)  # shape: (m,)
    Lambda = -solve(delassus, rhs, "symbolicqr")

    # 5) Finally, compute qdot_plus = qdot_minus + M^{-1} * J^T * Lambda
    qdot_plus = qdot_minus + M_inv @ (J_T @ Lambda)

    return qdot_plus
