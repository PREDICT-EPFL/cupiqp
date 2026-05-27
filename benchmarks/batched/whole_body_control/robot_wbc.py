"""Unified Wbc QP factory across robot types — self-contained.

This module bundles every piece needed to build the batched Wbc QP for
a given robot:

    * QP dimensions and weights (WbcDims, WbcWeights).
    * Block-by-block QP assembly (equality, inequality, cost).
    * Pinocchio-backed dynamics for the supported robots.
    * The class hierarchy and registry callers consume.

There is no external dependency on ``legged_wbc_problem.py``; the
legged QP-assembly helpers (originally in that module) are inlined
below as private module-level functions.

``RobotWbc`` is the abstract base; concrete subclasses fill in dynamics
and problem layout for specific robots:

    Legged (floating base, contact forces):
        - AnymalCWbc    (quadruped, 4 feet, na=12)
        - UnitreeH1Wbc  (humanoid,  2 feet, na=19)

    Fixed-base arm (no contacts, end-effector tracking):
        - Iiwa14ArmWbc  (7-DoF KUKA arm — close analog to ABB IRB1600,
                        which can drop in by subclassing FixedBaseArmWbc
                        once a URDF is available)

Add a new robot by subclassing ``LeggedRobotWbc`` or ``FixedBaseArmWbc``
and overriding the class-level constants (URDF package, foot/EE frame
name(s), actuated-joint count, torque limit).

Glossary (used throughout this module)
--------------------------------------
    na  -- number of *actuated* joints (DoFs with a motor / torque cmd).
           Floating-base robots have ``na = nv - 6`` (the 6 free-flyer
           DoFs are unactuated). Fixed-base arms have ``na = nv``.
                * ANYmal C:   na = 12
                * Unitree H1: na = 19
                * iiwa14:     na = 7

    nv  -- generalized velocity dimension (pinocchio's ``model.nv``).
           For a floating-base robot ``nv = 6 + na`` (3 base trans + 3
           base rot + na joint vels). For a fixed-base robot ``nv = na``.

    nq  -- generalized coordinate dimension (pinocchio's ``model.nq``).
           Differs from ``nv`` when joints use non-Euclidean parametrizations
           (e.g. quaternion free-flyer has nq = nv + 1 because the base
           orientation is stored as a 4-vector quaternion).

    nc  -- number of point contacts (feet). 0 for fixed-base arms.

    u_dot in R^nv     generalized acceleration  (decision variable).
    tau   in R^na     joint torques             (decision variable).
    F     in R^{3*nc} stacked 3-D contact forces (decision, legged only).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import pinocchio as pin

from batched_solver_interface import BatchedQPData


# ===========================================================================
# Dimensions and weights
# ===========================================================================
@dataclass
class WbcDims:
    """Decision-variable and constraint-row counts for the legged Wbc QP.

    Layout reminder::

        x = [ u_dot (nv) ; F (3*nc) ; tau (na) ]   in R^n
        n_eq    = nv  + 3*nc       (EoM + per-foot equation)
        n_ineq  = 4*nc             (lateral friction, ±F_x|y - mu F_z <= 0)
        box     : tau, F_z         (torque limits via x_l/x_u, F_z >= 0 stance)

    The friction *normal* row F_z >= 0 and the torque limits ±tau <= tau_lim
    are box constraints on individual decision variables — they live in
    ``x_l/x_u`` rather than ``G`` so the IPM doesn't carry extra G rows
    for what is really just a per-variable bound.
    """
    n_base: int = 6                    # floating-base DoFs (3 trans + 3 rot)
    n_joints: int = 12                 # na (actuated joints)
    n_contacts: int = 4                # nc (3-D point contacts)

    @property
    def nv(self) -> int:      return self.n_base + self.n_joints
    @property
    def n_F(self) -> int:     return 3 * self.n_contacts
    @property
    def n(self) -> int:       return self.nv + self.n_F + self.n_joints
    @property
    def n_eq(self) -> int:    return self.nv + 3 * self.n_contacts
    @property
    def n_ineq(self) -> int:  return 4 * self.n_contacts

    @property
    def slice_udot(self) -> slice: return slice(0, self.nv)
    @property
    def slice_F(self) -> slice:    return slice(self.nv, self.nv + self.n_F)
    @property
    def slice_tau(self) -> slice:  return slice(self.nv + self.n_F, self.n)


@dataclass
class WbcWeights:
    """Cost-task weights, friction coefficient, and torque limits.

    ``torque_limit`` is a scalar or a length-``na`` vector; the inequality
    builder broadcasts via ``_resolve_torque_limit`` so subclasses can
    either pin a single conservative bound or supply per-joint datasheet
    values.
    """
    w_swing: float = 100.0
    w_base:  float = 10.0
    w_cf:    float = 0.05
    friction_coef: float = 0.7
    torque_limit: Union[float, Sequence[float], np.ndarray] = 80.0
    static_regularization: float = 1e-4


# ===========================================================================
# Generic helpers
# ===========================================================================
def _resolve_torque_limit(limit: Union[float, Sequence[float], np.ndarray],
                          na: int) -> np.ndarray:
    """Broadcast a scalar or a length-``na`` vector to a ``(na,)`` array.

    Replaces the ``np.tile(..., na // len(...))`` pattern from the original
    quadruped code, which silently produced a wrong-length array when
    ``na`` was not a multiple of the sequence length (e.g. H1's na=19).
    """
    arr = np.atleast_1d(np.asarray(limit, dtype=np.float64))
    if arr.size == 1:
        return np.full(na, float(arr[0]), dtype=np.float64)
    if arr.size == na:
        return arr.astype(np.float64, copy=False)
    raise ValueError(
        f"torque_limit must be a scalar or a length-{na} vector; "
        f"got size {arr.size}."
    )


def _load_pinocchio_robot(package_name: str, *, floating_base: bool):
    """Load a URDF from ``robot_descriptions`` (lazy, cached on disk)."""
    from robot_descriptions.loaders.pinocchio import load_robot_description
    root = pin.JointModelFreeFlyer() if floating_base else None
    return load_robot_description(package_name, root_joint=root)


def _force_one_stance(contact_flag: np.ndarray) -> np.ndarray:
    """Ensure every batch row has at least one stance foot (avoids the
    fully-airborne degenerate case where the floating-base EoM becomes
    rank-deficient through F)."""
    no_stance = ~contact_flag.any(axis=1)
    if no_stance.any():
        contact_flag[no_stance, 0] = True
    return contact_flag


def _clipped_normal(rng: np.random.Generator,
                    scale: float,
                    size,
                    clip_sigma: float = 3.0) -> np.ndarray:
    """Sample ``scale * N(0, 1)`` clipped to ``±clip_sigma * scale``.

    Plain Gaussian state perturbations produce ~0.3% of samples beyond 3σ
    (and ~5% beyond 2σ). At B≥1k those long-tail draws land near kinematic
    singularities (M near-singular, J near rank-deficient) and feed the IPM
    a quietly broken QP. Clipping to a fixed multiple of the std-dev kills
    that tail while preserving the bulk of the natural distribution.

    ``size = ()`` returns a 0-D ndarray (use ``float(...)`` to scalarise).
    """
    z = np.clip(rng.standard_normal(size), -clip_sigma, +clip_sigma)
    return scale * z


# ===========================================================================
# Legged Wbc QP assembly  (inlined from the former legged_wbc_problem.py)
# ===========================================================================
def _lateral_friction_pyramid(mu: float):
    """4-row lateral friction in **double-sided** ``h_l <= G F <= h_u`` form.

    The friction cone ``|F_x| <= mu*F_z`` / ``|F_y| <= mu*F_z`` decomposes
    naturally into two pairs of bounds, each pair sharing a G coefficient
    up to sign on ``mu``. Writing the four scalar inequalities

        F_x - mu*F_z <= 0          F_y - mu*F_z <= 0
        F_x + mu*F_z >= 0          F_y + mu*F_z >= 0

    as ``h_l <= G x <= h_u`` uses *both* sides of the constraint rather
    than stacking eight one-sided rows. Layout per foot::

        row 0:  -inf <=  F_x - mu*F_z  <=   0
        row 1:    0  <=  F_x + mu*F_z  <= +inf
        row 2:  -inf <=  F_y - mu*F_z  <=   0
        row 3:    0  <=  F_y + mu*F_z  <= +inf

    The 5th row of the standard pyramid, ``F_z >= 0``, is a box constraint
    handled separately via ``x_l``.

    Returns ``(G_block (4, 3), h_l_block (4,), h_u_block (4,))``.
    """
    G = np.array([
        [ 1.0, 0.0, -mu ],   # F_x - mu*F_z  in  (-inf, 0]
        [ 1.0, 0.0,  mu ],   # F_x + mu*F_z  in  [0, +inf)
        [ 0.0, 1.0, -mu ],   # F_y - mu*F_z  in  (-inf, 0]
        [ 0.0, 1.0,  mu ],   # F_y + mu*F_z  in  [0, +inf)
    ], dtype=np.float64)
    h_l = np.array([-np.inf, 0.0, -np.inf, 0.0], dtype=np.float64)
    h_u = np.array([    0.0, np.inf,   0.0, np.inf], dtype=np.float64)
    return G, h_l, h_u


def _build_legged_equality(M, J, dJv, nle, contact_flag, dims: WbcDims):
    """``(A_eq, b_eq)`` for the legged QP.

    Row layout (``n_eq = nv + 3*nc``)::

        rows 0 .. nv-1            : floating-base EoM
                                    M u_dot - J^T F - S^T tau = -nle
        rows nv .. nv + 3*nc - 1  : per-foot equation, 3 rows per foot
                                    stance i :  J_i u_dot = -dJ_i v
                                    swing  i :  F_i       =  0
    """
    B = M.shape[0]
    nv, n_F, na = dims.nv, dims.n_F, dims.n_joints
    n = dims.n
    nc = dims.n_contacts

    A = np.zeros((B, dims.n_eq, n), dtype=np.float64)
    b = np.zeros((B, dims.n_eq), dtype=np.float64)

    # 1) floating-base EoM
    A[:, :nv, :nv] = M
    A[:, :nv, nv:nv + n_F] = -np.transpose(J, (0, 2, 1))
    # Selection matrix S^T: zero on the 6 base rows, -I on the na joint rows
    # (the 6-DoF floating base is unactuated by definition of ``na``).
    A[:, 6:nv, nv + n_F:] = -np.eye(na)
    b[:, :nv] = -nle

    # 2) per-foot equation: stance -> J_i u_dot = -dJ_i v, swing -> F_i = 0
    for i in range(nc):
        rows = slice(nv + 3 * i, nv + 3 * (i + 1))
        for k in range(B):
            if contact_flag[k, i]:
                A[k, rows, :nv] = J[k, 3 * i:3 * (i + 1), :]
                b[k, rows] = -dJv[k, 3 * i:3 * (i + 1)]
            else:
                A[k, rows, nv + 3 * i:nv + 3 * (i + 1)] = np.eye(3)
                # b stays at 0 here (zero-force constraint).
    return A, b


def _build_legged_inequality(contact_flag, w: WbcWeights, dims: WbcDims):
    """``(G, h_l, h_u)`` for the legged QP, in the double-sided form
    ``h_l <= G x <= h_u``. Row layout (``n_ineq = 4*nc``)::

        rows 4*i .. 4*i + 3 : double-sided lateral friction pyramid on foot i.
                              Stance feet get the actual friction G coefficients;
                              swing feet get zero G coefficients so the rows are
                              vacuously satisfied (G x = 0, and 0 lies inside
                              every (h_l, h_u) pair the pyramid sets).

    The finite-bound *pattern* of ``h_l`` and ``h_u`` is kept uniform across
    the batch (cuPIQP requirement); only G varies per stance/swing pattern.
    """
    B = contact_flag.shape[0]
    nc = dims.n_contacts
    n = dims.n
    n_ineq = dims.n_ineq

    fp_G, fp_hl, fp_hu = _lateral_friction_pyramid(w.friction_coef)

    # h_l / h_u: uniform across the batch — same pyramid bounds on every
    # foot regardless of stance. (Swing feet have G=0, so 0 satisfies every
    # row trivially.)
    h_l = np.tile(np.tile(fp_hl, nc), (B, 1))
    h_u = np.tile(np.tile(fp_hu, nc), (B, 1))

    G = np.zeros((B, n_ineq, n), dtype=np.float64)
    for k in range(B):
        for i in range(nc):
            if contact_flag[k, i]:
                rows = slice(4 * i, 4 * (i + 1))
                cols = slice(dims.nv + 3 * i, dims.nv + 3 * (i + 1))
                G[k, rows, cols] = fp_G
    return G, h_l, h_u


def _build_legged_box(contact_flag, w: WbcWeights, dims: WbcDims):
    """``(x_l, x_u)`` for the legged QP.

    Box-constraint layout (kept *uniform* across the batch — cuPIQP rejects
    a per-batch-varying finite-bound pattern in ``x_l/x_u``)::

        u_dot block (0 : nv)        :  unbounded
        F block     (nv : nv+n_F)   :  per foot (F_x, F_y, F_z)
                                       F_x, F_y :  unbounded
                                                  (lateral friction in G
                                                   couples them to F_z)
                                       F_z      :  [0, +inf)  for every foot
                                                   (swing F is pinned to 0
                                                    by the equality block and
                                                    0 trivially satisfies the
                                                    bound)
        tau block   (nv+n_F : n)    :  [-tau_lim, +tau_lim]   per actuated joint
    """
    B = contact_flag.shape[0]
    na, nc = dims.n_joints, dims.n_contacts
    n = dims.n

    x_l = np.full((B, n), -np.inf, dtype=np.float64)
    x_u = np.full((B, n), +np.inf, dtype=np.float64)

    # F_z >= 0 for every foot (uniform across stance / swing — see docstring).
    for i in range(nc):
        fz_idx = dims.nv + 3 * i + 2
        x_l[:, fz_idx] = 0.0

    # Torque limits: -tau_lim <= tau <= tau_lim (broadcast across the batch).
    tau_lim = _resolve_torque_limit(w.torque_limit, na)         # (na,)
    x_l[:, dims.slice_tau] = -tau_lim
    x_u[:, dims.slice_tau] = +tau_lim
    return x_l, x_u


def _build_legged_cost(J, dJv, p_swing_err, v_swing_err,
                       base_accel_target, F_des, contact_flag,
                       w: WbcWeights, dims: WbcDims,
                       swing_kp: float = 350.0, swing_kd: float = 37.0):
    """``(P, c)`` from three weighted least-squares tasks.

    Tasks:
      * Swing leg:    ``A_sw = [J | 0 | 0]`` sliced to swing-foot rows.
                      b_sw  = K_p (p_des - p) + K_d (v_des - v) - dJ_sw v
      * Base accel:   ``A_ba = [I_6 padded to nv | 0 | 0]``, b_ba = target.
      * Contact force:``A_cf = [0 | I_{3*nc} | 0]``, b_cf = F_des.

    ``P = sum_i w_i A_i^T A_i + static_regularization * I``,
    ``c = -sum_i w_i A_i^T b_i``.
    """
    B = J.shape[0]
    n = dims.n
    nv = dims.nv

    # Swing-leg task. Rows for stance feet are zeroed so they contribute
    # nothing to A^T A or A^T b.
    A_sw = np.zeros((B, 3 * dims.n_contacts, n), dtype=np.float64)
    b_sw = np.zeros((B, 3 * dims.n_contacts), dtype=np.float64)
    for k in range(B):
        for i in range(dims.n_contacts):
            if not contact_flag[k, i]:
                rows = slice(3 * i, 3 * (i + 1))
                A_sw[k, rows, :nv] = J[k, 3 * i:3 * (i + 1), :]
                b_sw[k, rows] = (
                    swing_kp * p_swing_err[k, 3 * i:3 * (i + 1)]
                    + swing_kd * v_swing_err[k, 3 * i:3 * (i + 1)]
                    - dJv[k, 3 * i:3 * (i + 1)]
                )

    # Base-accel task.
    A_ba = np.zeros((B, 6, n), dtype=np.float64)
    A_ba[:, :, :6] = np.eye(6)
    b_ba = base_accel_target

    # Contact-force task.
    A_cf = np.zeros((B, dims.n_F, n), dtype=np.float64)
    A_cf[:, :, dims.slice_F] = np.eye(dims.n_F)
    b_cf = F_des

    A_swT = np.transpose(A_sw, (0, 2, 1))
    A_baT = np.transpose(A_ba, (0, 2, 1))
    A_cfT = np.transpose(A_cf, (0, 2, 1))

    P = (w.w_swing * A_swT @ A_sw
         + w.w_base  * A_baT @ A_ba
         + w.w_cf    * A_cfT @ A_cf
         + w.static_regularization * np.eye(n)[None])

    c = -(w.w_swing * np.einsum("bij,bj->bi", A_swT, b_sw)
          + w.w_base  * np.einsum("bij,bj->bi", A_baT, b_ba)
          + w.w_cf    * np.einsum("bij,bj->bi", A_cfT, b_cf))
    return P, c


def _build_legged_qp(*, M, J, dJv, nle,
                     p_swing_err, v_swing_err,
                     base_accel_target, F_des,
                     contact_flag,
                     weights: WbcWeights, dims: WbcDims) -> BatchedQPData:
    """Assemble the legged-Wbc QP for a batch of states.

    Shapes:
        M       : (B, nv, nv)        joint-space inertia
        J       : (B, 3*nc, nv)      contact Jacobian, 3 rows per foot
        dJv     : (B, 3*nc)          dJ * v
        nle     : (B, nv)            Coriolis + gravity
        p_swing_err, v_swing_err : (B, 3*nc) per-foot cartesian errors
        base_accel_target        : (B, 6)
        F_des                    : (B, 3*nc)
        contact_flag             : (B, nc) bool
    """
    A, b = _build_legged_equality(M, J, dJv, nle, contact_flag, dims)
    G, h_l, h_u = _build_legged_inequality(contact_flag, weights, dims)
    x_l, x_u = _build_legged_box(contact_flag, weights, dims)
    P, c = _build_legged_cost(J, dJv, p_swing_err, v_swing_err,
                              base_accel_target, F_des, contact_flag,
                              weights, dims)
    return BatchedQPData(P=P, c=c, A=A, b=b,
                         G=G, h_l=h_l, h_u=h_u,
                         x_l=x_l, x_u=x_u)


# ===========================================================================
# Abstract base
# ===========================================================================
class RobotWbc(ABC):
    """Abstract batched Wbc QP generator for one robot type."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def n_var(self) -> int: ...

    @property
    @abstractmethod
    def n_eq(self) -> int: ...

    @property
    @abstractmethod
    def n_ineq(self) -> int: ...

    @abstractmethod
    def generate_problems(self, B: int, seed: int = 0) -> tuple[BatchedQPData, dict]:
        """Build ``B`` random Wbc QPs.

        Returns
        -------
        data : BatchedQPData
        info : dict with robot-specific extras (e.g. ``contact_flag`` for
               legged robots; ``q``/``v``/``p_ee_target`` for arms).
        """
        ...

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(name={self.name!r}, "
                f"n={self.n_var}, p={self.n_eq}, m={self.n_ineq})")


# ===========================================================================
# Legged: floating base + N point contacts
# ===========================================================================
class LeggedRobotWbc(RobotWbc):
    """Floating-base robot with N point contacts (quadrupeds, humanoids).

    Subclasses must set ``URDF_PACKAGE``, ``FOOT_FRAMES``, ``N_ACTUATED``,
    and ``TORQUE_LIMIT``. Everything else (dynamics computation, contact
    sampling, QP assembly) is driven from those constants.
    """

    URDF_PACKAGE: str = ""
    FOOT_FRAMES: tuple[str, ...] = ()
    N_ACTUATED: int = 0
    TORQUE_LIMIT: Union[float, np.ndarray] = 80.0
    # State-sampling scales: each *_RAND_SCALE is the std-dev of the
    # Gaussian noise added to the corresponding axis when sampling
    # random states for problem generation. The noise is then clipped
    # to ``±RAND_CLIP_SIGMA * scale`` so we never draw a configuration
    # arbitrarily deep into a kinematic singularity (which would feed
    # a near-rank-deficient ``M``/``J`` to the IPM).
    Q_JOINT_RAND_SCALE: float = 0.15   # std-dev on joint positions, radians
    BASE_Z_RAND_SCALE:  float = 0.03   # std-dev on base z-position,  metres
    BASE_Z_NOMINAL:     float = 0.55   # nominal stance height,       metres
    V_RAND_SCALE:       float = 0.1    # std-dev on generalized velocity
    RAND_CLIP_SIGMA:    float = 2.0    # clip Gaussian noise at this many std-devs
                                       # (2σ ≈ 95% of the natural distribution kept;
                                       # tighter than 3σ default to avoid corner cases
                                       # that destabilise qpax / qpth on ANYmal C)

    def __init__(self, contact_prob: float = 0.7,
                 weights: Optional[WbcWeights] = None):
        if not self.URDF_PACKAGE or not self.FOOT_FRAMES or self.N_ACTUATED <= 0:
            raise TypeError(
                f"{type(self).__name__} must define URDF_PACKAGE, FOOT_FRAMES, "
                f"and N_ACTUATED at the class level."
            )
        self.contact_prob = contact_prob

        # Pinocchio model + data (loaded once, reused for every batch element).
        # ``self.robot`` keeps the full ``robot_descriptions`` wrapper so
        # callers that need ``collision_model`` / ``visual_model`` (e.g. the
        # MeshCat viewer) don't have to reload the URDF.
        self.robot = _load_pinocchio_robot(self.URDF_PACKAGE, floating_base=True)
        self.model = self.robot.model
        self.data = self.model.createData()
        self.foot_frame_ids = [self.model.getFrameId(n) for n in self.FOOT_FRAMES]

        nv = self.model.nv
        nc = len(self.foot_frame_ids)
        na = self.N_ACTUATED
        n_base = nv - na
        if n_base != 6:
            raise RuntimeError(
                f"{type(self).__name__}: expected 6-DoF floating base, "
                f"got n_base={n_base} (nv={nv}, na={na}); "
                "is N_ACTUATED set correctly?"
            )

        self.dims = WbcDims(n_base=n_base, n_joints=na, n_contacts=nc)

        # Build a WbcWeights with the per-joint torque vector resolved to
        # the right length; the inequality builder calls
        # ``_resolve_torque_limit(w.torque_limit, na)`` anyway, so passing
        # it pre-broadcast is just future-proofing.
        if weights is None:
            weights = WbcWeights()
        weights.torque_limit = _resolve_torque_limit(self.TORQUE_LIMIT, na)
        self.weights = weights

        # Nominal stance configuration.
        self.q0 = pin.neutral(self.model)
        self.q0[2] = self.BASE_Z_NOMINAL

        # Cached gravity & mass (for the contact-force target).
        self._total_mass = pin.computeTotalMass(self.model)
        self._g_norm = abs(self.model.gravity.linear[2])

    # ----- RobotWbc interface -----
    @property
    def name(self) -> str:
        return type(self).__name__.replace("Wbc", "").lower()

    @property
    def n_var(self) -> int: return self.dims.n

    @property
    def n_eq(self) -> int: return self.dims.n_eq

    @property
    def n_ineq(self) -> int: return self.dims.n_ineq

    # ----- Dynamics -----
    def _compute_state_dynamics(self, q: np.ndarray, v: np.ndarray):
        """Return ``(M, J_stack, dJv_stack, nle, foot_pos, foot_vel)`` for one state."""
        model, data = self.model, self.data
        pin.forwardKinematics(model, data, q, v)
        pin.computeJointJacobians(model, data, q)
        pin.computeJointJacobiansTimeVariation(model, data, q, v)
        pin.updateFramePlacements(model, data)
        pin.crba(model, data, q)
        # CRBA only fills the upper triangle.
        data.M[np.tril_indices_from(data.M, k=-1)] = (
            data.M.T[np.tril_indices_from(data.M, k=-1)]
        )
        pin.nonLinearEffects(model, data, q, v)

        nv = model.nv
        nc = len(self.foot_frame_ids)
        J = np.zeros((3 * nc, nv), dtype=np.float64)
        dJv = np.zeros(3 * nc, dtype=np.float64)
        foot_pos = np.zeros(3 * nc, dtype=np.float64)
        foot_vel = np.zeros(3 * nc, dtype=np.float64)
        for i, fid in enumerate(self.foot_frame_ids):
            J_i = pin.getFrameJacobian(model, data, fid, pin.LOCAL_WORLD_ALIGNED)
            dJ_i = pin.getFrameJacobianTimeVariation(
                model, data, fid, pin.LOCAL_WORLD_ALIGNED,
            )
            vel_i = pin.getFrameVelocity(model, data, fid, pin.LOCAL_WORLD_ALIGNED)

            sl = slice(3 * i, 3 * (i + 1))
            J[sl] = J_i[:3]
            dJv[sl] = dJ_i[:3] @ v
            foot_pos[sl] = data.oMf[fid].translation
            foot_vel[sl] = vel_i.linear
        return data.M.copy(), J, dJv, data.nle.copy(), foot_pos, foot_vel

    def _sample_state(self, rng: np.random.Generator):
        clip = self.RAND_CLIP_SIGMA
        q = self.q0.copy()
        q[2] += float(_clipped_normal(rng, self.BASE_Z_RAND_SCALE, (), clip))
        # quaternion stays at identity in q[3:7]; only joint positions are
        # randomized. ``nq - 7`` because the free-flyer takes 7 of the nq
        # slots (3 xyz + 4 quaternion).
        q[7:] = self.q0[7:] + _clipped_normal(
            rng, self.Q_JOINT_RAND_SCALE, self.model.nq - 7, clip,
        )
        v = _clipped_normal(rng, self.V_RAND_SCALE, self.model.nv, clip)
        return q, v

    # ----- Problem generation -----
    def generate_problems(self, B: int, seed: int = 0) -> tuple[BatchedQPData, dict]:
        rng = np.random.default_rng(seed)
        nv, nc = self.dims.nv, self.dims.n_contacts

        M_b   = np.zeros((B, nv, nv), dtype=np.float64)
        J_b   = np.zeros((B, 3 * nc, nv), dtype=np.float64)
        dJv_b = np.zeros((B, 3 * nc), dtype=np.float64)
        nle_b = np.zeros((B, nv), dtype=np.float64)
        fp_b  = np.zeros((B, 3 * nc), dtype=np.float64)
        fv_b  = np.zeros((B, 3 * nc), dtype=np.float64)
        q_b   = np.zeros((B, self.model.nq), dtype=np.float64)
        v_b   = np.zeros((B, nv), dtype=np.float64)

        for k in range(B):
            q, v = self._sample_state(rng)
            q_b[k] = q
            v_b[k] = v
            M_b[k], J_b[k], dJv_b[k], nle_b[k], fp_b[k], fv_b[k] = (
                self._compute_state_dynamics(q, v)
            )

        # Swing-leg PD targets: small offset from current foot position.
        p_des = fp_b + rng.uniform(-0.05, 0.05, size=fp_b.shape)
        v_des = rng.uniform(-0.2, 0.2, size=fv_b.shape)
        p_swing_err = p_des - fp_b
        v_swing_err = v_des - fv_b

        base_accel_target = np.zeros((B, 6), dtype=np.float64)
        base_accel_target[:, :3] = rng.uniform(-1.0, 1.0, size=(B, 3))
        base_accel_target[:, 3:] = rng.uniform(-0.5, 0.5, size=(B, 3))

        # Random per-foot contact mask, then force at least one stance.
        contact_flag = rng.random((B, nc)) < self.contact_prob
        contact_flag = _force_one_stance(contact_flag)
        n_stance = contact_flag.sum(axis=1).astype(np.float64)

        # Gravity-comp contact-force reference: f_z = m*g / n_stance per stance foot.
        F_des = np.zeros((B, 3 * nc), dtype=np.float64)
        f_z = self._total_mass * self._g_norm / n_stance
        for i in range(nc):
            F_des[:, 3 * i + 2] = np.where(contact_flag[:, i], f_z, 0.0)

        data = _build_legged_qp(
            M=M_b, J=J_b, dJv=dJv_b, nle=nle_b,
            p_swing_err=p_swing_err, v_swing_err=v_swing_err,
            base_accel_target=base_accel_target, F_des=F_des,
            contact_flag=contact_flag,
            weights=self.weights, dims=self.dims,
        )
        return data, {
            "contact_flag": contact_flag,
            "q": q_b, "v": v_b,
            "foot_pos": fp_b, "foot_vel": fv_b,
        }


class AnymalCWbc(LeggedRobotWbc):
    """ANYmal C quadruped (4 point feet, na=12, nv=18)."""
    URDF_PACKAGE = "anymal_c_description"
    FOOT_FRAMES = ("LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT")
    N_ACTUATED = 12
    TORQUE_LIMIT = 80.0   # ~ ANYdrive abs torque limit
    BASE_Z_NOMINAL = 0.55


class UnitreeH1Wbc(LeggedRobotWbc):
    """Unitree H1 humanoid (2 ankle contacts, na=19, nv=25)."""
    URDF_PACKAGE = "h1_description"
    FOOT_FRAMES = ("left_ankle_link", "right_ankle_link")
    N_ACTUATED = 19
    # H1 torque ranges from ~40 N·m (small joints) to ~360 N·m (knees);
    # use a uniform conservative bound for the benchmark.
    TORQUE_LIMIT = 200.0
    BASE_Z_NOMINAL = 1.0
    Q_JOINT_RAND_SCALE = 0.1


# ===========================================================================
# Fixed-base arm: no floating base, no contacts, end-effector tracking
# ===========================================================================
class FixedBaseArmWbc(RobotWbc):
    """Fixed-base manipulator Wbc.

    Decision variable:  ``x = [u_dot ; tau]``  in  ``R^{2*na}``,  ``na = nv``.

    Constraints:
        Equality  (EoM):       ``M u_dot - tau = -nle``        (na rows)
        Inequality (torque):   ``-tau_lim <= tau <= tau_lim``  (2*na rows)

    Cost (weighted least squares):
        * end-effector linear-accel tracking   ``w_ee  || J_ee u_dot - b_ee ||^2``
                                               with ``b_ee = a_ee_des - dJ_ee v``
        * joint posture                        ``w_post || u_dot - u_dot_ref ||^2``
        * torque regularization                ``w_tau || tau ||^2``
    """

    URDF_PACKAGE: str = ""
    EE_FRAME: str = ""
    N_ACTUATED: int = 0
    TORQUE_LIMIT: Union[float, np.ndarray] = 100.0

    # Default cost weights & joint controller gains (override per robot if needed).
    W_EE: float = 100.0
    W_POSTURE: float = 1.0
    W_TAU: float = 1e-3
    POSTURE_KP: float = 25.0
    POSTURE_KD: float = 10.0
    # State-sampling scales (std-dev of the noise added when drawing random
    # states). Same convention as the legged side: *_RAND_SCALE. Clipped
    # at ``RAND_CLIP_SIGMA`` std-devs to avoid extreme configurations.
    Q_RAND_SCALE: float = 0.3   # std-dev on joint positions, radians
    V_RAND_SCALE: float = 0.5   # std-dev on joint velocities
    RAND_CLIP_SIGMA: float = 2.0   # ≈ 95% of natural distribution kept
    # Diagonal regularization ``+ ε·I`` added to P. See the analogous comment
    # on ``WbcWeights.static_regularization`` for the rationale and tuning
    # — iiwa14 doesn't hit the cuDSS-pivot pathology because its sparsity
    # is uniform across the batch (no contact_flag), but we keep the same
    # default so the fixed-base and legged paths look symmetric.
    STATIC_REGULARIZATION: float = 5e-6

    def __init__(self):
        if not self.URDF_PACKAGE or not self.EE_FRAME or self.N_ACTUATED <= 0:
            raise TypeError(
                f"{type(self).__name__} must define URDF_PACKAGE, EE_FRAME, "
                f"and N_ACTUATED at the class level."
            )

        robot = _load_pinocchio_robot(self.URDF_PACKAGE, floating_base=False)
        self.model = robot.model
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId(self.EE_FRAME)

        na = self.N_ACTUATED
        if self.model.nv != na:
            raise RuntimeError(
                f"{type(self).__name__}: expected nv == N_ACTUATED for fixed-base "
                f"arm, got nv={self.model.nv}, N_ACTUATED={na}."
            )

        self.na = na
        self.n = 2 * na  # [u_dot, tau]
        self.tau_lim = _resolve_torque_limit(self.TORQUE_LIMIT, na)
        self.q0 = pin.neutral(self.model)

    # ----- RobotWbc interface -----
    @property
    def name(self) -> str:
        return type(self).__name__.replace("ArmWbc", "").replace("Wbc", "").lower()

    @property
    def n_var(self) -> int: return self.n

    @property
    def n_eq(self) -> int: return self.na

    @property
    def n_ineq(self) -> int: return 0   # arm: torque limits live in x_l/x_u

    # ----- Dynamics -----
    def _compute_state_dynamics(self, q: np.ndarray, v: np.ndarray):
        """Return ``(M, J_ee_lin, dJv_ee_lin, nle, ee_pos, ee_vel_lin)``."""
        model, data = self.model, self.data
        pin.forwardKinematics(model, data, q, v)
        pin.computeJointJacobians(model, data, q)
        pin.computeJointJacobiansTimeVariation(model, data, q, v)
        pin.updateFramePlacements(model, data)
        pin.crba(model, data, q)
        data.M[np.tril_indices_from(data.M, k=-1)] = (
            data.M.T[np.tril_indices_from(data.M, k=-1)]
        )
        pin.nonLinearEffects(model, data, q, v)

        fid = self.ee_frame_id
        J6 = pin.getFrameJacobian(model, data, fid, pin.LOCAL_WORLD_ALIGNED)
        dJ6 = pin.getFrameJacobianTimeVariation(
            model, data, fid, pin.LOCAL_WORLD_ALIGNED,
        )
        vel = pin.getFrameVelocity(model, data, fid, pin.LOCAL_WORLD_ALIGNED)

        return (data.M.copy(), J6[:3].copy(), dJ6[:3] @ v,
                data.nle.copy(),
                data.oMf[fid].translation.copy(), vel.linear.copy())

    def _sample_state(self, rng: np.random.Generator):
        clip = self.RAND_CLIP_SIGMA
        q = self.q0 + _clipped_normal(rng, self.Q_RAND_SCALE, self.model.nq, clip)
        v = _clipped_normal(rng, self.V_RAND_SCALE, self.model.nv, clip)
        return q, v

    # ----- Problem generation -----
    def generate_problems(self, B: int, seed: int = 0) -> tuple[BatchedQPData, dict]:
        rng = np.random.default_rng(seed)
        na = self.na
        n = self.n  # 2 * na

        M_b   = np.zeros((B, na, na), dtype=np.float64)
        J_b   = np.zeros((B, 3, na), dtype=np.float64)
        dJv_b = np.zeros((B, 3), dtype=np.float64)
        nle_b = np.zeros((B, na), dtype=np.float64)
        ep_b  = np.zeros((B, 3), dtype=np.float64)
        ev_b  = np.zeros((B, 3), dtype=np.float64)
        q_b   = np.zeros((B, self.model.nq), dtype=np.float64)
        v_b   = np.zeros((B, na), dtype=np.float64)

        for k in range(B):
            q, v = self._sample_state(rng)
            q_b[k] = q
            v_b[k] = v
            M_b[k], J_b[k], dJv_b[k], nle_b[k], ep_b[k], ev_b[k] = (
                self._compute_state_dynamics(q, v)
            )

        # EE Cartesian-accel target (translational only).
        p_des = ep_b + rng.uniform(-0.05, 0.05, size=ep_b.shape)
        v_des = rng.uniform(-0.2, 0.2, size=ev_b.shape)
        kp_ee = 100.0
        kd_ee = 20.0
        a_ee_des = kp_ee * (p_des - ep_b) + kd_ee * (v_des - ev_b)
        # b_ee = a_ee_des - dJv (so the task is || J_ee u_dot - b_ee || )
        b_ee = a_ee_des - dJv_b

        # Joint posture reference acceleration:  u_dot_ref = -Kp(q - q_des) - Kd v
        q_des = self.q0[None, :].repeat(B, axis=0)
        # For fixed-base nq == nv, so q errors map 1:1 onto joint axes.
        u_dot_ref = -self.POSTURE_KP * (q_b - q_des) - self.POSTURE_KD * v_b

        # Assemble per-batch P, c, A, b, G, h_u.
        # Indices: u_dot occupies [0:na), tau occupies [na:2*na).
        I = np.eye(na, dtype=np.float64)

        # Cost contributions:
        # 1)  EE tracking on u_dot:    w_ee * || [J | 0] x - b_ee ||^2
        # 2)  posture on u_dot:        w_post * || [I | 0] x - u_dot_ref ||^2
        # 3)  torque regularization:   w_tau  * || [0 | I] x ||^2
        P = np.zeros((B, n, n), dtype=np.float64)
        c = np.zeros((B, n), dtype=np.float64)

        # u_dot - u_dot block:  w_ee J^T J  +  w_post I
        JT_J = np.einsum("bji,bjk->bik", J_b, J_b)  # (B, na, na)
        P[:, :na, :na] = self.W_EE * JT_J + self.W_POSTURE * I[None]
        # tau - tau block: w_tau * I
        P[:, na:, na:] = self.W_TAU * I[None]
        # Static regularization on P (see class-level STATIC_REGULARIZATION).
        P += self.STATIC_REGULARIZATION * np.eye(n)[None]

        # Linear term c = -gradient.
        c[:, :na] = (-self.W_EE * np.einsum("bji,bj->bi", J_b, b_ee)
                     - self.W_POSTURE * u_dot_ref)
        # tau side: 0 (torque regularization is to zero).

        # Equality:  M u_dot - tau = -nle
        A = np.zeros((B, na, n), dtype=np.float64)
        A[:, :, :na] = M_b
        A[:, :, na:] = -I[None]
        b = -nle_b

        # Torque limits as box constraints:  -tau_lim <= tau <= tau_lim.
        # No G/h_u block — the arm has no other inequality constraints.
        x_l = np.full((B, n), -np.inf, dtype=np.float64)
        x_u = np.full((B, n), +np.inf, dtype=np.float64)
        x_l[:, na:] = -self.tau_lim
        x_u[:, na:] = +self.tau_lim

        data = BatchedQPData(P=P, c=c, A=A, b=b, x_l=x_l, x_u=x_u)
        return data, {"q": q_b, "v": v_b, "p_ee_target": p_des}


class Iiwa14ArmWbc(FixedBaseArmWbc):
    """KUKA LBR iiwa14 (7-DoF arm).

    A close structural analog to a 6-DoF ABB IRB1600: same fixed-base
    decision-variable layout, same cost/constraint shape. Drop in an
    ABB URDF by subclassing ``FixedBaseArmWbc`` with the corresponding
    package name and end-effector frame.
    """
    URDF_PACKAGE = "iiwa14_description"
    EE_FRAME = "iiwa_link_ee"
    N_ACTUATED = 7
    # iiwa14 datasheet abs torque limits, in N·m (per-joint).
    TORQUE_LIMIT = np.array([320., 320., 176., 176., 110., 40., 40.])


# ===========================================================================
# Registry
# ===========================================================================
ROBOT_REGISTRY: dict[str, type[RobotWbc]] = {
    "anymal_c": AnymalCWbc,
    "h1":       UnitreeH1Wbc,
    "iiwa14":   Iiwa14ArmWbc,
}


def make_robot(name: str, **kwargs) -> RobotWbc:
    """Build a robot Wbc generator by name. ``kwargs`` are forwarded
    to the concrete subclass constructor (e.g. ``contact_prob`` for legged)."""
    if name not in ROBOT_REGISTRY:
        choices = ", ".join(sorted(ROBOT_REGISTRY))
        raise ValueError(f"unknown robot {name!r}; choose from: {choices}")
    return ROBOT_REGISTRY[name](**kwargs)


def list_robots() -> list[str]:
    return sorted(ROBOT_REGISTRY)
