"""Robot side of the batched CBF safety filter example (batched_robot_arm_cbf.ipynb).

Everything here is about the robot, not about the QP solver: the Franka Panda
model in Newton, its collision spheres, the scene of every arm, the Warp
kernels that write the safety filter QP of every arm from Newton's kinematics,
the closed-loop integration and logging, and the plots and the video. The
notebook only creates the QP solver and calls it.

Requires Newton with its simulation extras, robot_descriptions, matplotlib,
mediapy, Pillow and pyglet.
"""
import os
import xml.etree.ElementTree as ET

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import newton
import warp as wp
from PIL import Image, ImageDraw, ImageFont
from robot_descriptions import panda_mj_description


# ---------------------------------------------------------------------------
# The Panda in Newton
# ---------------------------------------------------------------------------

def panda_mjcf():
    """The Menagerie Panda as a 7-joint arm with a rigid, closed gripper."""
    path = panda_mj_description.MJCF_PATH
    root = ET.parse(path).getroot()
    root.find("compiler").set("meshdir", os.path.join(os.path.dirname(path), "assets"))
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "joint" and child.get("name", "").startswith("finger_joint"):
                parent.remove(child)
    for tag in ("tendon", "equality", "actuator", "keyframe"):
        for element in root.findall(tag):
            root.remove(element)
    return ET.tostring(root, encoding="unicode")


_arm = newton.ModelBuilder()
_arm.add_mjcf(panda_mjcf())


def build_model(B):
    """A Newton model with ``B`` Panda arms, one per world, on a ground plane (the table top)."""
    builder = newton.ModelBuilder()
    builder.replicate(_arm, B)
    builder.add_ground_plane()
    return builder.finalize()


na = _arm.joint_dof_count                                                    # joints
NB = _arm.body_count                                                         # bodies per arm
LINK = {label.split("/")[-1]: i for i, label in enumerate(_arm.body_label)}  # body index of every link, by name
HAND = LINK["hand"]
TCP_OFFSET = wp.constant(wp.vec3(0.0, 0.0, 0.1034))   # tool center point between the fingertips, in the hand frame
DT = 0.004                                            # 250 Hz control

# Joint limits of the model, Panda datasheet velocity limits (rad/s), and the "home" posture of the
# Menagerie model.
Q_MIN = np.array(_arm.joint_limit_lower)
Q_MAX = np.array(_arm.joint_limit_upper)
U_MAX = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
Q_HOME = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

# Collision model of OSCBF (https://github.com/StanfordASL/oscbf, MIT license):
# (link, center in the link frame [m], radius [m]). The spheres of link7 also cover the hand.
COLLISION_SPHERES = [
    ("link1", (0.0, 0.0, -0.15), 0.06),
    ("link1", (0.0, -0.065, 0.0), 0.06),
    ("link2", (0.0, 0.0, 0.065), 0.06),
    ("link2", (0.0, -0.14, 0.0), 0.06),
    ("link3", (0.0, 0.0, -0.065), 0.06),
    ("link3", (0.08, 0.065, 0.0), 0.055),
    ("link4", (0.0, 0.0, 0.065), 0.055),
    ("link4", (-0.08, 0.08, 0.0), 0.055),
    ("link5", (0.0, 0.0, -0.23), 0.06),
    ("link5", (0.0, 0.06, -0.18), 0.04),
    ("link5", (0.0, 0.08, -0.125), 0.025),
    ("link5", (0.0, 0.09, -0.075), 0.025),
    ("link5", (0.0, 0.08, 0.0), 0.055),
    ("link6", (0.0, 0.0, 0.01), 0.05),
    ("link6", (0.08, 0.035, 0.0), 0.05),
    ("link7", (0.0, 0.0, 0.08), 0.05),
    ("link7", (0.04, 0.04, 0.09), 0.04),
    ("link7", (0.055, 0.055, 0.15), 0.03),
    ("link7", (-0.055, -0.055, 0.15), 0.03),
    ("link7", (-0.055, -0.055, 0.11), 0.03),
    ("link7", (0.0, 0.0, 0.20), 0.02),
]
NS = len(COLLISION_SPHERES)
SPHERE_BODY = wp.array([LINK[link] for link, _, _ in COLLISION_SPHERES], dtype=int)
SPHERE_CENTER = wp.array([center for _, center, _ in COLLISION_SPHERES], dtype=wp.vec3)
SPHERE_RADIUS = wp.array([radius for _, _, radius in COLLISION_SPHERES], dtype=float)

# Cost weights, task and posture gains [1/s], CBF rates [1/s], and the safety margin delta [m],
# shared by all arms.
W_POS, W_ROT, W_POST = 1.0, 0.1, 1e-3
K_POS, K_ROT, K_POST = 10.0, 10.0, 1.0
ALPHA, ALPHA_JOINT = 10.0, 10.0
MARGIN = 0.005


# ---------------------------------------------------------------------------
# Kernels: kinematics and the QP of every arm
# ---------------------------------------------------------------------------

@wp.func
def point_jacobian_column(
    J: wp.array3d(dtype=float),            # type: ignore  # Jacobians of all arms from newton.eval_jacobian, (B, links * 6, na)
    body_q: wp.array(dtype=wp.transform),  # type: ignore  # world poses of all bodies, at the configuration of J
    body_com: wp.array(dtype=wp.vec3),     # type: ignore  # center of mass of every body, in the body frame
    w: int,                                # type: ignore  # arm (world) index
    k: int,                                # type: ignore  # body index within the arm
    x: wp.vec3,                            # type: ignore  # world position of a point fixed to body k
    j: int,                                # type: ignore  # joint index
):
    """Column ``j`` of the Jacobian of point ``x`` on body ``k`` of arm ``w``: linear and angular part."""
    X = body_q[w * NB + k]
    r = x - wp.transform_point(X, body_com[w * NB + k])
    row = k * 6
    lin = wp.vec3(J[w, row + 0, j], J[w, row + 1, j], J[w, row + 2, j])
    ang = wp.vec3(J[w, row + 3, j], J[w, row + 4, j], J[w, row + 5, j])
    return lin + wp.cross(ang, r), ang


@wp.kernel
def tcp_pose(
    body_q: wp.array(dtype=wp.transform),  # type: ignore  # world poses of all bodies, from newton.eval_fk
    p: wp.array(dtype=wp.vec3),            # type: ignore  # output: TCP position of every arm, (B,)
    R: wp.array(dtype=wp.mat33),           # type: ignore  # output: TCP orientation (body-to-world rotation) of every arm, (B,)
):
    """Position and orientation of every arm's TCP. Launch with ``dim = B``."""
    w = wp.tid()
    X = body_q[w * NB + HAND]
    p[w] = wp.transform_point(X, TCP_OFFSET)
    R[w] = wp.quat_to_matrix(wp.transform_get_rotation(X))


@wp.kernel
def sphere_centers(
    body_q: wp.array(dtype=wp.transform),   # type: ignore  # world poses of all bodies, from newton.eval_fk
    sphere_body: wp.array(dtype=int),       # type: ignore  # body index of every collision sphere, (NS,)
    sphere_center: wp.array(dtype=wp.vec3), # type: ignore  # center of every collision sphere in its body frame, (NS,)
    x: wp.array2d(dtype=wp.vec3),           # type: ignore  # output: world center of every collision sphere, (B, NS)
):
    """World centers of the collision spheres of every arm. Launch with ``dim = (B, NS)``."""
    w, s = wp.tid()
    x[w, s] = wp.transform_point(body_q[w * NB + sphere_body[s]], sphere_center[s])


@wp.kernel
def build_task_cost(
    # Newton model and state
    joint_q: wp.array(dtype=float),         # type: ignore  # joint positions q of all arms, (B * na,)
    body_q: wp.array(dtype=wp.transform),   # type: ignore  # world poses of all bodies at q
    body_com: wp.array(dtype=wp.vec3),      # type: ignore  # center of mass of every body, in the body frame
    J: wp.array3d(dtype=float),             # type: ignore  # Jacobians of all arms at q, (B, links * 6, na)
    # limits and targets
    q_home: wp.array(dtype=float),          # type: ignore  # home posture for the posture term, (na,)
    q_min: wp.array(dtype=float),           # type: ignore  # lower joint limits, (na,)
    q_max: wp.array(dtype=float),           # type: ignore  # upper joint limits, (na,)
    u_max: wp.array(dtype=float),           # type: ignore  # joint velocity limits, (na,)
    p_des: wp.array(dtype=wp.vec3),         # type: ignore  # desired TCP position p_EE, (B,)
    v_des: wp.array(dtype=wp.vec3),         # type: ignore  # desired TCP velocity v_EE, (B,)
    R_des: wp.array(dtype=wp.mat33),        # type: ignore  # desired TCP orientation R_EE, (B,)
    enabled: wp.array(dtype=int),           # type: ignore  # 1: filter on, 0: filter off (nominal controller), (1,)
    # QP data (float64), written in place
    P: wp.array3d(dtype=wp.float64),        # type: ignore  # output: cost matrices, (B, na, na)
    c: wp.array2d(dtype=wp.float64),        # type: ignore  # output: cost vectors, (B, na)
    x_l: wp.array2d(dtype=wp.float64),      # type: ignore  # output: lower bounds on u, (B, na)
    x_u: wp.array2d(dtype=wp.float64),      # type: ignore  # output: upper bounds on u, (B, na)
):
    """Cost and bounds of every arm's safety filter QP. Launch with ``dim = B``."""
    w = wp.tid()
    X = body_q[w * NB + HAND]
    p = wp.transform_point(X, TCP_OFFSET)

    # orientation error: axis-angle vector of R_des R^T
    E = R_des[w] * wp.transpose(wp.quat_to_matrix(wp.transform_get_rotation(X)))
    e_rot = 0.5 * wp.vec3(E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1])
    s = wp.length(e_rot)
    if s > 1.0e-9:
        e_rot = e_rot * (wp.atan2(s, 0.5 * (E[0, 0] + E[1, 1] + E[2, 2] - 1.0)) / s)

    # desired TCP velocity v_des
    v_lin = v_des[w] + K_POS * (p_des[w] - p)
    v_ang = K_ROT * e_rot

    for i in range(na):
        lin_i, ang_i = point_jacobian_column(J, body_q, body_com, w, HAND, p, i)
        for j in range(na):
            lin_j, ang_j = point_jacobian_column(J, body_q, body_com, w, HAND, p, j)
            p_ij = W_POS * wp.dot(lin_i, lin_j) + W_ROT * wp.dot(ang_i, ang_j)   # J^T W J
            if i == j:
                p_ij += W_POST
            P[w, i, j] = wp.float64(p_ij)
        q = joint_q[w * na + i]
        u_post = K_POST * (q_home[i] - q)
        c[w, i] = wp.float64(-(W_POS * wp.dot(lin_i, v_lin) + W_ROT * wp.dot(ang_i, v_ang)) - W_POST * u_post)

        # velocity limits, and the joint-limit CBFs when the filter is on
        lo = -u_max[i]
        hi = u_max[i]
        if enabled[0] != 0:
            lo = wp.max(lo, ALPHA_JOINT * (q_min[i] - q))
            hi = wp.min(hi, ALPHA_JOINT * (q_max[i] - q))
        x_l[w, i] = wp.float64(lo)
        x_u[w, i] = wp.float64(hi)


@wp.kernel
def build_collision_cbfs(
    # Newton model and state
    body_q: wp.array(dtype=wp.transform),     # type: ignore  # world poses of all bodies at q
    body_com: wp.array(dtype=wp.vec3),        # type: ignore  # center of mass of every body, in the body frame
    J: wp.array3d(dtype=float),               # type: ignore  # Jacobians of all arms at q, (B, links * 6, na)
    # collision model
    sphere_body: wp.array(dtype=int),         # type: ignore  # body index of every collision sphere, (NS,)
    sphere_center: wp.array(dtype=wp.vec3),   # type: ignore  # center of every collision sphere in its body frame, (NS,)
    sphere_radius: wp.array(dtype=float),     # type: ignore  # radius of every collision sphere, (NS,)
    object_center: wp.array2d(dtype=wp.vec3), # type: ignore  # center of every object of every arm, (B, N_o)
    object_radius: wp.array2d(dtype=float),   # type: ignore  # radius of every object of every arm, (B, N_o)
    enabled: wp.array(dtype=int),             # type: ignore  # 1: filter on, 0: filter off (all rows inactive), (1,)
    # QP data (float64), written in place
    G: wp.array3d(dtype=wp.float64),          # type: ignore  # output: CBF rows, (B, NS * (N_o + 1), na)
    h_u: wp.array2d(dtype=wp.float64),        # type: ignore  # output: CBF right-hand sides alpha h, (B, NS * (N_o + 1))
):
    """CBF rows of collision sphere ``s`` of arm ``w``, against every object and the table.
    Launch with ``dim = (B, NS)``.
    """
    w, s = wp.tid()
    k = sphere_body[s]
    x = wp.transform_point(body_q[w * NB + k], sphere_center[s])
    n_objects = object_center.shape[1]
    row0 = s * (n_objects + 1)
    for o in range(n_objects + 1):
        # the table (o == n_objects): normal e_z, distance to the plane z = 0
        n = wp.vec3(0.0, 0.0, 1.0)
        h = x[2] - sphere_radius[s] - MARGIN
        if o < n_objects:
            d = x - object_center[w, o]
            dist = wp.length(d)
            n = d / dist
            h = dist - sphere_radius[s] - object_radius[w, o] - MARGIN
        # n^T J_s u >= -alpha h  <=>  -n^T J_s u <= alpha h
        for j in range(na):
            lin, ang = point_jacobian_column(J, body_q, body_com, w, k, x, j)
            G[w, row0 + o, j] = wp.float64(-wp.dot(n, lin))
        if enabled[0] != 0:
            h_u[w, row0 + o] = wp.float64(ALPHA * h)
        else:
            h_u[w, row0 + o] = wp.float64(wp.inf)


# ---------------------------------------------------------------------------
# Kernels: reference, integration and logging
# ---------------------------------------------------------------------------

@wp.func
def clearance(
    body_q: wp.array(dtype=wp.transform),     # type: ignore  # world poses of all bodies
    sphere_body: wp.array(dtype=int),         # type: ignore  # body index of every collision sphere, (NS,)
    sphere_center: wp.array(dtype=wp.vec3),   # type: ignore  # center of every collision sphere in its body frame, (NS,)
    sphere_radius: wp.array(dtype=float),     # type: ignore  # radius of every collision sphere, (NS,)
    object_center: wp.array2d(dtype=wp.vec3), # type: ignore  # center of every object of every arm, (B, N_o)
    object_radius: wp.array2d(dtype=float),   # type: ignore  # radius of every object of every arm, (B, N_o)
    w: int,                                   # type: ignore  # arm (world) index
):
    """Smallest distance between arm ``w``'s collision spheres and its objects or the table."""
    d_min = float(1.0e6)
    for s in range(sphere_body.shape[0]):
        x = wp.transform_point(body_q[w * NB + sphere_body[s]], sphere_center[s])
        d_min = wp.min(d_min, x[2] - sphere_radius[s])
        for o in range(object_center.shape[1]):
            d_min = wp.min(d_min, wp.length(x - object_center[w, o]) - sphere_radius[s] - object_radius[w, o])
    return d_min


@wp.kernel
def tcp_reference(
    tick: wp.array(dtype=int),           # type: ignore  # current control tick, (1,)
    dt: float,                           # type: ignore  # control period [s]
    t_ramp: float,                       # type: ignore  # duration of the blend from home into the sweep [s]
    p_home: wp.vec3,                     # type: ignore  # TCP position at home, where every arm starts
    center: wp.array(dtype=wp.vec3),     # type: ignore  # center of each sweep, (B,)
    half_length: wp.array(dtype=float),  # type: ignore  # half length of each sweep [m], (B,)
    omega: wp.array(dtype=float),        # type: ignore  # angular frequency of each sweep [rad/s], (B,)
    direction: wp.array(dtype=wp.vec3),  # type: ignore  # unit direction of each sweep, (B,)
    p_des: wp.array(dtype=wp.vec3),      # type: ignore  # output: desired TCP position p_EE, (B,)
    v_des: wp.array(dtype=wp.vec3),      # type: ignore  # output: desired TCP velocity v_EE, (B,)
):
    """Desired TCP position and velocity of every arm at the current tick: a back-and-forth
    sweep, blended in from home. Launch with ``dim = B``.
    """
    w = wp.tid()
    t = float(tick[0]) * dt
    u = wp.min(t / t_ramp, 1.0)
    s = u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)        # smooth 0 -> 1 blend and its derivative
    ds = 30.0 * u * u * (1.0 - u) * (1.0 - u) / t_ramp
    shape = center[w] - p_home + half_length[w] * wp.sin(omega[w] * t) * direction[w]
    dshape = half_length[w] * omega[w] * wp.cos(omega[w] * t) * direction[w]
    p_des[w] = p_home + s * shape
    v_des[w] = ds * shape + s * dshape


@wp.kernel
def apply_and_log(
    u: wp.array2d(dtype=wp.float64),          # type: ignore  # QP solutions: joint velocity commands, (B, na)
    status: wp.array(dtype=wp.int32),         # type: ignore  # solver status of every QP, (B,)
    solved: int,                              # type: ignore  # status value of a solved QP
    tick: wp.array(dtype=int),                # type: ignore  # current control tick, (1,)
    stride: int,                              # type: ignore  # control ticks per logged frame
    body_q: wp.array(dtype=wp.transform),     # type: ignore  # world poses of all bodies at the current q
    sphere_body: wp.array(dtype=int),         # type: ignore  # body index of every collision sphere, (NS,)
    sphere_center: wp.array(dtype=wp.vec3),   # type: ignore  # center of every collision sphere in its body frame, (NS,)
    sphere_radius: wp.array(dtype=float),     # type: ignore  # radius of every collision sphere, (NS,)
    object_center: wp.array2d(dtype=wp.vec3), # type: ignore  # center of every object of every arm, (B, N_o)
    object_radius: wp.array2d(dtype=float),   # type: ignore  # radius of every object of every arm, (B, N_o)
    p_des: wp.array(dtype=wp.vec3),           # type: ignore  # desired TCP position of every arm, (B,)
    joint_q: wp.array(dtype=float),           # type: ignore  # in/out: joint positions of all arms, (B * na,)
    joint_qd: wp.array(dtype=float),          # type: ignore  # output: joint velocities of all arms, (B * na,)
    n_unsolved: wp.array(dtype=int),          # type: ignore  # in/out: number of unsolved arm-ticks, (1,)
    min_clearance: wp.array(dtype=float),     # type: ignore  # in/out: smallest clearance of every arm so far [m], (B,)
    log_clearance: wp.array2d(dtype=float),   # type: ignore  # output: clearance per frame [m], (N_FRAMES, B)
    log_err: wp.array2d(dtype=float),         # type: ignore  # output: TCP tracking error per frame [m], (N_FRAMES, B)
    log_q: wp.array2d(dtype=float),           # type: ignore  # output: joint positions per frame, (N_FRAMES, B * na)
    log_p: wp.array2d(dtype=wp.vec3),         # type: ignore  # output: TCP position per frame, (N_FRAMES, B)
    log_p_des: wp.array2d(dtype=wp.vec3),     # type: ignore  # output: desired TCP position per frame, (N_FRAMES, B)
):
    """Log every arm, then integrate its commanded joint velocities over one tick (an arm whose
    QP was not solved stops). Launch with ``dim = B``.
    """
    w = wp.tid()
    i = tick[0]
    d = clearance(body_q, sphere_body, sphere_center, sphere_radius, object_center, object_radius, w)
    min_clearance[w] = wp.min(min_clearance[w], d)
    if i % stride == 0:                                    # full state at the video frame rate
        f = i // stride
        p = wp.transform_point(body_q[w * NB + HAND], TCP_OFFSET)
        log_clearance[f, w] = d
        log_err[f, w] = wp.length(p - p_des[w])
        for j in range(na):
            log_q[f, w * na + j] = joint_q[w * na + j]
        log_p[f, w] = p
        log_p_des[f, w] = p_des[w]

    ok = status[w] == solved
    if not ok:
        wp.atomic_add(n_unsolved, 0, 1)
    for j in range(na):
        qd = float(0.0)
        if ok:
            qd = float(u[w, j])
        joint_qd[w * na + j] = qd
        joint_q[w * na + j] = joint_q[w * na + j] + DT * qd


@wp.kernel
def advance(
    tick: wp.array(dtype=int),  # type: ignore  # in/out: control tick counter, (1,)
):
    """Increment the control tick counter. Launch with ``dim = 1``."""
    tick[0] = tick[0] + 1


# ---------------------------------------------------------------------------
# The scene of every arm
# ---------------------------------------------------------------------------

def sample_scene(rng, B, n_objects, x_home, clearance=0.05):
    """Sample a sweep and a table of objects for every arm.

    The sweep is a back-and-forth line, 50 to 70 cm long, 10 to 20 cm above
    the table. The objects rest on the table, with a radius of 4 to 10 cm,
    scattered around the sweep. An object is rejected if it would come closer
    than ``clearance`` to the arm at home, so every arm starts safe.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    B : int
        Number of arms.
    n_objects : int
        Number of objects per arm.
    x_home : np.ndarray
        World centers of the collision spheres at the home posture, shape
        ``(NS, 3)``.
    clearance : float
        Minimum distance [m] between an object and the arm at home.

    Returns
    -------
    dict of np.ndarray
        ``"center"``, ``"direction"``, ``"half_length"``, ``"omega"``: center
        ``(B, 3)``, unit direction ``(B, 3)``, half length ``(B,)`` and angular
        frequency ``(B,)`` of every arm's sweep; ``"object_center"``
        ``(B, n_objects, 3)`` and ``"object_radius"`` ``(B, n_objects)``.
    """
    center = np.stack([rng.uniform(0.45, 0.6, B), rng.uniform(-0.05, 0.05, B), rng.uniform(0.1, 0.2, B)], axis=1)
    heading = rng.uniform(-0.3, 0.3, B)
    direction = np.stack([np.sin(heading), np.cos(heading), np.zeros(B)], axis=1)
    side = np.stack([np.cos(heading), -np.sin(heading), np.zeros(B)], axis=1)
    half_length = rng.uniform(0.25, 0.35, B)
    omega = 2 * np.pi / rng.uniform(3.0, 4.5, B)

    radius = SPHERE_RADIUS.numpy()
    object_center = np.zeros((B, n_objects, 3))
    object_radius = np.zeros((B, n_objects))
    for o in range(n_objects):
        todo = np.arange(B)                       # arms whose object o is still to be placed
        while len(todo) > 0:
            k = len(todo)
            r = rng.uniform(0.04, 0.10, k)
            along = rng.uniform(-1.0, 1.0, k) * half_length[todo]
            across = rng.uniform(-0.15, 0.15, k)
            c = center[todo] + along[:, None] * direction[todo] + across[:, None] * side[todo]
            c[:, 2] = r                           # resting on the table
            gap = np.linalg.norm(x_home[None] - c[:, None], axis=2) - radius[None] - r[:, None]
            ok = gap.min(axis=1) > clearance
            object_center[todo[ok], o] = c[ok]
            object_radius[todo[ok], o] = r[ok]
            todo = todo[~ok]
    return dict(center=center, direction=direction, half_length=half_length, omega=omega,
                object_center=object_center, object_radius=object_radius)


# ---------------------------------------------------------------------------
# The arms and their safety filter QPs
# ---------------------------------------------------------------------------

class PandaArms:
    """``B`` velocity-controlled Panda arms, each sweeping its gripper across its own table.

    The class owns everything on the robot side of the closed loop: the Newton
    model and state, the sweep and the objects of every arm, and the safety
    filter QP of every arm as float64 Warp arrays. A control tick is::

        arms.assemble_qps()             # P, c, G, h_u, x_l, x_u at the current state
        ...                             # solve the B QPs: u = joint velocities
        arms.apply(u, status, solved)   # log, integrate u over DT, advance the clock

    Both methods only launch GPU work on the current Warp stream and never
    touch the host, so a tick can be recorded into a CUDA graph. The clock is
    a tick counter on the GPU.

    Parameters
    ----------
    B : int
        Number of arms.
    n_objects : int
        Number of objects on the table of every arm.
    t_sim : float
        Duration of a run [s].
    t_ramp : float
        Duration of the blend from home into the sweep [s].
    fps : int
        Logging (and video) frame rate.
    seed : int
        Seed of the random scene.

    Attributes
    ----------
    P, c, G, h_u, x_l, x_u : wp.array(dtype=wp.float64)
        QP data of every arm, shapes ``(B, n, n)``, ``(B, n)``, ``(B, m, n)``,
        ``(B, m)``, ``(B, n)`` and ``(B, n)``, written by ``assemble_qps``.
    n, m : int
        Number of variables (joints) and of CBF constraints of every QP.
    n_steps, n_frames, stride : int
        Control ticks per run, logged frames per run, and ticks per frame.
    n_unsolved : wp.array(dtype=int)
        Number of arm-ticks whose QP was not solved since ``reset``, shape ``(1,)``.
    """

    def __init__(self, B, n_objects, t_sim, t_ramp=1.0, fps=25, seed=0):
        self.B = B
        self.n = na
        self.m = NS * (n_objects + 1)
        self.t_ramp = t_ramp
        self.fps = fps
        self.n_steps = int(round(t_sim / DT))
        self.stride = int(round(1 / (fps * DT)))
        self.n_frames = (self.n_steps + self.stride - 1) // self.stride
        self.model = build_model(B)
        self.state = self.model.state()

        # TCP pose and collision spheres at home, and the scene of every arm.
        home = self.model.state()
        home.joint_q.assign(np.tile(Q_HOME, B))
        newton.eval_fk(self.model, home.joint_q, home.joint_qd, home)
        p_home, self._R_des = wp.zeros(B, dtype=wp.vec3), wp.zeros(B, dtype=wp.mat33)
        wp.launch(tcp_pose, dim=B, inputs=[home.body_q], outputs=[p_home, self._R_des])
        x_home = wp.zeros((B, NS), dtype=wp.vec3)
        wp.launch(sphere_centers, dim=(B, NS), inputs=[home.body_q, SPHERE_BODY, SPHERE_CENTER], outputs=[x_home])
        self._p_home = wp.vec3(*p_home.numpy()[0])
        self.scene = sample_scene(np.random.default_rng(seed), B, n_objects, x_home.numpy()[0])
        self._sweep = [wp.array(self.scene["center"], dtype=wp.vec3), wp.array(self.scene["half_length"], dtype=float),
                       wp.array(self.scene["omega"], dtype=float), wp.array(self.scene["direction"], dtype=wp.vec3)]
        self._object_center = wp.array(self.scene["object_center"], dtype=wp.vec3)
        self._object_radius = wp.array(self.scene["object_radius"], dtype=float)

        # Desired TCP motion, filter flag, and the QP data.
        self._p_des = wp.zeros(B, dtype=wp.vec3)
        self._v_des = wp.zeros(B, dtype=wp.vec3)
        self._enabled = wp.ones(1, dtype=int)
        self.P = wp.zeros((B, na, na), dtype=wp.float64)
        self.c = wp.zeros((B, na), dtype=wp.float64)
        self.G = wp.zeros((B, self.m, na), dtype=wp.float64)
        self.h_u = wp.zeros((B, self.m), dtype=wp.float64)
        self.x_l = wp.zeros((B, na), dtype=wp.float64)
        self.x_u = wp.zeros((B, na), dtype=wp.float64)

        # Limits, Newton's Jacobians, clock and logs.
        self._q_home = wp.array(Q_HOME, dtype=float)
        self._q_min = wp.array(Q_MIN, dtype=float)
        self._q_max = wp.array(Q_MAX, dtype=float)
        self._u_max = wp.array(U_MAX, dtype=float)
        self._J = wp.zeros((B, self.model.max_joints_per_articulation * 6, na), dtype=float)
        self._tick = wp.zeros(1, dtype=int)
        self.n_unsolved = wp.zeros(1, dtype=int)
        self._log = dict(min_clearance=wp.zeros(B, dtype=float),
                         clearance=wp.zeros((self.n_frames, B), dtype=float),
                         err=wp.zeros((self.n_frames, B), dtype=float),
                         q=wp.zeros((self.n_frames, B * na), dtype=float),
                         p=wp.zeros((self.n_frames, B), dtype=wp.vec3),
                         p_des=wp.zeros((self.n_frames, B), dtype=wp.vec3))
        self.reset()

    @property
    def qp(self):
        """The QP data of every arm as keyword arguments of ``setup`` / ``update``."""
        return dict(P=self.P, c=self.c, G=self.G, h_u=self.h_u, x_l=self.x_l, x_u=self.x_u)

    def reset(self):
        """Put every arm back at home, at rest, and restart the clock and the logs."""
        self.state.joint_q.assign(np.tile(Q_HOME, self.B))
        self.state.joint_qd.zero_()
        self._tick.zero_()
        self.n_unsolved.zero_()
        self._log["min_clearance"].fill_(np.inf)

    def set_filter(self, enabled):
        """Turn the CBFs on (``True``) or off (``False``).

        With the filter off, ``h_u = +inf`` and the bounds are the velocity
        limits only, so the QP returns the nominal controller. The flag lives
        on the GPU: a captured CUDA graph sees the new value.
        """
        self._enabled.fill_(1 if enabled else 0)

    def assemble_qps(self):
        """Write the QP of every arm at the current tick and state.

        Computes the desired TCP motion, runs Newton's forward kinematics and
        Jacobians, and writes ``P``, ``c``, ``G``, ``h_u``, ``x_l`` and ``x_u``.
        """
        m, B = self.model, self.B
        wp.launch(tcp_reference, dim=B,
                  inputs=[self._tick, DT, self.t_ramp, self._p_home, *self._sweep],
                  outputs=[self._p_des, self._v_des])
        newton.eval_fk(m, self.state.joint_q, self.state.joint_qd, self.state)
        newton.eval_jacobian(m, self.state, self._J)
        wp.launch(build_task_cost, dim=B,
                  inputs=[self.state.joint_q, self.state.body_q, m.body_com, self._J, self._q_home, self._q_min,
                          self._q_max, self._u_max, self._p_des, self._v_des, self._R_des, self._enabled],
                  outputs=[self.P, self.c, self.x_l, self.x_u])
        wp.launch(build_collision_cbfs, dim=(B, NS),
                  inputs=[self.state.body_q, m.body_com, self._J, SPHERE_BODY, SPHERE_CENTER, SPHERE_RADIUS,
                          self._object_center, self._object_radius, self._enabled],
                  outputs=[self.G, self.h_u])

    def apply(self, u, status, solved):
        """Log every arm, integrate the commands ``u`` over one tick, and advance the clock.

        Parameters
        ----------
        u : wp.array2d(dtype=wp.float64)
            Joint velocity command of every arm, shape ``(B, n)``.
        status : wp.array(dtype=wp.int32)
            Solver status of every QP, shape ``(B,)``. An arm whose QP was
            not solved stops for this tick, which is safe among static objects.
        solved : int
            Status value of a solved QP.
        """
        log = self._log
        wp.launch(apply_and_log, dim=self.B,
                  inputs=[u, status, solved, self._tick, self.stride, self.state.body_q, SPHERE_BODY,
                          SPHERE_CENTER, SPHERE_RADIUS, self._object_center, self._object_radius, self._p_des],
                  outputs=[self.state.joint_q, self.state.joint_qd, self.n_unsolved, log["min_clearance"],
                           log["clearance"], log["err"], log["q"], log["p"], log["p_des"]])
        wp.launch(advance, dim=1, inputs=[self._tick])

    def logs(self):
        """Host copy of the logs since ``reset``.

        Returns
        -------
        dict of np.ndarray
            ``"min_clearance"``: smallest clearance of every arm [m], ``(B,)``;
            per frame: ``"clearance"`` and ``"err"`` (TCP tracking error) [m],
            ``(n_frames, B)``; ``"q"``, joint positions, ``(n_frames, B * n)``;
            ``"p"`` and ``"p_des"``, actual and desired TCP position,
            ``(n_frames, B, 3)``.
        """
        return {key: value.numpy() for key, value in self._log.items()}


# ---------------------------------------------------------------------------
# Plots and video
# ---------------------------------------------------------------------------

LABELS = dict(nominal="nominal controller", filtered="CBF safety filter")
COLORS = dict(nominal="C3", filtered="C0")


def plot_runs(arms, runs):
    """Histogram of the smallest clearance of every arm, and the TCP tracking error over time.

    Parameters
    ----------
    arms : PandaArms
        The arms of the runs.
    runs : dict
        ``{"nominal": logs, "filtered": logs}``, each from ``arms.logs()``.
    """
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.2))
    bins = np.linspace(-0.12, 0.08, 81) * 1e3
    for name, log in runs.items():
        ax0.hist(log["min_clearance"] * 1e3, bins=bins, color=COLORS[name], alpha=0.6, label=LABELS[name])
    ax0.axvline(0, color="k", lw=1)
    ax0.text(-2, ax0.get_ylim()[1] * 0.9, "collision", ha="right", va="top")
    ax0.set_xlabel("smallest clearance over the run [mm]"); ax0.set_ylabel("arms")
    ax0.set_title(f"Clearance of {arms.B} arms"); ax0.legend(loc="upper right"); ax0.grid(True, alpha=0.3)

    t = np.arange(arms.n_frames) * arms.stride * DT
    for name, log in runs.items():
        err = log["err"] * 1e3
        ax1.semilogy(t, np.median(err, axis=1), color=COLORS[name], label=f"{LABELS[name]}, median")
        ax1.fill_between(t, np.quantile(err, 0.05, axis=1), np.quantile(err, 0.95, axis=1),
                         color=COLORS[name], alpha=0.2)
    ax1.axvline(arms.t_ramp, color="0.5", ls=":", lw=1)
    ax1.set_ylim(bottom=0.05)
    ax1.set_xlabel("time [s]"); ax1.set_ylabel("TCP tracking error [mm]")
    ax1.set_title("Tracking error (median and 5-95 %)"); ax1.legend(loc="upper left"); ax1.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    plt.show()


def render_video(arms, runs, worlds, camera, width=960, height=720, trail_seconds=1.5):
    """Render both logged runs of some arms with Newton's OpenGL viewer, side by side.

    The viewer renders offscreen through EGL, so this also works without a
    display. Objects turn red while the arm is inside them; the white line is
    the desired path and the colored line the recent path of the TCP.

    Parameters
    ----------
    arms : PandaArms
        The arms of the runs.
    runs : dict
        ``{"nominal": logs, "filtered": logs}``, each from ``arms.logs()``.
    worlds : sequence of int
        Arms (worlds) to show. They are laid out on a grid, 1.1 m apart along x
        and 1.3 m along y.
    camera : tuple
        Camera position (``wp.vec3``), pitch and yaw [deg], as taken by
        ``ViewerGL.set_camera``.
    width, height : int
        Size of each half of the image [pixels].
    trail_seconds : float
        Length of the TCP trails [s].

    Returns
    -------
    list of np.ndarray
        One RGB image of shape ``(height, 2 * width, 3)`` per logged frame:
        the nominal run on the left, the filtered run on the right.
    """
    import pyglet
    pyglet.options["headless"] = True        # render offscreen through EGL, without a display

    worlds = np.asarray(worlds)
    n = len(worlds)
    model = build_model(n)
    viewer = newton.viewer.ViewerGL(width=width, height=height, headless=True)
    viewer.set_model(model)
    viewer.set_world_offsets((1.1, 1.3, 0.0))
    viewer.set_camera(*camera)
    offsets = viewer.world_offsets.numpy()                                  # (n, 3)
    trail_colors = matplotlib.colormaps["turbo"](np.linspace(0.08, 0.92, n))[:, :3]
    trail = int(trail_seconds * arms.fps)
    replay = model.state()
    x = wp.zeros((n, NS), dtype=wp.vec3)
    vec3 = lambda a: wp.array(np.ascontiguousarray(a, dtype=np.float32).reshape(-1, 3), dtype=wp.vec3)

    objects = arms.scene["object_center"][worlds] + offsets[:, None]       # (n, N_o, 3)
    object_radius = arms.scene["object_radius"][worlds]
    radii = wp.array(object_radius.ravel(), dtype=wp.float32)
    font = ImageFont.load_default(size=30)

    frames = []
    for f in range(arms.n_frames):
        halves = []
        for name in ("nominal", "filtered"):
            log = runs[name]
            replay.joint_q.assign(log["q"][f].reshape(arms.B, na)[worlds].ravel())
            newton.eval_fk(model, replay.joint_q, replay.joint_qd, replay)
            wp.launch(sphere_centers, dim=(n, NS), inputs=[replay.body_q, SPHERE_BODY, SPHERE_CENTER], outputs=[x])
            spheres = x.numpy() + offsets[:, None]                         # (n, NS, 3)
            gap = (np.linalg.norm(spheres[:, :, None] - objects[:, None], axis=3)
                   - SPHERE_RADIUS.numpy()[None, :, None] - object_radius[:, None])
            hit = (gap < 0).any(axis=1)                                    # (n, N_o)
            object_colors = np.where(hit[..., None], [0.9, 0.1, 0.1], [0.85, 0.65, 0.35])

            viewer.begin_frame(f * arms.stride * DT)
            viewer.log_state(replay)
            viewer.log_points("objects", vec3(objects), radii, vec3(object_colors))
            idx = np.arange(max(0, f - trail), f + 1)
            if len(idx) > 1:
                for line, data, colors, line_width in (("desired", log["p_des"], np.ones_like(trail_colors), 0.003),
                                                       ("trails", log["p"], trail_colors, 0.008)):
                    path = data[idx][:, worlds] + offsets                  # (T, n, 3)
                    seg_colors = np.repeat(colors[None], len(idx) - 1, axis=0)
                    viewer.log_lines(line, vec3(path[:-1]), vec3(path[1:]), vec3(seg_colors), line_width)
            viewer.end_frame()
            image = Image.fromarray(viewer.get_frame().numpy())
            ImageDraw.Draw(image).text((20, 15), LABELS[name], fill=(255, 255, 255), font=font)
            halves.append(np.asarray(image))
        frames.append(np.concatenate(halves, axis=1))
    viewer.close()
    return frames
