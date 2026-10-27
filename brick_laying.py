import time

import cv2
import numpy as np
import pybullet as p
import pybullet_data

rng = np.random.default_rng(0)

# arm, same as scara.urdf
L1, L2 = 0.4, 0.35  # link lengths, m
H = 0.6  # shoulder height
LIM = 2.9  # shoulder and elbow limit, rad
VMAX = np.array([2, 2, 0.5, 3])  # joint speed limits
FINGERS = [4, 5]
GRIP = 50  # N per finger. a 2.8 kg brick needs 17 N to hang

# IS 1077 brick, 90 x 90 mm section. half bat at the ends of the middle course
LENGTH = {"full": 0.19, "half": 0.09}
B = 0.09
MASS = {"full": 2.8, "half": 1.3}  # kg, burnt clay
JOINT = 0.01  # mortar gap
COURSES = [["full"] * 3, ["half", "full", "full", "half"], ["full"] * 3]
WALL_Y = 0.42
TRAY = np.array([0.35, -0.35])
CARRY_Z = 0.42  # clears the wall

p.connect(p.GUI)
p.resetDebugVisualizerCamera(1.5, 25, -40, [0.15, 0.05, 0.1])
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")
arm = p.loadURDF("scara.urdf", useFixedBase=True)

# friction grip. anchors stop contacts creeping, spinning friction stops the brick turning between the fingers
for f in FINGERS:
    p.changeDynamics(arm, f, lateralFriction=1.0, spinningFriction=0.02, frictionAnchor=1)
# fingers geared together like a real parallel gripper
gear = p.createConstraint(arm, FINGERS[0], arm, FINGERS[1], p.JOINT_GEAR, [1, 0, 0], [0, 0, 0], [0, 0, 0])
p.changeConstraint(gear, gearRatio=-1, erp=0.5, maxForce=1000)


def wrap(a, period=2 * np.pi):
    return (a + period / 2) % period - period / 2


q = np.zeros(4)


def ik(x, y, z, yaw):
    # law of cosines gives the elbow, two answers - elbow left or right
    c2 = (x * x + y * y - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    options = []
    for q2 in (np.arccos(c2), -np.arccos(c2)):
        q1 = wrap(np.arctan2(y, x) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2)))
        if abs(q1) < LIM and abs(q2) < LIM:
            options.append(np.array([q1, q2, H - z, wrap(yaw - q1 - q2)]))
    return min(options, key=lambda o: np.abs(o[:2] - q[:2]).sum())  # least travel


def step(n=1):
    for _ in range(n):
        p.stepSimulation()
        time.sleep(1 / 240)


def move(target):
    # eased start and stop, the slowest joint at its speed limit sets the time
    global q
    secs = max(0.5, np.max(1.5 * np.abs(target - q) / VMAX))
    for s in np.linspace(0, 1, int(secs * 240)):
        s = s * s * (3 - 2 * s)
        p.setJointMotorControlArray(arm, range(4), p.POSITION_CONTROL, targetPositions=q + (target - q) * s, forces=[500] * 4)
        step()
    step(30)
    q = target


def fingers(closed):
    # aims past the brick, so the motor presses with GRIP
    p.setJointMotorControlArray(arm, FINGERS, p.POSITION_CONTROL, targetPositions=[0.03 * closed] * 2, forces=[GRIP] * 2)
    step(120)


def feed(kind):
    # next brick on the tray, a bit off centre and turned
    half = [LENGTH[kind] / 2, B / 2, B / 2]
    xy = TRAY + rng.uniform(-0.03, 0.03, 2)
    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    look = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.7, 0.3, 0.2, 1])
    b = p.createMultiBody(MASS[kind], shape, look, [xy[0], xy[1], half[2]], p.getQuaternionFromEuler([0, 0, rng.uniform(-0.35, 0.35)]))
    p.changeDynamics(b, -1, lateralFriction=0.8, frictionAnchor=1)
    step(60)
    return b


# depth camera 1 m above the tray, looking down
W = 480
view = p.computeViewMatrix([TRAY[0], TRAY[1], 1.0], [TRAY[0], TRAY[1], 0], [0, 1, 0])
proj = p.computeProjectionMatrixFOV(40, 1, 0.1, 2.5)
to_world = np.linalg.inv(np.array(proj).reshape(4, 4, order="F") @ np.array(view).reshape(4, 4, order="F"))
u, v = np.meshgrid(np.arange(W), np.arange(W))


def see():
    depth = np.array(p.getCameraImage(W, W, view, proj, renderer=p.ER_TINY_RENDERER)[3]).reshape(W, W)
    # pixel + depth to a 3D point. tiny renderer samples pixel corners, rows counted from the bottom
    ndc = np.stack([2 * u / W - 1, 2 * (W - 1 - v) / W - 1, 2 * depth - 1, np.ones_like(depth)], -1)
    pts = ndc @ to_world.T
    pts = pts[..., :3] / pts[..., 3:]
    pts[..., 2] += rng.normal(0, 0.002, (W, W))  # 2 mm depth noise
    # top face of whatever is on the tray. the sides would drag the centre outward
    near = np.hypot(pts[..., 0] - TRAY[0], pts[..., 1] - TRAY[1]) < 0.2
    z = pts[..., 2]
    top = pts[near & (z > np.percentile(z[near & (z > 0.03)], 90) - 0.01)]
    (cx, cy), (a, b), ang = cv2.minAreaRect(top[:, :2].astype(np.float32))
    yaw = np.radians(ang) + (0 if a >= b else np.pi / 2)  # long side
    return cx, cy, np.median(top[:, 2]), wrap(yaw, np.pi)


# wall plan: brick centres in laying order, 10 mm joints
plan = []
for c, course in enumerate(COURSES):
    x = 0.0
    for kind in course:
        plan.append((kind, x + LENGTH[kind] / 2, WALL_Y, B * (c + 0.5)))
        x += LENGTH[kind] + JOINT

move(ik(0.3, 0.6, CARRY_Z, 0))  # over the wall, out of the camera's view
for n, (kind, tx, ty, tz) in enumerate(plan, 1):
    brick = feed(kind)
    x, y, top, yaw = see()

    move(ik(x, y, top + 0.1, yaw))
    move(ik(x, y, top, yaw))
    fingers(True)
    move(ik(x, y, CARRY_Z, yaw))

    move(ik(tx, ty, CARRY_Z, 0))
    move(ik(tx, ty, tz + B / 2 + 0.003, 0))  # 3 mm above its spot
    fingers(False)
    move(ik(tx, ty, CARRY_Z, 0))

    off = 1000 * np.linalg.norm(np.array(p.getBasePositionAndOrientation(brick)[0]) - [tx, ty, tz])
    print(f"brick {n:2d} {kind}: {off:.1f} mm from its spot")

while p.isConnected():  # keep the window open
    time.sleep(0.1)
