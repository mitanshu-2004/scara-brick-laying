# Camera-guided brick laying with a SCARA arm

A SCARA arm in PyBullet builds a small brick wall. Bricks come one at a time on a tray, a depth camera finds each one, a two-finger gripper picks it up by friction, and the arm lays it in running bond.

## Run

```bash
python brick_laying.py
```

## How it works

1. A brick arrives, a bit off centre and turned.
2. A depth camera above the tray turns every pixel into a 3D point. A rectangle fitted to the brick's top face gives its centre and angle.
3. IK works out the joint angles. The law of cosines gives the elbow, with two answers (elbow left or right), and it keeps the one inside the joint limits.
4. The gripper closes on the brick at 50 N per finger and lifts it. Only friction holds it. A 2.8 kg brick needs about 17 N per finger.
5. The arm carries it over the wall, lowers it to 3 mm above its spot and lets go.

Bricks are IS 1077 size, 190 x 90 x 90 mm. The middle line starts and ends with a half brick so its joints sit over the middle of the bricks below.

## Limits

Simulation only. The camera's position is known exactly, every brick is the same size, and there's no mortar.

