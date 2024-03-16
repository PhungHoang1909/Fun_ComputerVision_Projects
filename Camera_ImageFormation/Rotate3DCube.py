#     Rotate 3D cube by the axis (X, Y, Z), project rotated object to image plane  

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define points
A = np.array([1, 1, 1])
B = np.array([-1, 1, 1])
C = np.array([1, -1, 1])
D = np.array([-1, -1, 1])
E = np.array([1, 1, -1])
F = np.array([-1, 1, -1])
G = np.array([1, -1, -1])
H = np.array([-1, -1, -1])

camera = np.array([2, 3, 5])

Points = dict(zip("ABCDEFGH", [A, B, C, D, E, F, G, H]))

edges = ["AB", "CD", "EF", "GH", "AC", "BD", "EG", "FH", "AE", "CG", "BF", "DH"]
points = {k: v - camera for k, v in Points.items()}


def pinhole(v):
    x, y, z = v
    if z == 0:
        return np.array([float('inf'), float('inf')])
    return np.array([x / z, y / z])


def rotate(R, v):
    return np.dot(R, v)


angles = [10, 20, 30]

# Plot X 
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for i, angle_x in enumerate(angles):
    Rx = cv2.Rodrigues(np.array([np.radians(angle_x), 0, 0]))[0]

    # Apply rotation
    ps = {key: rotate(Rx, value) for key, value in points.items()}
    uvs = {key: pinhole(value) for key, value in ps.items()}

    for a, b in edges:
        ua, va = uvs[a]
        ub, vb = uvs[b]
        ax[i].plot([ua, ub], [va, vb], "ko-")

    ax[i].set_title(f"X{angle_x}")
    ax[i].axis("equal")
    ax[i].grid()

plt.show()

# Plot Y 
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for i, angle_y in enumerate(angles):
    Ry = cv2.Rodrigues(np.array([0, np.radians(angle_y), 0]))[0]

    # Apply rotation
    ps = {key: rotate(Ry, value) for key, value in points.items()}
    uvs = {key: pinhole(value) for key, value in ps.items()}

    for a, b in edges:
        ua, va = uvs[a]
        ub, vb = uvs[b]
        ax[i].plot([ua, ub], [va, vb], "ko-")

    ax[i].set_title(f"Y{angle_y}")
    ax[i].axis("equal")
    ax[i].grid()

plt.show()

# Plot Z
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for i, angle_z in enumerate(angles):
    Rz = cv2.Rodrigues(np.array([0, 0, np.radians(angle_z)]))[0]

    # Apply rotation
    ps = {key: rotate(Rz, value) for key, value in points.items()}
    uvs = {key: pinhole(value) for key, value in ps.items()}

    for a, b in edges:
        ua, va = uvs[a]
        ub, vb = uvs[b]
        ax[i].plot([ua, ub], [va, vb], "ko-")

    ax[i].set_title(f"Z{angle_z}")
    ax[i].axis("equal")
    ax[i].grid()

plt.show()