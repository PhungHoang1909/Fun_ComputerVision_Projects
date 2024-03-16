#  Perspective 3D cube to image plane

import numpy as np
import matplotlib.pyplot as plt

vec = np.array

A = vec([1, 1, 1])
B = vec([-1, 1, 1])
C = vec([1, -1, 1])
D = vec([-1, -1, 1])
E = vec([1, 1, -1])
F = vec([-1, 1, -1])
G = vec([1, -1, -1])
H = vec([-1, -1, -1])

camera = vec([2, 3, 5])

Points = dict(zip("ABCDEFGH", [A, B, C, D, E, F, G, H]))

edges = ["AB", "CD", "EF", "GH", "AC", "BD", "EG", "FH", "AE", "CG", "BF", "DH"]

points = {k: p - camera for k, p in Points.items()}

def pinhole(v):
    x, y, z = v
    return vec([x / z, y / z])

uvs = {k: pinhole(p) for k, p in points.items()}

plt.figure(figsize=(10, 10))
for a, b in edges:
    ua, va = uvs[a]
    ub, vb = uvs[b]
    plt.plot([ua, ub], [va, vb], "ko-")

plt.title('Pinhole Camera Projection of Cube Vertices')
plt.xlabel('u-axis')
plt.ylabel('v-axis')
plt.axis("equal")
plt.grid()
plt.show()
