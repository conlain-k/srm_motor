import numpy as np
import numba
from matplotlib import pyplot as plt

incr = 1

dt = 0.01

vel = 1

r2 = np.sqrt(2)

@numba.stencil
def propogate(func):
    delta = 0
    delta += func[1, 0]
    delta += func[-1, 0]
    delta += func[0, 1]
    delta += func[0, -1]

    delta += func[1, 1] / r2
    delta += func[-1, 1] / r2
    delta += func[1, -1] / r2
    delta += func[-1, -1] / r2

    return delta


def step(func):
    print("step")
    delta = propogate(func)
    func += vel * dt * delta
    func = np.minimum(func, np.ones(func.shape))
    return func


N = 20
pts = np.linspace(0, 1, N + 1)
X, Y = np.meshgrid(pts, pts)

Z = np.zeros(X.shape)
Z[N // 2, N // 2] = 1

for i in range(100):
    if i % 10 == 0:
        plt.figure()
        plt.imshow(Z)
        plt.colorbar()
        plt.savefig("burn_{}.png".format(i / 10))
    Z = step(Z)

plt.show()
