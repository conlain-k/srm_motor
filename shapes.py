import numpy as np
from numpy.linalg import norm


class ShapeCollection:
    def __init__(self):
        self.shapes = []

    def addShape(self, shape):
        self.shapes.append(shape)

    def computeMinDistance(self, X, Y):
        if len(self.shapes) == 0:
            return None
        dists = np.array([s.distance(X, Y) for s in self.shapes])

        minval = np.amin(dists, axis=0)
        maxval = np.amax(dists, axis=0)

        ret = minval

        # close = np.abs(minval - maxval) < 0.1

        # print(np.where(close))

        # ret[close] = np.average(dists, axis=0)[close]
        return ret


class Shape:
    def __init__(self):
        self.type = "shape"
        self.epsilon = 1e-5

    def distance(self, pos):
        raise ValueError()


class LineSegment(Shape):
    def __init__(self, p1, p2, width=None):
        super(LineSegment, self).__init__()
        if width is None:
            width = self.epsilon
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        self.length = norm(self.p2 - self.p1)
        self.width = width

        l = (self.p2 - self.p1)
        # normalized vector along length
        self.l = l / self.length
        self.n = np.array([-l[1], l[0]])

    def distance(self, X, Y):

        mid = (self.p1 + self.p2) / 2

        deltaX = X - mid[0]
        deltaY = Y - mid[1]

        delta_l = (deltaX * self.l[0]) + (deltaY * self.l[1])
        delta_n = (deltaX * self.n[0]) + (deltaY * self.n[1])

        dist = np.zeros_like(X)

        # mask for several conditions
        within_endpoints = np.abs(delta_l) < self.length / 2
        within_n = np.abs(delta_n) <= self.width

        dist = np.sqrt((np.abs(delta_l) - self.length / 2)**2 + delta_n ** 2)

        # if between endpoints, snap to the normal distance
        dist[within_endpoints] = np.abs(delta_n[within_endpoints])
        dist[within_endpoints & within_n] = np.abs(
            delta_n[within_endpoints & within_n]) - self.width

        return dist


class Circle(Shape):
    def __init__(self, center, radius):
        super(Circle, self).__init__()

        self.center = np.array(center)
        self.radius = np.array(radius)

    def distance(self, X, Y):
        dX = (X - self.center[0])
        dY = (Y - self.center[1])


        dist = np.sqrt(dX * dX + dY * dY) - self.radius

        return dist
