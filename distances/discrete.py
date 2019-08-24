import numpy as np
import math

from typing import Callable
from numba import jit
#
#
# class SparseArray(object):
#
#     def __init__(self):
#         self.array = dict()
#
#     def __setitem__(self, key, value):
#         self.array[key] = value
#
#     def __getitem__(self, item):
#         return self.array[item]
#
#     def __contains__(self, item):
#         return item in self.array


class NaiveFrechet(object):
    """
    Calculates the discrete Fréchet distance between two poly-lines using the
    naive recursive algorithm
    """

    def __init__(self, dist_func):
        """
        Initializes the instance with a pairwise distance function.
        :param dist_func: The distance function. It must accept two NumPy
        arrays containing the point coordinates (x, y), (lat, long)
        """
        self.dist_func = dist_func
        self.ca = np.array([0.0])

    def distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculates the Fréchet distance between poly-lines p and q
        This function implements the algorithm described by Eiter & Mannila
        :param p: Poly-line p
        :param q: Poly-line q
        :return: Distance value
        """

        def calculate(i: int, j: int) -> float:
            """
            Calculates the distance between p[i] and q[i]
            :param i: Index into poly-line p
            :param j: Index into poly-line q
            :return: Distance value
            """
            if self.ca[i, j] > -1.0:
                return self.ca[i, j]

            d = self.dist_func(p[i], q[j])
            if i == 0 and j == 0:
                self.ca[i, j] = d
            elif i > 0 and j == 0:
                self.ca[i, j] = max(calculate(i-1, 0), d)
            elif i == 0 and j > 0:
                self.ca[i, j] = max(calculate(0, j-1), d)
            elif i > 0 and j > 0:
                self.ca[i, j] = max(min(calculate(i-1, j),
                                        calculate(i-1, j-1),
                                        calculate(i, j-1)), d)
            else:
                self.ca[i, j] = np.infty
            return self.ca[i, j]

        n_p = p.shape[0]
        n_q = q.shape[0]
        self.ca = np.zeros((n_p, n_q))
        self.ca.fill(-1.0)
        return calculate(n_p - 1, n_q - 1)


@jit(nopython=True)
def distance_matrix(p: np.ndarray,
                    q: np.ndarray,
                    dist_func: Callable) -> np.ndarray:
    n_p = p.shape[0]
    n_q = q.shape[0]
    dist = np.zeros((n_p, n_q), dtype=np.float64)
    for i in range(n_p):
        for j in range(n_q):
            dist[i, j] = dist_func(p[i], q[j])
    return dist


class VectorizedNaiveFrechet(NaiveFrechet):

    def __init__(self, dist_func):
        NaiveFrechet.__init__(self, dist_func)
        self.dist = np.array([0.0])

    def distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculates the Fréchet distance between poly-lines p and q
        This function implements the algorithm described by Eiter & Mannila
        :param p: Poly-line p
        :param q: Poly-line q
        :return: Distance value
        """

        def calculate(i: int, j: int) -> float:
            """
            Calculates the distance between p[i] and q[i]
            :param i: Index into poly-line p
            :param j: Index into poly-line q
            :return: Distance value
            """
            if self.ca[i, j] > -1.0:
                return self.ca[i, j]

            d = self.dist[i, j]
            if i == 0 and j == 0:
                self.ca[i, j] = d
            elif i > 0 and j == 0:
                self.ca[i, j] = max(calculate(i-1, 0), d)
            elif i == 0 and j > 0:
                self.ca[i, j] = max(calculate(0, j-1), d)
            elif i > 0 and j > 0:
                self.ca[i, j] = max(min(calculate(i-1, j),
                                        calculate(i-1, j-1),
                                        calculate(i, j-1)), d)
            else:
                self.ca[i, j] = np.infty
            return self.ca[i, j]

        n_p = p.shape[0]
        n_q = q.shape[0]
        self.ca = np.zeros((n_p, n_q))
        self.ca.fill(-1.0)
        self.dist = distance_matrix(p, q, dist_func=self.dist_func)
        return calculate(n_p - 1, n_q - 1)


@jit(nopython=True)
def bresenham_pairs(x0: int, y0: int,
                    x1: int, y1: int) -> np.ndarray:
    """Generates the diagonal coordinates

    Parameters
    ----------
    x0 : int
        Origin x value
    y0 : int
        Origin y value
    x1 : int
        Target x value
    y1 : int
        Target y value

    Returns
    -------
    np.ndarray
        Array with the diagonal coordinates
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dim = max(dx, dy)
    i = 0
    pairs = np.zeros((dim, 2), dtype=np.int32)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx // 2
        while x != x1:
            pairs[i] = np.array([x, y])
            i += 1
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy // 2
        while y != y1:
            pairs[i] = np.array([x, y])
            i += 1
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    return pairs


@jit(nopython=True)
def pairwise_distance(p: np.ndarray,
                      q: np.ndarray,
                      bp: np.ndarray,
                      dist_func: Callable) -> np.ndarray:
    n = bp.shape[0]
    dist = np.zeros(n)
    for i in range(n):
        dist[i] = dist_func(p[bp[i, 0]], q[bp[i, 1]])
    return dist


@jit(nopython=True)
def fast_distance_matrix(p: np.ndarray,
                         q: np.ndarray,
                         dist_func: Callable) -> (np.ndarray, np.ndarray):
    n_p = p.shape[0]
    n_q = q.shape[0]
    bp = bresenham_pairs(0, 0, n_p, n_q)
    n_bp = bp.shape[0]

    epq = np.zeros(n_bp)
    for i in range(n_bp):
        epq[i] = dist_func(p[bp[i, 0]], q[bp[i, 1]])
    diag_max = epq.max()

    # Create the distance array
    dist = np.zeros((n_p, n_q), dtype=np.float64)
    dist.fill(-1.0)

    # Fill in the diagonal with the seed distance values
    for i in range(bp.shape[0]):
        dist[bp[i][0], bp[i][1]] = epq[i]

    for k in range(n_bp - 1):
        ij = bp[k]
        i0 = ij[0]
        j0 = ij[1]

        for i in range(i0 + 1, n_p):
            if dist[i, j0] == -1:
                d = dist_func(p[i], q[j0])
                if d < diag_max:
                    dist[i, j0] = d
                else:
                    break

        for j in range(j0 + 1, n_q):
            if dist[i0, j] == -1:
                d = dist_func(p[i0], q[j])
                if d < diag_max:
                    dist[i0, j] = d
                else:
                    break
    return dist, bp


@jit(nopython=True)
def get_corner_min(dist: np.ndarray, i: int, j: int) -> float:
    if i == 0 and j == 0:
        a = np.array([dist[i, j]])
    elif i == 0:
        a = np.array([dist[i, j-1]])
    elif j == 0:
        a = np.array([dist[i-1, j]])
    else:
        a = np.array([dist[i-1, j-1], dist[i, j-1], dist[i-1, j]])

        # Replace np.place(a, a == -1, np.inf)
        for i in range(a.shape[0]):
            if a[i] == -1:
                a[i] = np.inf
    return a.min()


@jit(nopython=True)
def fast_frechet_matrix(dist: np.ndarray,
                        bp: np.ndarray,
                        p: np.ndarray,
                        q: np.ndarray) -> np.ndarray:
    f = dist.copy()

    n_p = p.shape[0]
    n_q = q.shape[0]

    for k in range(bp.shape[0]):
        ij = bp[k]
        i0 = ij[0]
        j0 = ij[1]

        for i in range(i0, n_p):
            d = dist[i, j0]
            if d != -1:
                f[i, j0] = max(d, get_corner_min(f, i, j0))
            else:
                break

        for j in range(j0, n_q):
            d = dist[i0, j]
            if d != -1:
                f[i0, j] = max(d, get_corner_min(f, i0, j))
            else:
                break
    return f


class FastFrechet(object):

    def __init__(self, dist_func):
        """

        Parameters
        ----------
        dist_func:
        """
        self.dist_func = dist_func
        self.ca = np.array([0.0])
        self.f = np.array([0.0])

    def distance(self, p: np.ndarray, q: np.ndarray) -> float:
        self.ca, bp = fast_distance_matrix(p, q, self.dist_func)
        f = fast_frechet_matrix(self.ca, bp, p, q)
        self.f = f
        return f[p.shape[0]-1, q.shape[0]-1]


@jit(nopython=True)
def euclidean(p: np.ndarray, q: np.ndarray) -> float:
    d = p - q
    return math.sqrt(d[0] ** 2 + d[1] ** 2)


@jit(nopython=True)
def haversine(p: np.ndarray,
              q: np.ndarray) -> float:
    """
    Vectorized haversine distance calculation
    :p: Initial location in degrees [lat, lon]
    :q: Final location in degrees [lat, lon]
    :return: Distances in meters
    """
    earth_radius = 6378137.0

    rad_lat1 = math.radians(p[0])
    rad_lon1 = math.radians(p[1])
    rad_lat2 = math.radians(q[0])
    rad_lon2 = math.radians(q[1])

    d_lon = rad_lon2 - rad_lon1
    d_lat = rad_lat2 - rad_lat1

    a = math.sin(d_lat/2.0)**2 + math.cos(rad_lat1) * math.cos(rad_lat2) \
        * math.sin(d_lon/2.0)**2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    meters = earth_radius * c
    return meters


def main():
    np.set_printoptions(precision=4)

    frechet = FastFrechet(euclidean)
    p = np.array([[0.2, 2.0],
                  [1.5, 2.8],
                  [2.3, 1.6],
                  [2.9, 1.8],
                  [4.1, 3.1],
                  [5.6, 2.9],
                  [7.2, 1.3],
                  [8.2, 1.1]])
    q = np.array([[0.3, 1.6],
                  [3.2, 3.0],
                  [3.8, 1.8],
                  [5.2, 3.1],
                  [6.5, 2.8],
                  [7.0, 0.8],
                  [8.9, 0.6]])
    print(frechet.distance(p, q))

    print(frechet.ca)
    print(frechet.f)


if __name__ == "__main__":
    main()
