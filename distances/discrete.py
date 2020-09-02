import numpy as np
import math

from typing import Callable, Dict
from numba import jit, types
from numba import typed
from timeit import default_timer as timer


class DiscreteFrechet(object):
    """
    Calculates the discrete Fréchet distance between two poly-lines using the
    original recursive algorithm
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
def get_linear_frechet(p: np.ndarray, q: np.ndarray,
                       dist_func: Callable[[np.ndarray, np.ndarray], float]) \
        -> np.ndarray:
    n_p = p.shape[0]
    n_q = q.shape[0]
    ca = np.zeros((n_p, n_q), dtype=np.float64)

    for i in range(n_p):
        for j in range(n_q):
            d = dist_func(p[i], q[j])

            if i > 0 and j > 0:
                ca[i, j] = max(min(ca[i - 1, j],
                                   ca[i - 1, j - 1],
                                   ca[i, j - 1]), d)
            elif i > 0 and j == 0:
                ca[i, j] = max(ca[i - 1, 0], d)
            elif i == 0 and j > 0:
                ca[i, j] = max(ca[0, j - 1], d)
            elif i == 0 and j == 0:
                ca[i, j] = d
            else:
                ca[i, j] = np.infty
    return ca


class LinearDiscreteFrechet(DiscreteFrechet):

    def __init__(self, dist_func):
        DiscreteFrechet.__init__(self, dist_func)
        # JIT the numba code
        self.distance(np.array([[0.0, 0.0], [1.0, 1.0]]),
                      np.array([[0.0, 0.0], [1.0, 1.0]]))

    def distance(self, p: np.ndarray, q: np.ndarray) -> float:
        n_p = p.shape[0]
        n_q = q.shape[0]
        self.ca = get_linear_frechet(p, q, self.dist_func)
        return self.ca[n_p - 1, n_q - 1]


@jit(nopython=True)
def distance_matrix(p: np.ndarray,
                    q: np.ndarray,
                    dist_func: Callable[[np.array, np.array], float]) \
        -> np.ndarray:
    n_p = p.shape[0]
    n_q = q.shape[0]
    dist = np.zeros((n_p, n_q), dtype=np.float64)
    for i in range(n_p):
        for j in range(n_q):
            dist[i, j] = dist_func(p[i], q[j])
    return dist


class VectorizedDiscreteFrechet(DiscreteFrechet):

    def __init__(self, dist_func):
        DiscreteFrechet.__init__(self, dist_func)
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
    pairs = np.zeros((dim, 2), dtype=np.int64)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx // 2
        for i in range(dx):
            pairs[i, 0] = x
            pairs[i, 1] = y
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy // 2
        for i in range(dy):
            pairs[i, 0] = x
            pairs[i, 1] = y
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    return pairs


# @jit(nopython=True)
# def rc(row: types.int64, col: types.int64) -> types.int64:
#     return (row << 32) + col


@jit(nopython=True)
def get_rc(a: Dict, row: types.int64, col: types.int64,
           d: types.float64 = np.inf) -> types.float64:
    kk = (row, col)
    if kk in a:
        return a[kk]
    else:
        return d


@jit(nopython=True, fastmath=True)
def fast_distance_matrix(p: np.ndarray,
                         q: np.ndarray,
                         diag: np.ndarray,
                         dist_func: Callable[[np.array, np.array], float]) -> np.ndarray:
    n_diag = diag.shape[0]
    diag_max = 0.0
    i_min = 0
    j_min = 0

    # Create the distance array
    dist = dict()  # typed.Dict.empty(key_type=types.int64, value_type=types.float64)

    # Fill in the diagonal with the seed distance values
    for i in range(n_diag):
        di = diag[i, 0]
        dj = diag[i, 1]
        d = dist_func(p[di], q[dj])
        diag_max = max(diag_max, d)
        dist[(di, dj)] = d

    for k in range(n_diag - 1):
        i0 = diag[k, 0]
        j0 = diag[k, 1]

        p_i0 = p[i0]
        q_j0 = q[j0]

        for i in range(i0 + 1, p.shape[0]):
            kk = (i, j0)
            if kk not in dist:
                d = dist_func(p[i], q_j0)
                if d < diag_max or i < i_min:
                    dist[kk] = d
                else:
                    break
        i_min = i

        for j in range(j0 + 1, q.shape[0]):
            kk = (i0, j)
            if kk not in dist:
                d = dist_func(p_i0, q[j])
                if d < diag_max or j < j_min:
                    dist[kk] = d
                else:
                    break
        j_min = j
    return dist


@jit(nopython=True, fastmath=True)
def get_corner_min(f_mat: np.ndarray, i: int, j: int) -> float:
    if i > 0 and j > 0:
        a = min(get_rc(f_mat, i - 1, j - 1),
                get_rc(f_mat, i, j - 1),
                get_rc(f_mat, i - 1, j))
    elif i == 0 and j == 0:
        a = f_mat[(i, j)]
    elif i == 0:
        a = f_mat[(i, j - 1)]
    else:  # j == 0:
        a = f_mat[(i - 1, j)]
    return a


@jit(nopython=True)
def fast_frechet_matrix(dist,
                        diag: np.ndarray,
                        p: np.ndarray,
                        q: np.ndarray):

    for k in range(diag.shape[0]):
        i0 = diag[k, 0]
        j0 = diag[k, 1]

        for i in range(i0, p.shape[0]):
            kk = (i, j0)
            if kk in dist:
                c = get_corner_min(dist, i, j0)
                if c > dist[kk]:
                    dist[kk] = c
            else:
                break

        # Add 1 to j0 to avoid recalculating the diagonal
        for j in range(j0 + 1, q.shape[0]):
            kk = (i0, j)
            if kk in dist:
                c = get_corner_min(dist, i0, j)
                if c > dist[kk]:
                    dist[kk] = c
            else:
                break
    return dist


class FastDiscreteFrechet(object):

    def __init__(self, dist_func):
        """

        Parameters
        ----------
        dist_func:
        """
        self.dist_func = dist_func
        self.ca = typed.Dict.empty(key_type=types.int64,
                                   value_type=types.float64)
        # JIT the numba code
        self.distance(np.array([[0.0, 0.0], [1.0, 1.0]]),
                      np.array([[0.0, 0.0], [1.0, 1.0]]))

    def distance(self, p: np.ndarray, q: np.ndarray) -> float:
        diagonal = bresenham_pairs(0, 0, p.shape[0], q.shape[0])
        ca = fast_distance_matrix(p, q, diagonal, self.dist_func)
        ca = fast_frechet_matrix(ca, diagonal, p, q)
        self.ca = ca
        return ca[(p.shape[0]-1, q.shape[0]-1)]


@jit(nopython=True)
def euclidean(p: np.ndarray, q: np.ndarray) -> float:
    d = p - q
    return math.sqrt(np.dot(d, d))


@jit(nopython=True)
def haversine(p: np.ndarray,
              q: np.ndarray) -> float:
    """
    Vectorized haversine distance calculation
    :p: Initial location in radians
    :q: Final location in radians
    :return: Distance
    """
    d_lon = q[1] - p[1]
    d_lat = p[1] - p[0]

    a = math.sin(d_lat/2.0)**2 + math.cos(p[0]) * math.cos(q[0]) \
        * math.sin(d_lon/2.0)**2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return c


@jit(nopython=True)
def earth_haversine(p: np.ndarray, q: np.ndarray) -> float:
    """
    Vectorized haversine distance calculation
    :p: Initial location in degrees [lat, lon]
    :q: Final location in degrees [lat, lon]
    :return: Distances in meters
    """
    earth_radius = 6378137.0
    return haversine(np.radians(p), np.radians(q)) * earth_radius


def print_sparse_matrix(d: Dict):
    for key, value in d.items():
        col = key & ((1 << 32) - 1)
        row = key >> 32
        print("({0},{1}): {2}".format(row, col, value))


def main():
    np.set_printoptions(precision=4)

    fast_frechet = FastDiscreteFrechet(euclidean)
    linear_frechet = LinearDiscreteFrechet(euclidean)
    slow_frechet = DiscreteFrechet(euclidean)

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

    # d = np.zeros((p.shape[0], q.shape[0]))
    # for i in range(p.shape[0]):
    #     for j in range(q.shape[0]):
    #         d[i, j] = euclidean(p[i], q[j])

    start = timer()
    distance = slow_frechet.distance(p, q)
    end = timer()
    slow_time = end - start
    print("Slow : {}".format(slow_time))
    print(distance)

    start = timer()
    distance = linear_frechet.distance(p, q)
    end = timer()
    linear_time = end - start
    print("Linear : {}".format(linear_time))
    print(distance)

    start = timer()
    distance = fast_frechet.distance(p, q)
    end = timer()
    fast_time = end - start
    print("Fast : {}".format(fast_time))
    print(distance)

    print("")
    print("{} times faster than slow".format(slow_time / fast_time))
    print("{} times faster than linear".format(linear_time / fast_time))

    # print(frechet.distance(p, q))
    #
    # print_sparse_matrix(fast_frechet.ca)
    # print_sparse_matrix(fast_frechet.f)


if __name__ == "__main__":
    main()
