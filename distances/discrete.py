import numpy as np
import math


class Frechet(object):
    """
    Calculates the discrete Fréchet distance between two poly-lines
    """

    def __init__(self, dist_func):
        """
        Initializes the instance with a distance function.
        :param dist_func: The distance function. It must accept two NumPy
        arrays containing the point coordinates (x, y), (lat, long)
        """
        self.dist_func = dist_func
        self.ca = np.array([0.0])
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
        self.dist = self.dist_func(p, q)
        return calculate(n_p - 1, n_q - 1)


def euclidean(p: np.ndarray, q: np.ndarray) -> float:
    d = p - q
    return math.sqrt(np.dot(d, d))


def np_euclidean(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Calculates the point-to-point distance between poly-lines p and q
    :param p: Poly-line p
    :param q: Poly-line q
    :return: Distance array
    """
    n_p = p.shape[0]
    n_q = q.shape[0]
    pp = np.repeat(p, n_q, axis=0)
    qq = np.tile(q, (n_p, 1))
    dd = pp - qq
    dist = np.sqrt(dd[:, 0] ** 2 + dd[:, 1] ** 2).reshape(n_p, n_q)
    return dist


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
    frechet = Frechet(euclidean)
    p = np.array([[0.0, 0.0],
                  [1.0, 0.0]])
    q = np.array([[0.0, 1.0],
                  [1.0, 1.0],
                  [1.0, 2.0]])
    print(frechet.distance(p, q))
    print(frechet.ca)


if __name__ == "__main__":
    main()
