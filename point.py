from typing import Tuple
from math import sqrt, sin, cos, pi
import numpy as np
from scipy.optimize import minimize
from random import random

"""
There is a somewhat decent bound on the the max distance that is described in this Math overflow answer:
https://mathoverflow.net/questions/167793/bound-on-maximum-distance-between-points-on-a-unit-n-sphere

m Points on a d-sphere have a maximum possible distance of D
m*V(d, 2 arcsin(D(m, d)/4) ) < 1, where V(d, r) is the hyperspherical cap function.
https://en.wikipedia.org/wiki/Spherical_cap#Hyperspherical_cap

"""

class Point:
    def __init__(self, coordinates: Tuple[float], r=1):
        self.r = r
        self.spherical_coordinates = coordinates
        self.rectangular_coordinates = self.spherical_to_rectangular()

    def spherical_to_rectangular(self):
        """See https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates for details on hyperspherical coordinates"""
        rectangular_coordinates = []
        #The first n coordinates are of the form sin(t1)*sin(t2)*...*sin(t_n-1)*cos(t_n)
        for i in range(len(self.spherical_coordinates)):
            coordinate = self.r
            for j in range(i):
                coordinate *= sin(self.spherical_coordinates[j])
            coordinate *= cos(self.spherical_coordinates[i])
            rectangular_coordinates.append(coordinate)

        # We need the final coordinate now that is all sin
        coordinate = self.r
        for j in range(len(self.spherical_coordinates)):
            coordinate *= sin(self.spherical_coordinates[j])
        rectangular_coordinates.append(coordinate)

        #print('Rect Dim:', len(rectangular_coordinates),
        # 'Dist:', sqrt(sum(coor**2 for coor in rectangular_coordinates)))
        return tuple(rectangular_coordinates)

    def __len__(self):
        return len(self.rectangular_coordinates)

    def __str__(self):
        return "(" + ", ".join([str(i/pi) + "π" for i in self.spherical_coordinates]) + ")"

    def __repr__(self):
        return str(self)


def dot_product(point1: Point, point2: Point):
    """Takes the dot product of two points (tuples of coordinates)"""
    assert len(point1) == len(point2)
    running_total = 0
    for i, j in zip(point1.rectangular_coordinates, point2.rectangular_coordinates):
        running_total += i*j
    return running_total


def l2_distance(point1: Point, point2: Point):
    """Takes the euclidean distance of two points (tuples of coordinates)"""
    assert len(point1) == len(point2)
    running_total = 0
    for i, j in zip(point1.rectangular_coordinates, point2.rectangular_coordinates):
        running_total += (i-j)**2
    return sqrt(running_total)


def cosine_distance(point1: Point, point2: Point):
    """Takes the cosine distance of two points (tuples of coordinates)
    https://en.wikipedia.org/wiki/Cosine_similarity
    """
    assert len(point1) == len(point2)
    return dot_product(point1, point2)/(point1.r * point2.r)


class NSphere:
    def __init__(self, dimension: int, num_points: int):
        self.points: [Point] = []
        self.dimension = dimension
        self.num_points = num_points
        self.even_distribute_points()

    def even_distribute_points(self):
        """Initializes the points on the NSphere evenly (or approximately evenly)"""
        if self.dimension == 1:  # 1-sphere is a circle
            """
            For a circle (1-sphere), points are distributed with an angle of 2*pi*i/num, where i is the i-th point
            
            I discovered that when you pick a point and average the distances to all other points, the quantity 
            approaches 4/pi as N->infinity.
            """

            self.points = [Point(tuple([2*pi*i/self.num_points])) for i in range(self.num_points)]
        elif self.dimension == 2:  # 2-Sphere is a regular sphere
            """
            Approximate an even distribution with a fibonacci_lattice
            
            It looks like it may be impossible to get a perfect even distribution of points on a sphere for a general N. 
            Exact solutions seem to only exist for N = 2, 4, 6, 8, 10, 12, 20. Every paper that I have read show that 
            any approximation algorithm still has areas that are uneven for any other values.
            
            Finding the arrangement that minimizes the max-distance between two points is sometimes called the 
            "Tammes Problem" or "Spherical Covering". The "Thomson Problem" is similar, but uses energy instead of 
            distance as the metric.
            
            As of 2015, the problem is only solved for N = 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, and 14
            
            The proof for N=14 requires complicated graph theory and topology.
            
            Reference: 
            Basic Run-down -- https://www.maths.unsw.edu.au/about/distributing-points-sphere
            Discussion & approximate algorithms -- http://emis.icm.edu.pl/journals/EM/expmath/volumes/12/12.2/pp199_209.pdf
            Solution for N=14 -- https://arxiv.org/pdf/1410.2536.pdf
            List of approximate minimum angular-separations -- http://neilsloane.com/packings/
            """
            self.fibonacci_lattice()
        else:
            """
            For 3-spheres and larger, we can slowly, non-deterministically, yet still practically approach somewhat of 
            a stable solution. Start with randomly placing points across the n-sphere, then maximize the minimum l2 
            distance of the points. This rarely finds an exact solution (likely since they don't exist), but seems to
            always get close.
            
            For 3-spheres and larger, this problem gets REALLY hard. I am still trying to wrap my head around this case.
            I cannot find any resources that show that there are ANY exact solutions for large n-spheres. A lot of the 
            literature on this problem is applied to error codes and code correction, so I'm having to learn a lot.
            
            In general this problem is called the "Spherical Code Problem". A spherical code is a list of parameters 
            (n, N, t) where n = num_dimensions, N = num_points, and t is the maximum dot product (cosine distance) 
            between two points. 
            
            Reference:
            List of approximately ideal spherical codes -- http://neilsloane.com/packings/
            """
            # raise NotImplementedError
            for i in range(self.num_points):
                coordinates: tuple[float] = tuple([2*pi * random() for _ in range(self.dimension)])
                self.points.append(Point(coordinates))
            self.optimize_input_coordinates()

    def fibonacci_lattice(self):
        """Spread out the points using the fibonacci lattice
        Reference: http://extremelearning.com.au/evenly-distributing-points-on-a-sphere/"""
        assert self.dimension == 2  # This only makes sense for a 2-sphere (a normal sphere)
        n = self.num_points
        i = np.arange(0, n, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * i / n)
        goldenRatio = (1 + 5 ** 0.5) / 2
        theta = 2 * np.pi * i / goldenRatio
        self.points = [Point(tuple([float(p), float(t)])) for p, t in zip(phi, theta)]

    def get_l2_distance_matrix(self):
        """Returns a matrix A where element A_ij is the l2 distance between point i and point j on the n-sphere"""
        matrix = np.array([l2_distance(self.points[i], self.points[j])
                           for i in range(self.num_points)
                           for j in range(self.num_points)]).reshape((self.num_points, self.num_points))
        return matrix

    def get_cosine_distance_matrix(self):
        """Returns a matrix A where element A_ij is the l2 distance between point i and point j on the n-sphere"""
        matrix = np.array([cosine_distance(self.points[i], self.points[j])
                           for i in range(self.num_points)
                           for j in range(self.num_points)]).reshape((self.num_points, self.num_points))
        return matrix

    def get_average_l2_distance_matrix(self):
        """Returns a 1d np array with the average distance to other points for each point"""
        return self.get_l2_distance_matrix().sum(axis=0)/self.num_points

    def get_average_cosine_distance_matrix(self):
        """Returns a 1d np array with the average distance to other points for each point"""
        return self.get_cosine_distance_matrix().sum(axis=0)/self.num_points

    def points_from_matrix(self, angle_matrix: np.array, num_points: int, num_dimensions: int):
        """Takes a (num_points * num dimensions) and turns it into the points on the sphere"""
        angle_matrix = angle_matrix.reshape(-1, 1)
        angle_matrix = angle_matrix.reshape((num_points, num_dimensions))  # Put it in an easier to work with shape
        self.points = []
        for point in angle_matrix:
            self.points.append(Point(tuple(point)))

    def matrix_from_points(self):
        matrix = np.vstack(tuple(point.spherical_coordinates for point in self.points)).ravel()
        return matrix

    def get_complete_average_l2_distance(self, points_matrix: np.array) -> float:
        """
        Returns the average over all points of the average l2 distance from a fixed point to other points.
        :param points_matrix: a 1D numpy array of the point coordinates back to back.
        :return: float representing the average of the average distances
        """
        self.points_from_matrix(points_matrix, self.num_points, self.dimension)
        return self.get_average_l2_distance_matrix().mean()

    def get_complete_max_l2_distance(self, points_matrix: np.array) -> float:
        """
        Returns the max over all points of the average l2 distance from a fixed point to other points.
        :param points_matrix: a 1D numpy array of the point coordinates back to back.
        :return: float representing the max of the average distances
        """
        self.points_from_matrix(points_matrix, self.num_points, self.dimension)
        return self.get_average_l2_distance_matrix().max()

    def get_complete_min_l2_distance(self, points_matrix: np.array) -> float:
        """
        Returns the max over all points of the average l2 distance from a fixed point to other points.
        :param points_matrix: a 1D numpy array of the point coordinates back to back.
        :return: float representing the max of the average distances
        """
        self.points_from_matrix(points_matrix, self.num_points, self.dimension)
        return self.get_average_l2_distance_matrix().min()

    def get_complete_min_l2_distance_reciprical(self, points_matrix: np.array) -> float:
        """Returns the reciprocal of self.get_complete_min_l2_distance, since scipy.optimize doesn't have a
        maximize function?"""
        return 1/self.get_complete_min_l2_distance(points_matrix)

    def optimize_input_coordinates(self):
        """Uses some vector calculus magic to attempt to optimize the input coordinates.
        NOTE: This is incredibly slow compared to the fibonacci lattice method, so I recommend starting with that one,
        for d = 2.
        """
        points_matrix = self.matrix_from_points()
        optimized_result = minimize(self.get_complete_min_l2_distance_reciprical, points_matrix)
        self.points_from_matrix(optimized_result.x, self.num_points, self.dimension)


if __name__ == '__main__':
    s = NSphere(1, 6)
    print(f'Circle with 6 points l2 distance and averages')
    l2_matrix = s.get_l2_distance_matrix().round(4)
    print('l2 matrix:', l2_matrix)
    print('average l2 distances:', s.get_average_l2_distance_matrix().round(4))
    print('coordinate matrix:', s.matrix_from_points())
    print('Maximum distance between two points: ', l2_matrix.max())
    print('Now, we will run the optimizer on this as a sanity check.')
    s.optimize_input_coordinates()
    print('l2 matrix:', s.get_l2_distance_matrix().round(4))
    print('average l2 distances:', s.get_average_l2_distance_matrix().round(4))
    print('coordinate matrix:', s.matrix_from_points())
    print('Maximum distance between two points: ', l2_matrix.max())
    print('')

    print(f'Circle with 6 points cosine distance and averages')
    print(s.get_cosine_distance_matrix().round(4))
    print(s.get_average_cosine_distance_matrix().round(4))
    print('\n\n')


    s2 = NSphere(2, 10)

    print(f'Sphere (fibonacci lattice) with 10 points l2 distance and averages')
    l2_matrix = s2.get_l2_distance_matrix()
    print('l2 matrix:\n', l2_matrix.round(3))
    print('average l2 distances:', s2.get_average_l2_distance_matrix().round(3))
    print('coordinate matrix:\n', s2.matrix_from_points())
    print('Maximum distance between two points: ', l2_matrix.max())
    print('Now, we will run the optimizer on this as a sanity check.')
    s.optimize_input_coordinates()
    print('l2 matrix:\n', s2.get_l2_distance_matrix().round(4))
    l2_matrix = s2.get_l2_distance_matrix().round(4)
    print('average l2 distances:', l2_matrix.round(3))
    print('coordinate matrix:\n', s2.matrix_from_points())
    print('Maximum distance between two points: ', l2_matrix.max())
    print('')

    print(f'Sphere (fibonacci lattice) with 10 points cosine distance and averages')
    print(s2.get_cosine_distance_matrix().round(3))
    print(s2.get_average_cosine_distance_matrix().round(3))
    print('\n\n')
    """
    s3 = NSphere(1, 100)

    print(f'Circle with 100 points l2 distance and averages')
    l2_matrix = s3.get_l2_distance_matrix()
    print(l2_matrix.round(3))
    print(s3.get_average_l2_distance_matrix().round(3))
    print('Maximum distance between two points: ', l2_matrix.max())
    print('')

    print(f'Circle with 100 points cosine distance and averages')
    print(s3.get_cosine_distance_matrix().round(3))
    print(s3.get_average_cosine_distance_matrix().round(3))
    print('\n\n')
    """
    s4 = NSphere(3, 10)
    print(f'3-sphere with 10 points l2 distance and averages')
    l2_matrix = s4.get_l2_distance_matrix()
    print('l2 matrix:\n', l2_matrix.round(3))
    print('average l2 distances:', s4.get_average_l2_distance_matrix().round(4))
    print('coordinate matrix:\n', s4.matrix_from_points())
    print('Maximum distance between two points: ', l2_matrix.max())
    print('\n\n')