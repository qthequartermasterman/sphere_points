from typing import Tuple
from math import sqrt, sin, cos, pi
import numpy as np


class Point:
    def __init__(self, coordinates: Tuple[float], r=1):
        self.r = r
        self.spherical_coordinates = coordinates
        self.rectangular_coordinates = self.spherical_to_rectangular()

    def spherical_to_rectangular(self):
        """See https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates for details on hyperspherical coordinates"""
        rectangular_coordinates = []
        for i in range(len(self.spherical_coordinates)):
            coordinate = self.r
            for j in range(i-1):
                coordinate *= sin(self.spherical_coordinates[j])
            coordinate *= cos(self.spherical_coordinates[i])
            rectangular_coordinates.append(coordinate)
        for j in range(len(self.spherical_coordinates)):
            coordinate = self.r
            coordinate *= sin(self.spherical_coordinates[j])
            rectangular_coordinates.append(coordinate)
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
            For 3-spheres and larger, this problem gets REALLY hard. I am still trying to wrap my head around this case.
            I cannot find any resources that show that there are ANY exact solutions for large n-spheres. A lot of the 
            literature on this problem is applied to error codes and code correction, so I'm having to learn a lot.
            
            In general this problem is called the "Spherical Code Problem". A spherical code is a list of parameters 
            (n, N, t) where n = num_dimensions, N = num_points, and t is the maximum dot product (cosine distance) 
            between two points. 
            
            Reference:
            List of approximately ideal spherical codes -- http://neilsloane.com/packings/
            """
            raise NotImplementedError

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
        return self.get_l2_distance_matrix().sum(axis=0)

    def get_average_cosine_distance_matrix(self):
        """Returns a 1d np array with the average distance to other points for each point"""
        return self.get_cosine_distance_matrix().sum(axis=0)


s = NSphere(1, 6)
print(f'Circle with 6 points l2 distance and averages')
print(s.get_l2_distance_matrix().round(4))
print(s.get_average_l2_distance_matrix().round(4))

print(f'Circle with 6 points cosine distance and averages')
print(s.get_cosine_distance_matrix().round(4))
print(s.get_average_cosine_distance_matrix().round(4))

s2 = NSphere(2, 10)

print(f'Sphere (fibonacci lattice) with 10 points l2 distance and averages')
print(s2.get_l2_distance_matrix().round(3))
print(s2.get_average_l2_distance_matrix().round(3))

print(f'Sphere (fibonacci lattice) with 10 points cosine distance and averages')
print(s2.get_cosine_distance_matrix().round(3))
print(s2.get_average_cosine_distance_matrix().round(3))
