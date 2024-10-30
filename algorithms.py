# packages:
import mpmath
import numpy as np
from scipy.stats import multivariate_normal, uniform, norm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
import random
import time
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from mpl_toolkits.mplot3d import Axes3D
from itertools import product
import itertools


# create partition
def partitioning(X, dim, K, n, datadep, tolerance):
    if datadep == True:
        # create datadependent partition

        # List to store the partition boundaries for each dimension
        partition = np.zeros([dim, int(K ** (1 / dim)) + 1])

        # number of cells per dimenson
        num_blocks = int(np.ceil(K ** (1 / dim)))

        # Create partitions for each dimension
        for d in range(dim):

            # Sort the data points in the d-th dimension
            sorted_points = np.sort(X[:, d])

            # Calculate the size of each block
            block_size = n // num_blocks
            remainder = n % num_blocks

            start_idx = 0

            for i in range(num_blocks):
                # Determine the end index for this block
                end_idx = start_idx + block_size
                if i < remainder:
                    end_idx += 1  # Distribute the remainder evenly

                # Add the lower and upper bound
                lower_bound = sorted_points[start_idx]
                partition[d, i] = lower_bound
                if i == num_blocks - 1:
                    upper_bound = sorted_points[end_idx - 1] if end_idx < n else sorted_points[-1]
                    partition[d, i + 1] = upper_bound

                start_idx = end_idx
    else:
        # deterministic equidistant partition
        minx = np.min(X, axis=0) - tolerance
        maxx = np.max(X, axis=0) + tolerance
        partition = np.array([np.linspace(minx[i], maxx[i], num=int(K ** (1 / dim)) + 1) for i in range(len(minx))])
    return partition


# compute basis

# compute monomial basis
def monomialbasis(dim, K, degree, X, Y, partition):
    V = []

    # store the local degree
    deg_cell = np.zeros(K)

    counter = np.zeros(dim)
    old_l = 0
    storeX = []
    storeY = []

    # iterate through all cells
    for j in range(K):
        # determine upper and lower bounds of the cell
        lowerbounds = np.zeros(dim)
        upperbounds = np.zeros(dim)
        for l in range(dim):
            lowerbounds[l] = partition[l, int(counter[l])]
            upperbounds[l] = partition[l, int(counter[l]) + 1]

        # define the indicator function for the current cell
        u = lambda x, j=j, lowerbounds=lowerbounds, upperbounds=upperbounds: np.where(
            np.all((lowerbounds <= x) & (x < upperbounds)), 1, 0)

        # count the number of data points in this cell
        numberofx = 0
        storeX.append([])
        storeY.append([])
        for l, x in enumerate(X):
            if np.where(np.all((lowerbounds <= x) & (x < upperbounds)), 1, 0):
                numberofx = numberofx + 1
                storeX[j].append(x)
                storeY[j].append(Y[l])

        # compute the local dimension/degree
        deg_cell[j] = min(math.comb(degree + dim, dim), numberofx)

        countbasis = 0
        for exponents in product(range(degree + 1), repeat=dim):
            if sum(exponents) <= degree:
                # define monomial on the cell
                if countbasis < deg_cell[j]:
                    countbasis = countbasis + 1
                    monomial = lambda x, ex=exponents, u=u: np.prod([x[i] ** ex[i] for i in range(dim)]) * u(x)
                    V.append(monomial)
        done = False
        # set up the counter
        for l in range(dim):
            if (done == False) & (counter[l] < K ** (1 / dim) - 1):
                if old_l < l:
                    for ll in range(l):
                        counter[ll] = 0
                old_l = l
                counter[l] = counter[l] + 1
                done = True
    return V, deg_cell, storeX, storeY


# Empirical scalar product
def empirical_inner_product(f, g, X, n):
    return (1 / n) * np.sum(np.apply_along_axis(f, 1, X) * np.apply_along_axis(g, 1, X))


# Gram-Schmidt-Orthonormalization
def gram_schmidt(X, V, deg_cell, n):
    U = []  # list for othonormal basis
    count = 0
    partitioncount = 0
    for i in range(len(V)):
        u_i = V[i]
        if i == count + deg_cell[partitioncount]:
            count = count + deg_cell[partitioncount]
            partitioncount = partitioncount + 1

        # Subtract the projections
        for j in range(int(count), i):
            u_j = U[j]
            # projections of V[i] onto U[j]
            proj = empirical_inner_product(V[i], u_j, X, n)
            u_i = lambda x, u_i=u_i, u_j=u_j, proj=proj: u_i(x) - proj * u_j(x)

        # normalize u_i and add the new orthonormal polynom
        norm = np.sqrt(empirical_inner_product(u_i, u_i, X, n))
        U.append(lambda x, u_i=u_i, norm=norm: u_i(x) / norm)
    return U


# define the matrix B
def matrixB(V, K, storeX, deg_cell):
    B = []
    num_poly = 0
    for j in range(K):
        B.append([])
        B[j] = np.zeros([len(storeX[j]), int(deg_cell[j])])
        for l, func in enumerate(V[num_poly: num_poly + int(deg_cell[j])]):
            for i, x_i in enumerate(storeX[j]):
                B[j][i, l] = func(x_i)
        num_poly = num_poly + int(deg_cell[j])
    return B


# computation of the local functional
def localfunctionalJ(a, B_j, Y_j):
    return np.linalg.norm(np.dot(B_j, a) - Y_j, ord=2) ** 2


# computation of the functional
def functionalJ(a, B, storeY, n, K, deg_cell):
    sum = 0
    coordinate = 0
    # local computation
    for j in range(K):
        sum = sum + localfunctionalJ(a[coordinate: coordinate + int(deg_cell[j])], B[j], storeY[j])
        coordinate = coordinate + int(deg_cell[j])
    return sum / n


# computation of the local gradient
def gradientJ(a, B_j, Y):
    return - 2 * np.dot(np.transpose(B_j), Y - np.dot(B_j, a))


# compute the linear combination of a parameter a and the Basis V
def linearcombination(a, V):
    return lambda x: sum(f(x) for f in [lambda x, i=i: a[i] * V[i](x) for i in range(len(a))])


# define LS regression estimator
def LSestimator(a, V, point):
    return linearcombination(a, V)(point)


def gradient_descent(a_0, stepsize, B, num_iterations, deg_cell, storeY, K):
    a = a_0.copy()
    a_history = [a.copy()]  # list to store the history
    start_time = time.time()  # Record the start time
    time_history = [0]  # List to store the time at each step

    for i in range(num_iterations):
        coordinate = 0
        for j in range(K):
            new_coordinate = coordinate + int(deg_cell[j])
            gradient = gradientJ(a[coordinate: new_coordinate], B[j], storeY[j])
            a[coordinate: new_coordinate] = a[coordinate: new_coordinate] - stepsize * gradient
            coordinate = new_coordinate
        a_history.append(a.copy())
        time_history.append(time.time() - start_time)
    return a, a_history, time_history


def gradient_descent_armijo(a_0, B, num_iterations, deg_cell, storeY, K, c=1e-4, beta=0.5, alpha_init=0.005):
    a = a_0.copy()
    a_history = [a.copy()]  # List to store the history of 'a'
    start_time = time.time()  # Record the start time
    time_history = [0]  # List to store the time at each step

    for i in range(num_iterations):
        coordinate = 0
        for j in range(K):
            new_coordinate = coordinate + int(deg_cell[j])

            # Gradient calculation for the current block
            gradient = gradientJ(a[coordinate: new_coordinate], B[j], storeY[j])

            # Armijo rule for step size determination
            alpha = alpha_init
            J_old = localfunctionalJ(a[coordinate: new_coordinate], B[j], storeY[j])
            while True:
                # New candidate point based on the current alpha
                a_new = a[coordinate: new_coordinate] - alpha * gradient

                # Compute the new objective function value
                J_new = localfunctionalJ(a_new, B[j], storeY[j])

                # Check Armijo condition
                if J_new <= J_old - c * alpha * np.linalg.norm(gradient) ** 2:
                    break  # The Armijo condition is satisfied, so exit the loop
                else:
                    alpha *= beta  # Reduce step size

            # Update 'a' with the new step
            a[coordinate: new_coordinate] = a_new
            coordinate = new_coordinate

        a_history.append(a.copy())  # Store a copy of the updated 'a'
        time_history.append(time.time() - start_time)

    return a, a_history, time_history
