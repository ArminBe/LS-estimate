import numpy as np
from algorithms import *

############################################################################################################

# Experiment rates w.r.t. n

dim = 2  # Dimension
degree = 2  # maximal degree
K = 100  # sample size
datadependent = True
tolerance = 0.05
iterations = 20


#################################################################################################

print('for n = 100')
n = 100  # number of data points

# generate X normal distributed
mean = np.zeros(dim)
cov = 1 * np.identity(dim)
X = np.random.multivariate_normal(mean, cov, size=n)

# Define Y as a function of X, e.g. Y = X^2 + noise
epsilon = np.random.multivariate_normal(mean, 0.1 * cov, size=n)  # noise
Y = 0.5 * X[:, 0] ** 4 - 2 * X[:, 0] ** 2 - 4 * X[:, 1] ** 2 + 0.5 * epsilon[:, 1]

# partition
partition = partitioning(X, dim, K, n, datadependent, tolerance)

#compute Basis
V, deg_cell, storeX, storeY = monomialbasis(dim, K, degree, X, Y, partition)

# Perform the Gram-Schmidt orthonormalization
print('orthonormalize Basis')
time_ONB_start = time.time()
orthonormal_basis = []
num_poly = 0
for j in range(K):
    orthonormal_basis.append(gram_schmidt(storeX[j], V[num_poly: num_poly + int(deg_cell[j])], deg_cell, n))
    num_poly = num_poly + int(deg_cell[j])
    print(f'{j + 1} cells out of {K} cells done')
orthonormal_basis = list(itertools.chain(*orthonormal_basis))
time_ONB = time.time() - time_ONB_start
print(f'Done with the computation of ONB. It took {time_ONB} seconds')

# compute matrix
B = matrixB(V, K, storeX, deg_cell)
B_ONB = matrixB(orthonormal_basis, K, storeX, deg_cell)

# compute optimal a
a_opt = np.zeros(len(V))
coordinate = 0
for j in range(K):
    new_coordinate = coordinate + int(deg_cell[j])
    a_opt[coordinate: new_coordinate] = 1 / n * np.dot(np.transpose(B_ONB[j]), storeY[j])
    coordinate = new_coordinate
minJ = functionalJ(a_opt, B_ONB, storeY, n, K, deg_cell)
print(f'minimum of J: {minJ} \n computed using the orthonormal basis')

# choose an initialization
start_a , a_history_armijo, time_history_armijo = gradient_descent_armijo(np.zeros(len(V)), B, 100, deg_cell, storeY, K)

# GD with armijo step size rule with
aopt_armijo_1, a_history_armijo, time_history_armijo = gradient_descent_armijo(start_a, B, iterations, deg_cell, storeY, K)
print(f' after {iterations} iterations of the gradient descent with Armijo step size rule, the functional J takes the '
      f'value {functionalJ(aopt_armijo_1, B, storeY, n, K, deg_cell)}')

y_values = np.zeros(4)
y_values[0] = np.absolute(functionalJ(aopt_armijo_1, B, storeY, n, K, deg_cell) - minJ)



print('for n = 500')
n = 500  # number of data points

# generate X normal distributed
mean = np.zeros(dim)
cov = 1 * np.identity(dim)
X = np.random.multivariate_normal(mean, cov, size=n)

# Define Y as a function of X, e.g. Y = X^2 + noise
epsilon = np.random.multivariate_normal(mean, 0.1 * cov, size=n)  # noise
Y = 0.5 * X[:, 0] ** 4 - 2 * X[:, 0] ** 2 - 4 * X[:, 1] ** 2 + 0.5 * epsilon[:, 1]

# partition
partition = partitioning(X, dim, K, n, datadependent, tolerance)

V, deg_cell, storeX, storeY = monomialbasis(dim, K, degree, X, Y, partition)

# Perform the Gram-Schmidt orthonormalization
print('orthonormalize Basis')
time_ONB_start = time.time()
orthonormal_basis = []
num_poly = 0
for j in range(K):
    orthonormal_basis.append(gram_schmidt(storeX[j], V[num_poly: num_poly + int(deg_cell[j])], deg_cell, n))
    num_poly = num_poly + int(deg_cell[j])
    print(f'{j + 1} cells out of {K} cells done')
orthonormal_basis = list(itertools.chain(*orthonormal_basis))
time_ONB = time.time() - time_ONB_start
print(f'Done with the computation of ONB. It took {time_ONB} seconds')

# compute matrix
B = matrixB(V, K, storeX, deg_cell)
B_ONB = matrixB(orthonormal_basis, K, storeX, deg_cell)

# compute optimal a
a_opt = np.zeros(len(V))
coordinate = 0
for j in range(K):
    new_coordinate = coordinate + int(deg_cell[j])
    a_opt[coordinate: new_coordinate] = 1 / n * np.dot(np.transpose(B_ONB[j]), storeY[j])
    coordinate = new_coordinate
minJ = functionalJ(a_opt, B_ONB, storeY, n, K, deg_cell)
print(f'minimum of J: {minJ} \n computed using the orthonormal basis')

# choose an initialization
start_a , a_history_armijo, time_history_armijo = gradient_descent_armijo(np.zeros(len(V)), B, 100, deg_cell, storeY, K)

# GD with armijo step size rule with
aopt_armijo_2, a_history_armijo, time_history_armijo = gradient_descent_armijo(start_a, B, iterations, deg_cell, storeY, K)
print(f' after {iterations} iterations of the gradient descent with Armijo step size rule, the functional J takes the '
      f'value {functionalJ(aopt_armijo_2, B, storeY, n, K, deg_cell)}')


y_values[1] = np.absolute(functionalJ(aopt_armijo_2, B, storeY, n, K, deg_cell) - minJ)



print('for n = 1000')
n = 1000  # number of data points

# generate X normal distributed
mean = np.zeros(dim)
cov = 1 * np.identity(dim)
X = np.random.multivariate_normal(mean, cov, size=n)

# Define Y as a function of X, e.g. Y = X^2 + noise
epsilon = np.random.multivariate_normal(mean, 0.1 * cov, size=n)  # noise
Y = 0.5 * X[:, 0] ** 4 - 2 * X[:, 0] ** 2 - 4 * X[:, 1] ** 2 + 0.5 * epsilon[:, 1]

# partition
partition = partitioning(X, dim, K, n, datadependent, tolerance)

V, deg_cell, storeX, storeY = monomialbasis(dim, K, degree, X, Y, partition)

# Perform the Gram-Schmidt orthonormalization
print('orthonormalize Basis')
time_ONB_start = time.time()
orthonormal_basis = []
num_poly = 0
for j in range(K):
    orthonormal_basis.append(gram_schmidt(storeX[j], V[num_poly: num_poly + int(deg_cell[j])], deg_cell, n))
    num_poly = num_poly + int(deg_cell[j])
    print(f'{j + 1} cells out of {K} cells done')
orthonormal_basis = list(itertools.chain(*orthonormal_basis))
time_ONB = time.time() - time_ONB_start
print(f'Done with the computation of ONB. It took {time_ONB} seconds')

# compute matrix
B = matrixB(V, K, storeX, deg_cell)
B_ONB = matrixB(orthonormal_basis, K, storeX, deg_cell)

# compute optimal a
a_opt = np.zeros(len(V))
coordinate = 0
for j in range(K):
    new_coordinate = coordinate + int(deg_cell[j])
    a_opt[coordinate: new_coordinate] = 1 / n * np.dot(np.transpose(B_ONB[j]), storeY[j])
    coordinate = new_coordinate
minJ = functionalJ(a_opt, B_ONB, storeY, n, K, deg_cell)
print(f'minimum of J: {minJ} \n computed using the orthonormal basis')

# choose an initialization
start_a , a_history_armijo, time_history_armijo = gradient_descent_armijo(np.zeros(len(V)), B, 100, deg_cell, storeY, K)

# GD with armijo step size rule with
aopt_armijo_3, a_history_armijo, time_history_armijo = gradient_descent_armijo(start_a, B, iterations, deg_cell, storeY, K)
print(f' after {iterations} iterations of the gradient descent with Armijo step size rule, the functional J takes the '
      f'value {functionalJ(aopt_armijo_3, B, storeY, n, K, deg_cell)}')


y_values[2] = np.absolute(functionalJ(aopt_armijo_3, B, storeY, n, K, deg_cell) - minJ)


print('for n = 2000')
n = 2000  # number of data points

# generate X normal distributed
mean = np.zeros(dim)
cov = 1 * np.identity(dim)
X = np.random.multivariate_normal(mean, cov, size=n)

# Define Y as a function of X, e.g. Y = X^2 + noise
epsilon = np.random.multivariate_normal(mean, 0.1 * cov, size=n)  # noise
Y = 0.5 * X[:, 0] ** 4 - 2 * X[:, 0] ** 2 - 4 * X[:, 1] ** 2 + 0.5 * epsilon[:, 1]

partition = partitioning(X, dim, K, n, datadependent, tolerance)

V, deg_cell, storeX, storeY = monomialbasis(dim, K, degree, X, Y, partition)

# Perform the Gram-Schmidt orthonormalization
print('orthonormalize Basis')
time_ONB_start = time.time()
orthonormal_basis = []
num_poly = 0
for j in range(K):
    orthonormal_basis.append(gram_schmidt(storeX[j], V[num_poly: num_poly + int(deg_cell[j])], deg_cell, n))
    num_poly = num_poly + int(deg_cell[j])
    print(f'{j + 1} cells out of {K} cells done')
orthonormal_basis = list(itertools.chain(*orthonormal_basis))
time_ONB = time.time() - time_ONB_start
print(f'Done with the computation of ONB. It took {time_ONB} seconds')

# compute matrix
B = matrixB(V, K, storeX, deg_cell)
B_ONB = matrixB(orthonormal_basis, K, storeX, deg_cell)

# compute optimal a
a_opt = np.zeros(len(V))
coordinate = 0
for j in range(K):
    new_coordinate = coordinate + int(deg_cell[j])
    a_opt[coordinate: new_coordinate] = 1 / n * np.dot(np.transpose(B_ONB[j]), storeY[j])
    coordinate = new_coordinate
minJ = functionalJ(a_opt, B_ONB, storeY, n, K, deg_cell)
print(f'minimum of J: {minJ} \n computed using the orthonormal basis')

# choose an initialization
start_a , a_history_armijo, time_history_armijo = gradient_descent_armijo(np.zeros(len(V)), B, 100, deg_cell, storeY, K)

# GD with armijo step size rule with
aopt_armijo_4, a_history_armijo, time_history_armijo = gradient_descent_armijo(start_a, B, iterations, deg_cell, storeY, K)
print(f' after {iterations} iterations of the gradient descent with Armijo step size rule, the functional J takes the '
      f'value {functionalJ(aopt_armijo_4, B, storeY, n, K, deg_cell)}')

y_values[3] = np.absolute(functionalJ(aopt_armijo_4, B, storeY, n, K, deg_cell) - minJ)


plt.yscale('log')

x_values = np.array([25, 64, 100, 196])

plt.plot(x_values, y_values, 'ro')

plt.show()






'''
#############################################################################################
# Experiment rates of Gradient Descent

dim = 2  # Dimension
degree = 3  # maximal degree
n = 3000  # sample size
K = 15 ** dim  # number of cells
datadependent = True
tolerance = 0.05

###################################################################################################

# generate X normal distributed
mean = np.zeros(dim)
cov = 1 * np.identity(dim)
X = np.random.multivariate_normal(mean, cov, size=n)

# generate X uniformly distributed
# min_value = -5  # Minimum value of the cube along each dimension
# max_value = 5  # Maximum value of the cube along each dimension

# Generate n points uniformly distributed within the d-dimensional cube
# X = np.random.uniform(low=min_value, high=max_value, size=(n, dim))

# Define Y as a function of X, e.g. Y = X^2 + noise
epsilon = np.random.multivariate_normal(mean, 0.1 * cov, size=n)  # noise
Y = 0.5 * X[:, 0] ** 4 - 2 * X[:, 0] ** 2 - 4 * X[:, 1] ** 2 + 0.5 * epsilon[:, 1]

#################################################################################################

partition = partitioning(X, dim, K, n, datadependent, tolerance)

V, deg_cell, storeX, storeY = monomialbasis(dim, K, degree, X, Y, partition)

# Perform the Gram-Schmidt orthonormalization
print('orthonormalize Basis')
time_ONB_start = time.time()
orthonormal_basis = []
num_poly = 0
for j in range(K):
    orthonormal_basis.append(gram_schmidt(storeX[j], V[num_poly: num_poly + int(deg_cell[j])], deg_cell, n))
    num_poly = num_poly + int(deg_cell[j])
    print(f'{j + 1} cells out of {K} cells done')
orthonormal_basis = list(itertools.chain(*orthonormal_basis))
time_ONB = time.time() - time_ONB_start
print(f'Done with the computation of ONB. It took {time_ONB} seconds')

# compute matrix
B = matrixB(V, K, storeX, deg_cell)
B_ONB = matrixB(orthonormal_basis, K, storeX, deg_cell)


# Experiment
print('Starting experiment')

# compute optimal a
a_opt = np.zeros(len(V))
coordinate = 0
for j in range(K):
    new_coordinate = coordinate + int(deg_cell[j])
    a_opt[coordinate: new_coordinate] = 1 / n * np.dot(np.transpose(B_ONB[j]), storeY[j])
    coordinate = new_coordinate
minJ = functionalJ(a_opt, B_ONB, storeY, n, K, deg_cell)
print(f'minimum of J: {minJ} \n computed using the orthonormal basis')

# choose an initialization
start_a = np.zeros(len(V))

# choose stepsize
stepsize_ONB = 1 / (40 * n)
a, a_history_ONB, time_history_ONB = gradient_descent(start_a, stepsize_ONB, B_ONB, 200, deg_cell, storeY, K)

# without ONB GD
stepsize = 1 / (20 * n)
iterations = 20000
aopt_GD, a_history_GD, time_history_GD = gradient_descent(start_a, stepsize, B, iterations, deg_cell, storeY, K)
print(f' after {iterations} iterations of the gradient descent with constant step size, the functional J takes the '
      f'value {functionalJ(aopt_GD, B, storeY, n, K, deg_cell)}')

# without ONB GD with armijo step size rule with
aopt_armijo, a_history_armijo, time_history_armijo = gradient_descent_armijo(start_a, B, iterations, deg_cell, storeY, K)
print(f' after {iterations} iterations of the gradient descent with Armijo step size rule, the functional J takes the '
      f'value {functionalJ(aopt_armijo, B, storeY, n, K, deg_cell)}')


# convergence comparison, error vs iteration
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# create x-values
# x_values_ONB = np.arange(len(a_history_ONB))
x_values_GD = np.arange(len(a_history_GD))
x_values_armijo = np.arange(len(a_history_armijo))

# compute the error
# Parameter error
# y_values = np.linalg.norm(a_history - a_opt, axis=1)
# y_values = np.linalg.norm(a_history[0:5000] - aopt, axis=1)
# y_values = np.linalg.norm(a_history[0:5000] - aopt, axis=1)

# functional error
y_values2 = ((1 - n * stepsize_ONB) / (1 + n * stepsize_ONB)) ** x_values_ONB
# y_values_ONB = [functionalJ(a, B_ONB, storeY, n, K, deg_cell) for a in a_history_ONB] - minJ
y_values_GD = [functionalJ(a, B, storeY, n, K, deg_cell) for a in a_history_GD] - minJ
y_values_armijo = [functionalJ(a, B, storeY, n, K, deg_cell) for a in a_history_armijo] - minJ

axs[0].set_yscale('log')
axs[1].set_yscale('log')
# axs[2].set_yscale('log')


# create the plot
# axs[2].plot(x_values_ONB, y_values2, 'y', label='((1 - n^0.5 * tau)/(1 + n^0.5 * tau))^k')
# axs[2].plot(x_values_ONB, y_values_ONB, 'r', label='GD with ONB')
axs[0].plot(x_values_GD, y_values_GD, 'b', label='GD without ONB')
axs[1].plot(x_values_armijo, y_values_armijo, 'g', label='GD with Armijo')

# labels
# axs[2].set_xlabel('Iterations')
axs[0].set_xlabel('Iterations')
axs[1].set_xlabel('Iterations')
# axs[2].set_ylabel('Error')

# title
# axs[2].set_title('GD with ONB')
axs[0].set_title('GD without ONB')
axs[1].set_title('GD with Armijo')

# show plot
axs[0].grid(True)
axs[1].grid(True)
# axs[2].grid(True)

# create legend
fig.legend(loc='upper center', ncol=4)

# Adjust spacing between the plots
plt.subplots_adjust(wspace=0.5, hspace=0.4)

plt.show()


# convergence comparison, Error vs time plot
plt.plot(time_history_GD, y_values_GD, 'b', label="Gradient Descent")
plt.plot(time_history_armijo, y_values_armijo, 'g', label="Gradient Descent with Armijo Rule")

# logarithmic scale
plt.yscale('log')

# Beschriftungen und Titel
plt.xlabel("Time (seconds)")
plt.ylabel("Error")
plt.legend(loc="upper center")
plt.grid(True)

# Plot anzeigen
plt.show()
'''
