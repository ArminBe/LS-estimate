from algorithms import *

############################################################################################################

# Experiment rates w.r.t. K_n

dim = 2  # Dimension
degree = 2  # maximal degree
n = 2000  # sample size
datadependent = True
tolerance = 0.05
iterations = 20

###################################################################################################

# generate X normal distributed
mean = np.zeros(dim)
cov = 1 * np.identity(dim)
X = np.random.multivariate_normal(mean, cov, size=n)

# Define Y as a function of X, e.g. Y = X^2 + noise
epsilon = np.random.multivariate_normal(mean, 0.1 * cov, size=n)  # noise
Y = 0.5 * X[:, 0] ** 4 - 2 * X[:, 0] ** 2 - 4 * X[:, 1] ** 2 + 0.5 * epsilon[:, 1]

#################################################################################################

print('for K = 25')
K = 5 ** dim  # number of cells

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
aopt_armijo_1, a_history_armijo, time_history_armijo = gradient_descent_armijo(start_a, B, iterations, deg_cell, storeY, K)
print(f' after {iterations} iterations of the gradient descent with Armijo step size rule, the functional J takes the '
      f'value {functionalJ(aopt_armijo_1, B, storeY, n, K, deg_cell)}')

y_values = np.zeros(4)
y_values[0] = functionalJ(aopt_armijo_1, B, storeY, n, K, deg_cell) - minJ

print('for K = 64')
K = 8 ** dim  # number of cells

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


y_values[1] = functionalJ(aopt_armijo_2, B, storeY, n, K, deg_cell) - minJ



print('for K = 100')
K = 10 ** dim  # number of cells

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


y_values[2] = functionalJ(aopt_armijo_3, B, storeY, n, K, deg_cell) - minJ


print('for K = 196')
K = 14 ** dim  # number of cells

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

y_values[3] = functionalJ(aopt_armijo_4, B, storeY, n, K, deg_cell) - minJ

plt.yscale('log')

x_values = np.array([25, 64, 100, 196])

plt.plot(x_values, y_values, 'ro')

plt.show()
