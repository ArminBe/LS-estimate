from algorithms import *

#############################################################################################

dim = 2  # Dimension
degree = 2  # maximal degree
n = 1000  # sample size
K = 7 ** dim  # number of cells
datadependent = False
tolerance = 0.05

###################################################################################################

# generate X normal distributed
mean = np.zeros(dim)
cov = 1 * np.identity(dim)
# X = np.random.multivariate_normal(mean, cov, size=n)

# generate X uniformly distributed
min_value = -5  # Minimum value of the cube along each dimension
max_value = 5  # Maximum value of the cube along each dimension

# Generate n points uniformly distributed within the d-dimensional cube
X = np.random.uniform(low=min_value, high=max_value, size=(n, dim))

# Define Y as a function of X
epsilon = np.random.multivariate_normal(mean, 0.1 * cov, size=n)  # noise


# Y = 0.5 * X[:, 0] ** 4 - 2 * X[:, 0] ** 2 - 4 * X[:, 1] ** 2 + 0.5 * epsilon[:, 1]

def f(x):
    # Fall: x[1] > 0
    if x[0] > 0:
        return np.exp(0.75 * x[0]) + 2 * x[1] ** 2
    # Fall: x[1] < 0
    elif x[0] < 0:
        return x[0] * np.cos(2 * x[1])
    # Fall: x[1] == 0
    else:
        return 0


Y = np.apply_along_axis(f, 1, X) + epsilon[:, 1]

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

# initialize a
start_a = np.zeros(len(V))

# with ONB

# compute optimal a
a_opt = np.zeros(len(V))
coordinate = 0
for j in range(K):
    new_coordinate = coordinate + int(deg_cell[j])
    a_opt[coordinate: new_coordinate] = 1 / n * np.dot(np.transpose(B_ONB[j]), storeY[j])
    coordinate = new_coordinate
minJ = functionalJ(a_opt, B_ONB, storeY, n, K, deg_cell)
print(f'minimum of J: {minJ} \n computed using the orthonormal basis')

# iterative methods
'''
# without ONB GD
print('Gradient descent with constant stepsize')
stepsize = 1 / (3 * n)
iterations = 10000
aopt, a_history, time_history = gradient_descent(start_a, stepsize, B, iterations, deg_cell, storeY, K)

# without ONB GD with Armijo condition
print('Gradient descent with Armijo step size rule')
iterations = 10000
aopt, a_history, time_history = gradient_descent_armijo(start_a, B, iterations, deg_cell, storeY, K)
'''

# show LS estimator for an 2 D example


# Create a range of x and y values for the plot
minx = np.min(X, axis=0)
maxx = np.max(X, axis=0)
x_vals = np.linspace(minx[0], maxx[0]-tolerance, 50)  # 50 values in the range from -5 to 5 for the x-dimension
y_vals = np.linspace(minx[1], maxx[1]-tolerance, 50)  # 50 values in the range from -5 to 5 for the y-dimension

# Create a meshgrid for 3D plotting
X1, Y1 = np.meshgrid(x_vals, y_vals)

# Compute Z-values (the LSestimator function applied to each (x, y) pair)
Z = np.array([[LSestimator(a_opt, orthonormal_basis, [x, y]) for x, y in zip(x_row, y_row)] for x_row, y_row in zip(X1, Y1)])

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X1, Y1, Z, cmap='viridis')

# Data points
X_data = X[:, 0]  # first dimension of data points
Y_data = X[:, 1]  # second dimension of data points
Z_data = Y  # Corresponding labels

# Scatter the data points on the same plot
ax.scatter(X_data, Y_data, Z_data, color='r', s=50, label='Data points')

# Axis labels
ax.set_xlabel('X_1 Dimension')
ax.set_ylabel('X_2 Dimension')
ax.set_zlabel('LSestimator Value')

# Add a title
ax.set_title('3D Plot of LSestimator')

# Show the plot
plt.show()


'''
# Visualize the change during the iterations

# Compute Z-values (the LSestimator function applied to each (x, y) pair)
Z = np.array([[LSestimator(a_history[5], V, [x, y]) for x, y in zip(x_row, y_row)] for x_row, y_row in zip(X1, Y1)])

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X1, Y1, Z, cmap='viridis')

# Data points
X_data = X[:, 0]  # first dimension of data points
Y_data = X[:, 1]  # second dimension of data points
Z_data = Y  # Corresponding labels

# Scatter the data points on the same plot
ax.scatter(X_data, Y_data, Z_data, color='r', s=50, label='Data points')

# Axis labels
ax.set_xlabel('X_1 Dimension')
ax.set_ylabel('X_2 Dimension')
ax.set_zlabel('LSestimator Value')

# Add a title
ax.set_title('3D Plot of LSestimator')

# Show the plot
plt.show()

# Compute Z-values (the LSestimator function applied to each (x, y) pair)
Z = np.array([[LSestimator(a_history[10], V, [x, y]) for x, y in zip(x_row, y_row)] for x_row, y_row in zip(X1, Y1)])

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X1, Y1, Z, cmap='viridis')

# Data points
X_data = X[:, 0]  # first dimension of data points
Y_data = X[:, 1]  # second dimension of data points
Z_data = Y  # Corresponding labels

# Scatter the data points on the same plot
ax.scatter(X_data, Y_data, Z_data, color='r', s=50, label='Data points')

# Axis labels
ax.set_xlabel('X_1 Dimension')
ax.set_ylabel('X_2 Dimension')
ax.set_zlabel('LSestimator Value')

# Add a title
ax.set_title('3D Plot of LSestimator')

# Show the plot
plt.show()

# Compute Z-values (the LSestimator function applied to each (x, y) pair)
Z = np.array([[LSestimator(a_history[15], V, [x, y]) for x, y in zip(x_row, y_row)] for x_row, y_row in zip(X1, Y1)])

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X1, Y1, Z, cmap='viridis')

# Data points
X_data = X[:, 0]  # first dimension of data points
Y_data = X[:, 1]  # second dimension of data points
Z_data = Y  # Corresponding labels

# Scatter the data points on the same plot
ax.scatter(X_data, Y_data, Z_data, color='r', s=50, label='Data points')

# Axis labels
ax.set_xlabel('X_1 Dimension')
ax.set_ylabel('X_2 Dimension')
ax.set_zlabel('LSestimator Value')

# Add a title
ax.set_title('3D Plot of LSestimator')

# Show the plot
plt.show()

# Compute Z-values (the LSestimator function applied to each (x, y) pair)
Z = np.array([[LSestimator(a_history[20], V, [x, y]) for x, y in zip(x_row, y_row)] for x_row, y_row in zip(X1, Y1)])

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X1, Y1, Z, cmap='viridis')

# Data points
X_data = X[:, 0]  # first dimension of data points
Y_data = X[:, 1]  # second dimension of data points
Z_data = Y  # Corresponding labels

# Scatter the data points on the same plot
ax.scatter(X_data, Y_data, Z_data, color='r', s=50, label='Data points')

# Axis labels
ax.set_xlabel('X_1 Dimension')
ax.set_ylabel('X_2 Dimension')
ax.set_zlabel('LSestimator Value')

# Add a title
ax.set_title('3D Plot of LSestimator')

# Show the plot
plt.show()
'''

'''
# single cell
# Create a range of x and y values for the plot
minx = partition[:, 0]
maxx = partition[:, 1]
x_vals = np.linspace(minx[0], maxx[0]-tolerance, 50)  # 50 values in the range from -5 to 5 for the x-dimension
y_vals = np.linspace(minx[1], maxx[1]-tolerance, 50)  # 50 values in the range from -5 to 5 for the y-dimension

# Create a meshgrid for 3D plotting
X1, Y1 = np.meshgrid(x_vals, y_vals)

# Compute Z-values (the LSestimator function applied to each (x, y) pair)
Z = np.array([[LSestimator(a_opt[0:len(storeX[0])], orthonormal_basis[0:len(storeX[0])], [x, y]) for x, y in zip(x_row, y_row)] for x_row, y_row in zip(X1, Y1)])

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X1, Y1, Z, cmap='viridis')

# Data points
X_data = np.array(storeX[0])[:, 0]  # first dimension of data points
Y_data = np.array(storeX[0])[:, 1]  # second dimension of data points
Z_data = storeY[0]  # Corresponding labels

# Scatter the data points on the same plot
ax.scatter(X_data, Y_data, Z_data, color='r', s=50, label='Data points')

# Axis labels
ax.set_xlabel('X_1 Dimension')
ax.set_ylabel('X_2 Dimension')
ax.set_zlabel('LSestimator Value')

# Add a title
ax.set_title('3D Plot of LSestimator')

# Show the plot
plt.show()
'''
