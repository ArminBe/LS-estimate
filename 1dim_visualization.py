# packages:
import numpy as np
from scipy.stats import multivariate_normal, uniform, norm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
import random
import time
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from mpl_toolkits.mplot3d import Axes3D

#############################################################################################

dim = 1  # Dimension
degree = 2  # maximal degree
n = 200  # Anzahl an gezogenen Punkten/Samples im Zustandsraum
K = 8   # Gesamtanzahl an Zellen;
tolerance = 0.05

###################################################################################################


# Generiere X normalverteilt
X = np.random.normal(0, 1, n)

# Definiere Y als eine Funktion von X, z.B. Y = X^2 + Rauschterm
epsilon = np.random.normal(0, 0.1, n)  # Kleiner normalverteilter Rauschterm
Y = np.exp(X) - 3 * X**2 + 5 * epsilon  # Y ist nicht normalverteilt


#################################################################################################

# Change for other dimensions
minx = np.amin(X) - tolerance
maxx = np.amax(X) + tolerance
partition = np.linspace(minx, maxx, num=K + 1)

V = []
deg_cell = np.zeros(K)

for j in range(K):
    u = lambda x, j=j: np.where((partition[j] <= x) & (x < partition[j + 1]), 1, 0)
    numberofx = 0
    for x in X:
        if np.where((partition[j] <= x) & (x < partition[j + 1]), 1, 0):
            numberofx = numberofx + 1
    deg_cell[j] = min(degree + 1, numberofx)
    for k in range(min(degree + 1, numberofx)):
        V.append(lambda x, k=k, u=u: x ** k * u(x))


# Empirical scalar product
def empirical_inner_product(f, g, X):
    return (1 / n) * np.sum(f(X) * g(X))


# Gram-Schmidt-Orthonormalization
def gram_schmidt(X, V):
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
            proj = empirical_inner_product(V[i], u_j, X) / empirical_inner_product(u_j, u_j, X)
            u_i = lambda x, u_i=u_i, u_j=u_j, proj=proj: u_i(x) - proj * u_j(x)

        # Add the new orthogonal polynom
        U.append(u_i)
        print("done", i)

    # normalize the polynoms
    U_norm = []
    for u in U:
        norm = np.sqrt(empirical_inner_product(u, u, X))
        U_norm.append(lambda x, u=u, norm=norm: u(x) / norm)
    return U_norm


# Perform the Gram-Schmidt orthonormalization
orthonormal_basis = gram_schmidt(X, V)

print("done!")

# define B
B = np.zeros((n, len(V)))

for i, x_i in enumerate(X):
    for j, func in enumerate(orthonormal_basis):
        B[i, j] = func(x_i)
    # print(i)


# computation of the functional
def functionalJ(a):
    return np.linalg.norm(np.dot(B, a) - Y, ord=2) ** 2


# computation of the gradient
def gradientJ(a):
    return - 2 * np.dot(np.transpose(B), Y - np.dot(B, a))


# compute the linear combination of a parameter a and the Basis V
def linearcombination(a, V):
    return lambda x: sum(f(x) for f in [lambda x, i=i: a[i] * V[i](x) for i in range(len(a))])


# define LS regression estimator
def LSestimator(a, point):
    return linearcombination(a, orthonormal_basis)(point)


# gradient descent
def gradient_descent(a_0, stepsize, num_iterations):
    a = a_0
    #print(functionalJ(a))
    a_history = [a]  # list to store the history
    for i in range(num_iterations):
        gradient = gradientJ(a)
        a = a - stepsize * gradient
        a_history.append(a)
        #print(functionalJ(a))
    return a, a_history


###################################
# Experiment zu rate

# Mit ONB

start_a = np.zeros(len(V)) # start

stepsize = 1/(40*n)
aopt, a_history = gradient_descent(start_a, 1/(2*n), 1)
a, a_history = gradient_descent(start_a, stepsize, 200)

# Erzeuge einen Bereich von x-Werten, die du plotten möchtest
x_values =  np.arange(len(a_history))

# Berechne die y-Werte, indem du LSestimator(a, x) auf jeden x-Wert anwendest
y_values = np.linalg.norm(a_history-aopt, axis=1)
y_values2 = ((1- n * stepsize)/(1 + n * stepsize))**x_values


plt.yscale('log')

# Erstelle den Plot
plt.plot(x_values, y_values, 'r', label='GD with ONB')
plt.plot(x_values, y_values2, 'g', label='((1 - n * tau)/(1 + n * tau))^k')


plt.xlabel('Iterationsschritte')
plt.ylabel('Error')

plt.legend()

# Zeige das Plot
plt.grid(True)
plt.show()
print(functionalJ(aopt))


# Ohne ONB
print('ohne ONB')
# define B
B = np.zeros((n, len(V)))

for i, x_i in enumerate(X):
    for j, func in enumerate(V):
        B[i, j] = func(x_i)
    # print(i)

# computation of the functional
def functionalJ2(a):
    return np.linalg.norm(np.dot(B, a) - Y, ord=2) ** 2

# computation of the gradient
def gradientJ2(a):
    return - 2 * np.dot(np.transpose(B), Y - np.dot(B, a))

# gradient descent
def gradient_descent2(a_0, stepsize, num_iterations):
    a = a_0
    a_history = [a]  # list to store the history
    for i in range(num_iterations):
        gradient = gradientJ2(a)
        a = a - stepsize * gradient
        a_history.append(a)
        print(functionalJ2(a))
    return a, a_history

start_a = np.zeros(len(V)) # start
stepsize = 1/(10*n)
aopt2, a_history = gradient_descent2(start_a, 1/(10*n), 10000)
a, a_history = gradient_descent2(start_a, stepsize, 100)

# Erzeuge einen Bereich von x-Werten, die du plotten möchtest
x_values = np.arange(len(a_history))

# Berechne die y-Werte, indem du LSestimator(a, x) auf jeden x-Wert anwendest
y_values = np.linalg.norm(a_history-aopt2, axis=1)
# y_values2 = ((1-stepsize)/(1+stepsize))**x_values


plt.yscale('log')

# Erstelle den Plot
plt.plot(x_values, y_values, label='GD without ONB')
# plt.plot(x_values, y_values2, label='((1-tau)/(1+tau))^k')


plt.xlabel('Iterationsschritte')
plt.ylabel('Error')

plt.legend()
# Zeige das Plot
plt.grid(True)
plt.show()
print(functionalJ2(aopt2))




def LSestimator2(a, point):
    return linearcombination(a, V)(point)


# Plot zum Veranschaulichen

# Plotten der generierten Paare
plt.scatter(X, Y, alpha=0.6)
plt.title("Scatterplot der 100 i.i.d. Paare (X_i, Y_i)")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)


# Erzeuge einen Bereich von x-Werten, die du plotten möchtest
x_values = np.linspace(minx, maxx, 400)  # 400 Werte zwischen -10 und 10

# Berechne die y-Werte, indem du LSestimator(a, x) auf jeden x-Wert anwendest
y_values = LSestimator2(aopt2, x_values)
y_values2 = LSestimator(aopt, x_values)

# Erstelle den Plot
plt.plot(x_values, y_values, 'r', label='without ONB')
plt.plot(x_values, y_values2, 'g', label='with ONB')

# Füge Titel und Achsenbeschriftungen hinzu
plt.title('Plot der Funktion LSestimator(a, x)')
plt.xlabel('x')
plt.ylabel('LSestimator(a, x)')

# Zeige eine Legende
plt.legend()

# Zeige das Plot
plt.grid(True)
plt.show()













# datadependent partition


















