import numpy as np
import matplotlib.pyplot as plt
from numpy import *
from matplotlib.pyplot import *
import time
import random


def initialize():
    global y, x, LENGTH_OF_X_ARRAY, ALPHA, BETA, ETA, DECREMENTAL_RATE
    # Static variables.
    y = np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68])
    x = np.array([10., 8., 13., 9., 11., 14., 6., 4., 12., 7., 5.])
    LENGTH_OF_X_ARRAY = len(x)
    # Set random alpha, beta and eta.
    ALPHA = random.uniform(0.001, 0.01)
    # Number of iteration
    BETA = random.uniform(0.1, 1.0)
    # Learning rate
    ETA = random.uniform(0.1, 3.0)
    DECREMENTAL_RATE = 1.1


def initiliaze_matrix_variables():
    w0 = np.array([2., 1.])
    w = w0.copy()
    p = np.zeros(2)
    W = np.zeros((2, 10000000000))
    return w, p, W


def calculate_square_error(A, w):
    funct = A.dot(w)
    err = y - funct
    E_min = np.sum(err ** 2) / LENGTH_OF_X_ARRAY
    return E_min


def calculate_best_square_error_with_numpy(A):
    w_best, E, rank, s = np.linalg.lstsq(A, y)

    print("Best numpy result: ", w_best)

    E_min = calculate_square_error(A, w_best)
    print("Numpy minimum square error result: ", E_min)
    return E_min


def create_diagram(A, w):
    global ln
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.gca()

    # Set the x and y axis borders.
    ax1.set_xlim((4, 14))
    ax1.set_ylim((4, 14))

    func = A.dot(w)

    ln = plt.Line2D(xdata=x, ydata=func, linestyle='-', linewidth=2)

    ax1.add_line(ln)

    plt.plot(x, y, 'bo', alpha=0.5, markersize=5)
    return plt


class GradientDescent(object):
    def __init__(self, alpha, beta, eta):
        self.alpha = alpha
        self.beta = beta
        self.eta = eta

    def design_matrix(self):
        # Design matrix
        A = np.vstack((np.ones(LENGTH_OF_X_ARRAY), x)).T
        return A

    def calculate_gradient_descent(self, w, W, p, E_min):
        epoch = 0
        A = self.design_matrix()

        while True:
            f = A.dot(w)
            err = y - f

            W[:, epoch] = w

            ln.set_xdata(x)
            ln.set_ydata(f)

            # Mean square error
            E = np.sum(err ** 2) / LENGTH_OF_X_ARRAY
            if epoch == 0:
                temp_E = E

            # Gradient
            dE = -2. * A.T.dot(err) / LENGTH_OF_X_ARRAY
            p = dE + self.beta * p

            if epoch % 1000 == 0:
                print(epoch, ':', E)
            # print(w)
            w -= self.alpha * p

            if temp_E > E:
                self.eta /= DECREMENTAL_RATE
                temp_E = E
                if temp_E == E_min:
                    print (epoch, ':', temp_E)
                    print ("Eta: ", self.eta)
                    print ("Alpha: ", self.beta)
                    break
            else:
                self.eta *= 1.8
            epoch += 1


if __name__ == '__main__':
    initialize()
    gradient_descent = GradientDescent(ALPHA, BETA, ETA)
    w, p, W = initiliaze_matrix_variables()
    A = gradient_descent.design_matrix()
    plt = create_diagram(A, w)
    E_min = calculate_best_square_error_with_numpy(A)
    gradient_descent.calculate_gradient_descent(w, W, p, E_min)
    plt.show()
    # To see plot diagram for a while.
    time.sleep(1)
