import unittest
import sys
import examples as examples
import numpy as np
import math as math
import src.unconstrained_min as unconstrained
import matplotlib.pyplot as plt

def  plot_f_heights(X,Y, f, path_taken, title):
    params = list(zip(np.ravel(X), np.ravel(Y)))
    vals = []
    for p in params:
        vals.append(f(np.array(p))[0])
    Z = np.array(vals).reshape(X.shape[0], X.shape[1])
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z, 10)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_title(title)
    for v in path_taken:
        plt.plot(v['location'][0],v['location'][1] ,'x', c='red', ms=7)
    plt.show()

    x_axis = range(len(path_taken))
    values = list(map(get_val, path_taken))
    plt.plot(x_axis, values, 'ro', ms=2)
    plt.xlabel('iterations')
    plt.ylabel('objective function')
    plt.show()



def get_val(x):
    return x["val"]



def plot_f_heights_no_path(X, Y, f, path_taken, title):
    params = list(zip(np.ravel(X), np.ravel(Y)))
    vals = []
    vals2 = []
    vals3 = []
    for p in params:
        vals.append(f[0](np.array(p))[0])
        vals2.append(f[1](np.array(p))[0])
        vals3.append(f[2](np.array(p))[0])
    Z = np.array(vals).reshape(X.shape[0], X.shape[1])
    Z2 = np.array(vals2).reshape(X.shape[0], X.shape[1])
    Z3 = np.array(vals3).reshape(X.shape[0], X.shape[1])
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z, 5)
    CS2 = ax.contour(X, Y, Z2, 5)
    CS3 = ax.contour(X, Y, Z3, 5)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_title(title)
    plt.show()




def plot_f_heights_no_path2(X, Y, f, path_taken, title):
    d = np.linspace(-2, 16, 300)
    x, y = np.meshgrid(d, d)
    plt.imshow(((y >= 2) & (2 * y <= 25 - x) & (4 * y >= 2 * x - 8) & (y <= 2 * x - 5)).astype(int),
               extent=(x.min(), x.max(), y.min(), y.max()), origin="lower", cmap="Greys", alpha=0.3);



class TestStringMethods(unittest.TestCase):

    def test_quad_min(self):
        x0 = np.array([1, 1])
        step_size = 0.032
        max_itr = 100
        step_tolerance = 1 / (10 ** 8)
        obj_tolerance = 1 / (10 ** 12)
        res = unconstrained.line_search(examples.two_ways_matrix_multiplication_3, x0, step_size, obj_tolerance,
                                             step_tolerance, max_itr,'nt')
        X, Y = np.meshgrid(
            np.linspace(-2, 2),
            np.linspace(-2, 2),
            indexing='ij',
        )
        plot_f_heights(X, Y, examples.two_ways_matrix_multiplication_3, res['path_taken'], 'quad min')
        print(res['res'])

    def test_rosenbrock_min(self):
        x0 = np.array([2, 2])
        step_size = 0.00201
        max_itr = 100000
        step_tolerance = 1 / (10 ** 8)
        obj_tolerance = 1 / (10 ** 7)
        f  = examples.rosenbrock(np.array([1,1]), True)
        res = unconstrained.line_search(examples.rosenbrock, x0, step_size, obj_tolerance,
                                             step_tolerance, max_itr, 'nt', init_step_len= 0.001)
        # res = unconstrained.gradient_descent(examples.rosenbrock, x0, step_size, obj_tolerance,
        #                                      step_tolerance, max_itr)
        X, Y = np.meshgrid(
            np.linspace(-3, 3),
            np.linspace(-3, 3),
            indexing='ij',
        )
        plot_f_heights(X, Y, examples.rosenbrock, res['path_taken'], 'rosenbrock min')
        print(res['res'])
    # #
    # def test_lin_min(self):
    #     x0 = np.array([2, 2])
    #     step_size = 0.1
    #     max_itr = 100
    #     step_tolerance = 1 / (10 ** 8)
    #     obj_tolerance = 1 / (10 ** 7)
    #     res = unconstrained.gradient_descent(examples.linear, x0, step_size, obj_tolerance,
    #                                          step_tolerance, max_itr)
    #
    #     X, Y = np.meshgrid(
    #         np.linspace(-10, 10),
    #         np.linspace(-10, 10),
    #         indexing='ij',
    #     )
    #     plot_f_heights(X, Y, examples.linear, res['path_taken'], 'linear failure')
    #     print(res['res'])

    def test_abs_min(self):
        X, Y = np.meshgrid(
            np.linspace(-3, 3),
            np.linspace(-3, 3),
            indexing='ij',
        )
        plot_f_heights_no_path(X, Y, [examples.abs_val, examples.constrain1, examples.constrain2], None, 'abs min')



def objective_2(x):
    return [0.5 * x[0] - x[1], 9]


def plot_f_heights_no_path(X, Y, f, path_taken, title):
    params = list(zip(np.ravel(X), np.ravel(Y)))
    vals = []
    min = [0, 0]
    for p in params:
        v = f[0](np.array(p))[0]
        if v < min[0]:
            min = [v, p]
        vals.append(v)
    Z = np.array(vals).reshape(X.shape[0], X.shape[1])
    fig, ax = plt.subplots(figsize=(10, 10))

    #     d = np.linspace(-2,16,300)
    ax.imshow((

                      (-Y + X - 1 <= 0) &
                      ((-1 / 3) * Y + X + (-5 / 3) <= 0)
                      & (Y - 4 <= 0) & (X - 3 <= 0) & (Y >= 0) & (X >= 0)

              ).astype(int),
              extent=(-5, 5, -5, 5), origin='lower', cmap="Greys", alpha=0.3);

    # plot the lines defining the constraints
    CS = ax.contour(X, Y, Z, 5)

    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_title(title)
    plt.show()
    print(min)


if __name__ == '__main__':
    unittest.main()
    # d = np.linspace(-5, 5, 300)
    # X, Y = np.meshgrid(
    #     d,
    #     d,
    #     indexing='ij',
    # )
    # plot_f_heights_no_path(X, Y, [objective_2], None, 'question 2 min and constraints')


