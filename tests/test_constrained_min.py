import unittest
import sys
import examples as examples
import numpy as np
import math as math
import matplotlib.pyplot as plt
import src.contrained_min

def  plot_f_heights(X, Y, f, path_taken, title):
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

def get_val(x):
    return x["val"]


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
    ax.imshow(((-Y - X + 1 <= 0) &
               (Y - 1 <= 0) &
               (X - 2 <= 0) &
               (- Y <= 0)
              ).astype(int),
              extent=(-5, 5, -5, 5), origin='lower', cmap="Greys", alpha=0.3);

    # plot the lines defining the constraints
    CS = ax.contour(X, Y, Z, 5)

    ax.clabel(CS, inline=True, fontsize=4)
    ax.set_title(title)
    for v in path_taken:
        plt.plot(v['location'][0],v['location'][1] ,'x', c='red', ms=2)
    plt.show()

    # Ploting the values of the objective
    x_axis = range(len(path_taken))
    values = list(map(get_val, path_taken))
    plt.plot(x_axis, values, 'ro', ms=2)
    plt.xlabel('iterations')
    plt.ylabel('objective function')
    plt.show()

    print(min)

def objective_2(x):
    return [-(x[0] + x[1]), 9]


class TestStringMethods(unittest.TestCase):

    def test_qp_min(self):
        path_taken = src.contrained_min.interior_pt(examples.test_qp,np.array([]), np.array([[1,1, 1]]),
                                           np.array([[1]]), np.array([[0.4], [0.1], [0.5]]))

        print(path_taken)


    def test_lp_min(self):
        path_taken = src.contrained_min.interior_pt_no_eq_constraint(examples.test_lp,np.array([[]]), np.array([]), np.array([]),
                                                            np.array([[0.5], [0.75]]))

        d = np.linspace(-5, 5, 300)
        X, Y = np.meshgrid(
            d,
            d,
            indexing='ij',
        )

        plot_f_heights_no_path(X, Y, [objective_2], path_taken, 'lp minimization, no equality constraints')

if __name__ == '__main__':
    unittest.main()