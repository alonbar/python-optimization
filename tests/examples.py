# Question 3 implememntation
import numpy as np
import math as math



def two_ways_matrix_multiplications(Q, x, calc_hessian):
    val = x.T.dot(Q.dot(x))
    grad = Q.dot(x)  # Like we proved in HW1, this is the grad of the function x^T * Q * x
    hessian = np.empty([Q.shape[0], Q.shape[0]])
    # The hessian of a quadratic form is q_i_j + q_j_i , where q_i_j is the scalar of the original matrix Q
    if calc_hessian:
        for i in range(hessian.shape[0]):
            for j in range(hessian.shape[0]):
                hessian[i, j] = Q[i, j] + Q[j, i]
    return [val, grad, hessian]


def test_qp(x, t):
    obj_val = t * (x[0][0] ** 2 + x[1][0] ** 2 + (x[2][0] + 1) ** 2) - math.log(x[0][0]) - math.log(x[1][0]) - math.log(x[2][0])
    grad = np.array([[2 * x[0][0] * t - (1 / x[0][0])], [2 * x[1][0] * t - (1 / x[1][0])], [2 * x[2][0] + 2 - (1 / x[2][0])]])
    hessian = np.array([[2 * t + (1 / (x[0][0] ** 2)), 0, 0], [0, 2 * t + (1 / (x[1][0] ** 2)), 0], [0, 0, 2 * t + (1 / (x[2][0] ** 2))]])
    return [obj_val, grad, hessian]


def test_lp(x, t):
    obj_val = -t * (x[0][0] + x[1][0]) - math.log(-(-x[0][0] - x[1][0] + 1)) - math.log(-(x[1][0] - 1)) - math.log(-(x[0][0] - 2)) - math.log(x[1][0])
    grad = np.array([[-1 - (1 / (x[0][0] + x[1][0] - 1 )) + (1 / (2 - x[0][0]))], [-1 - (1 / (x[0][0] + x[1][0] - 1)) + (1 / (1 - x[1][0])) - 1/x[1][0]]])
    hessian = np.array([[(1 / ((x[0][0] + x[1][0] - 1) ** 2)) + (1 / (2 - x[0][0]) ** 2), 1/ (x[0][0] + x[1][0] - 1)**2], [1 / (x[0][0] + x[1][0] - 1) ** 2, (1 / (x[0][0] + x[1][0] - 1 )) + (1/(1-x[1][0])**2) + 1/x[1][0] ** 2]])
    return [obj_val, grad, hessian]


def two_ways_matrix_multiplication_1(x, calc_hessian=False):
    Q = np.array([[1, 0], [0, 1]])
    return two_ways_matrix_multiplications(Q, x, calc_hessian)


def two_ways_matrix_multiplication_2(x, calc_hessian=False):
    Q = np.array([[5, 0], [0, 1]])
    return two_ways_matrix_multiplications(Q, x, calc_hessian)


def two_ways_matrix_multiplication_3(x, calc_hessian=False):
    A = np.array([[math.sqrt(3) / 2, -0.5], [0.5, math.sqrt(3) / 2]])
    simple = np.array([[5, 0], [0, 1]])
    Q = A.T.dot(simple.dot(A))
    return two_ways_matrix_multiplications(Q, x, calc_hessian)


def rosenbrock(x, calc_hessian=False):
    val = sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)
    # print('comparing rosen: mine: %f them: %f' % (rosen(x), val))
    grad = np.array([400 * x[0]**3 - 400 * x[0] * x[1] + 2 * x[0] - 2, 200 * (x[1] - x[0] ** 2)])
    hessian = None
    if calc_hessian:
        hessian = np.array([[1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]], [-400 * x[0], 200]])
    return [val, grad, hessian]


def linear(x):
    a = np.array([2, 3])
    val = a.T.dot(x)
    return [val, a]  # Took the grad I solved in HW1


def abs_val(x):
    return [abs(x[0]) + abs(x[1]), 9]


def constrain1(x):
    return [(x[0] - 1) ** 2 + (x[1] - 1) ** 2 - 1, 9]


def objective_2(x):
    return [0.5 * x[0] - x[1], 9]


def constraint_2_1(x):
    return [0.5 * x[0] - x[1], 9]



if __name__ == "__main__":
    # execute only if run as a script
    print(linear(np.array([2, 1])))
