# -*- encoding:utf-8 -*-
"""
Use perceptron to simulate logic gates
"""

import numpy as np
import matplotlib.pyplot as plt

dataset = {
    'and': np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]],
                    dtype=np.float32),
    'or': np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]],
                   dtype=np.float32),
    'not': np.array([[0, 1], [1, 0]], dtype=np.float32)
}

coeff_dict = {'and': {}, 'or': {}, 'not': {}}

params = (100, 0.1)  # (number of iteration, learning rate)

sigmoid = lambda x: 1.0 / (1 + np.exp(-3.0 * x))


def train(data, params):
    num_of_iter, lr = params
    label = data[:, -1]
    x = data[:, :-1]
    W = np.random.normal(0, 0.3, x.shape[1])
    b = np.random.normal(1, 0.2, 1)

    residual_array = []

    for i in range(num_of_iter):
        residual = []

        for j in range(x.shape[0]):
            y = perceptron(W, b, x[j], sigmoid)
            delta_W = lr * x[j] * (label[j] - y)
            W += delta_W
            delta_b = lr * (label[j] - y)
            b += delta_b

            # add residual
            residual.append(abs(label[j] - y))

        residual_array.append(np.average(residual))

    plt.figure()
    plt.plot(range(num_of_iter), residual_array)
    plt.title('residual curve during training')
    plt.show()

    return W, b


def perceptron(weights, bias, input, activate):
    return activate((np.matmul(weights, input) + bias))


def judge(x):

    a = abs(x - 1)
    b = abs(x)
    if a > b:
        return 0
    else:
        return 1


def gen_dict(dataset):
    for k, v in dataset.items():
        _w, _b = train(v, params)
        coeff_dict[k]['w'] = _w
        coeff_dict[k]['b'] = _b


def AND(x):
    _w = coeff_dict['and']['w']
    _b = coeff_dict['and']['b']
    return judge(perceptron(_w, _b, x, sigmoid))


def OR(x):
    _w = coeff_dict['or']['w']
    _b = coeff_dict['or']['b']
    return judge(perceptron(_w, _b, x, sigmoid))


def NOT(x):
    _w = coeff_dict['not']['w']
    _b = coeff_dict['not']['b']
    return judge(perceptron(_w, _b, x, sigmoid))


def XOR(x):
    A = x[0]
    B = x[1]
    A_prime = NOT([A])
    B_prime = NOT([B])

    lhs = [A, B_prime]
    rhs = [B, A_prime]
    return OR([AND(lhs), AND(rhs)])


def NAND(x):
    return NOT([AND(x)])


# 8-3 encoder
def _74LS148(x):
    A0 = OR([OR([x[0], x[2]]), OR([x[4], x[6]])])
    A1 = OR([OR([x[0], x[1]]), OR([x[4], x[5]])])
    A2 = OR([OR([x[0], x[1]]), OR([x[2], x[3]])])
    return [A2, A1, A0]


gen_dict(dataset)
testset = [[0, 0], [0, 1], [1, 0], [1, 1]]
INPUT = [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]]

print("--- test ---")
for i in testset:
    print("AND: {} {}".format(str(i), str(AND(i))))
for i in testset:
    print("OR: {} {}".format(str(i), str(OR(i))))
for i in [[0], [1]]:
    print("NOT: {} {}".format(str(i), str(NOT(i))))
for i in testset:
    print("XOR: {} {}".format(str(i), str(XOR(i))))
for i in testset:
    print("NAND: {} {}".format(str(i), str(NAND(i))))

for i in INPUT:
    print(_74LS148(i))