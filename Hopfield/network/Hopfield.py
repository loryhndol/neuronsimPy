# -*- coding: utf-8 -*-
import copy

import numpy as np

from VisualizeMatrix import VisualizeMatrix


def Activate(x):
    # sigmoid activation function
    return 1 / (np.exp(-3 * x) + 1)


def Accuracy(x, y):
    length = len(x)
    counter = 0
    for i in range(length):
        if x[i] == y[i]:
            counter += 1
    return counter / length


def MSE(x, y):
    return np.sqrt(np.sum((x - y) ** 2)) / len(x)


def Sign(x, threshold):
    x[x > threshold] = 1
    x[x <= threshold] = -1
    return x


class Hopfield(object):
    """
    suppose the input image is m * n, and Hopfield network will have m * n neurons
    """
    threshold = 0.5
    learning_rate = 0.01
    weight_dict = {
        0: [],
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [],
        7: [],
        8: [],
        9: []
    }

    bias_dict = {
        0: [],
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [],
        7: [],
        8: [],
        9: []
    }

    status_dict = {
        0: [],
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [],
        7: [],
        8: [],
        9: []
    }

    def __init__(self, num_of_neurons, num_of_iter) -> None:
        self.num_of_iter = num_of_iter
        self.N = num_of_neurons
        X = []
        for i in range(num_of_neurons ** 2):
            X.append(random.gauss(0, 0.3))
        X = np.array(X)
        X = X.reshape(num_of_neurons, num_of_neurons)
        X = np.triu(X)
        X += X.T - np.diag(X.diagonal()) * 2
        self.weight = X
        self.bias = np.random.randn(num_of_neurons) - 0.5

    def Train(self, dataset):
        matrices = dataset['matrix']
        labels = dataset['label']

        for i in range(len(matrices)):
            weights = self.weight.copy()
            bias = self.bias.copy()
            mat = np.array(matrices[i])
            label = labels[i]

            mat = mat.flatten()

            for iteration in range(self.num_of_iter):
                mat = Activate(np.matmul(weights, mat) + self.bias)
                Sign(mat, self.threshold)

                # Hebb's Rule
                for rows in range(self.N):
                    for cols in range(rows, self.N, 1):
                        if mat[rows] * mat[cols] > 0:
                            weights[rows][cols] += self.learning_rate
                            weights[cols][rows] += self.learning_rate
                            bias[rows] += self.learning_rate
                            bias[cols] += self.learning_rate
                        else:
                            weights[rows][cols] -= self.learning_rate
                            weights[cols][rows] -= self.learning_rate
                            bias[rows] -= self.learning_rate
                            bias[cols] -= self.learning_rate
            weights = weights - np.diag(weights.diagonal())
            self.weight_dict[label].append(weights)
            self.bias_dict[label].append(bias)
            self.status_dict[label].append(mat)

        for num in range(10):
            tmp = np.sum(self.weight_dict[num], axis=0)
            self.weight_dict[num] = tmp / self.N

            tmp = np.sum(self.bias_dict[num], axis=0)
            self.bias_dict[num] = tmp / self.N

            cnt = len(self.status_dict[num])
            tmp = np.sum(self.status_dict[num], axis=0)
            self.status_dict[num] = tmp / cnt

    def Predict(self, _data):
        scores = []
        _status = np.array(_data).flatten()
        for num in range(10):
            weight = self.weight_dict[num]
            bias = self.bias_dict[num]
            if num == 1:
                VisualizeMatrix(np.array(weight).reshape(30, 30), 'weight @ 1')
            _res = Activate(np.matmul(weight, _status) + bias)
            Sign(_res, self.threshold)
            if num == 1:
                VisualizeMatrix(_res.reshape(6, 5), 'after-threshold-activation @ 1')
            sc = MSE(_res, self.status_dict[num])
            scores.append(sc)

        return scores


if __name__ == '__main__':
    from DataMaker import *

    training_set = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6,
                    6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9]
    test_set = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    model = Hopfield(6 * 5, 50)  # iterates 50 rounds
    data = DataMaker(training_set).AddNoise(0.1)
    model.Train(data)

    examples = DataMaker(test_set).AddNoise(0.08)

    val = []

    for dat in examples['matrix']:
        VisualizeMatrix(dat, 'raw')
        result_vector = model.Predict(dat)
        res = np.argmin(result_vector)
        print('The number is %s' % res)
        val.append(res)

    model_score = Accuracy(val, test_set)
    print('Accuracy score %s' % model_score)
