# -*- coding: utf-8 -*-
import copy
from collections import defaultdict
import numpy as np
from VisualizeMatrix import VisualizeMatrix


def Activate(x):
    # tanh作为激活函数
    return np.tanh(x)


def Accuracy(x, y):
    length = len(x)
    counter = 0
    for i in range(length):
        if x[i] == y[i]:
            counter += 1
    return counter / length


def MSE(x, y):
    return np.sqrt(np.sum((x - y)**2)) / len(x)


def Sign(x):
    x[x >= 0] = 1.0
    x[x < 0] = -1.0
    return x


class Hopfield(object):
    """
    假设输入图像的大小为m*n,则Hopfield网络中的神经元个数为m*n
    """
    learning_rate = 1e-1
    weight_dict = defaultdict()

    status_dict = defaultdict()

    def __init__(self, num_of_neurons, num_of_iter) -> None:
        self.num_of_iter = num_of_iter
        self.N = num_of_neurons
        X = []
        for i in range(num_of_neurons**2):
            X.append(random.gauss(0, 0.3))
        X = np.array(X)
        X = X.reshape(num_of_neurons, num_of_neurons)
        X = np.triu(X)
        X += X.T - np.diag(X.diagonal())
        self.weight = X

    def Train(self, dataset):
        matrices = dataset['matrix']
        labels = dataset['label']

        for i in range(10):
            self.weight_dict[i] = copy.deepcopy(self.weight)

        for i in range(len(matrices)):  # for each number matrix
            mat = np.array(matrices[i])
            label = labels[i]
            weights = self.weight_dict[label]

            mat = mat.flatten()

            for iteration in range(self.num_of_iter):
                y = Activate(np.matmul(weights, mat))  # update status
                y = Sign(y)
                output = y
                # Hebb规则 dW = CYX
                delta_w = self.learning_rate * np.matmul(
                    y.reshape((30, 1)), mat.reshape((1, 30)))
                weights -= delta_w

            if label not in self.status_dict:
                self.status_dict[label] = output
            else:
                self.status_dict[label] += output

        self.status_dict[label] /= len(self.status_dict[label])

    def Predict(self, _data):
        scores = []
        _status = np.array(_data).flatten()
        for num in range(10):
            weight = self.weight_dict[num]
            # VisualizeMatrix(weight.reshape(30, 30), 'weight @ %s' % num)
            _res = Activate(np.matmul(weight, _status))
            _res = Sign(_res)
            # VisualizeMatrix(_res.reshape(6, 5), 'activation @ %s' % num)
            sc = MSE(_res, self.status_dict[num])
            scores.append(sc)

        return scores


if __name__ == '__main__':
    from DataMaker import *

    training_set = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 100
    test_set = np.random.randint(0, 10, 10)
    model = Hopfield(6 * 5, 200)  # 迭代10轮
    data = DataMaker(training_set).AddNoise(0.2)
    model.Train(data)

    examples = DataMaker(test_set).AddNoise(0.1)

    val = []

    for t, dat in zip(test_set, examples['matrix']):
        # VisualizeMatrix(dat, 'raw')
        result_vector = model.Predict(dat)
        res = np.argmax(result_vector)
        print('test: {} predict: {}'.format(t, res))
        val.append(res)

    model_score = Accuracy(val, test_set)
    print('Accuracy score %s' % model_score)
