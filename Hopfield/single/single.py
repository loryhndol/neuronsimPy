# -*- coding:utf-8 -*-
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt

"""
name: single.py
params: 
NUM: number of neurons
C: capacitance
R: resistance
I: bias current intensity
T: weights for each input
U_init: initial voltage of the neuron
Inputs: input of the neuron
fps: frames per second of the simulation
"""

NUM = 10
C = 0.2
R = 0.3
I = np.linspace(-0.1, 0.1, NUM)
T = [0.5, 0.2, -0.5]

U_init = 0.3
Inputs = [0.5, 0.4, -0.1]
fps = 100
timeInterval = 1.0 / fps


def f_u(x):
    return 1 / (1 + np.exp(-3 * x))


def transform(f, u, T):
    temp = map(lambda x, y: x * y, list(map(f, u)), T)
    return reduce(lambda x, y: x + y, list(temp))


def updateVoltage(x_base, x_input, bias):
    t = -1 / R * x_base + transform(f_u, x_input, T) + bias
    return 1 / C * t


def single_neuron(id):
    voltage = U_init
    logging = [U_init]
    for i in range(fps):
        delta = updateVoltage(voltage, Inputs, I[id]) * timeInterval
        voltage = voltage + delta
        logging.append(voltage)
    return logging


def run():
    res = []
    for i in range(NUM):
        res.append(single_neuron(i))
    x_array = np.linspace(0, 1, fps + 1)
    plt.figure()
    plt.title('Intensity-time curve on a single neuron')
    plt.xlabel('time')
    plt.ylabel('electric current')
    plt.xlim(0, 1)
    for i in range(NUM):
        y_array = res[i]
        plt.plot(x_array,
                 y_array,
                 label=round(I[i], 3),
                 color=plt.get_cmap('coolwarm')((I[i] + 0.1) * 5))
        if i == 0 or i == NUM - 1:
            # x: 0-0.4s y: steady-state value
            plt.plot([0, 0.4], [y_array[-1], y_array[-1]], 'r--')
            plt.text(0,
                     y_array[-1] + 0.002,
                     str(round(y_array[-1], 4)),
                     fontdict={
                         'size': 8,
                         'color': 'r'
                     })

    plt.legend(title='Bias Intensity')
    plt.show()


if __name__ == '__main__':
    run()
