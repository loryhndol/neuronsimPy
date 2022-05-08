# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt


class VisualizeMatrix(object):
    def __init__(self, m, title):
        plt.figure()
        plt.title(title)
        plt.imshow(m)
        plt.show()
