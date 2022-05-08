# -*- coding: utf-8 -*-

import Digits
import random


class DataMaker(object):

    def __init__(self, list_of_numbers) -> None:
        self.ret_matrix = []
        self.ret_label = list_of_numbers
        for i in list_of_numbers:
            self.ret_matrix.append(Digits.GetDigits(i))

    def GetData(self):
        ret = {
            'label': self.ret_label,
            'matrix': self.ret_matrix
        }
        return ret

    def AddNoise(self, sigma):
        for x in self.ret_matrix:
            mu = 0
            _sigma = sigma
            for row in range(6):
                for col in range(5):
                    t = x[row][col] + random.gauss(mu, _sigma)
                    if t > 1.0:
                        x[row][col] = 1.0
                    elif t < -1.0:
                        x[row][col] = -1.0
                    else:
                        x[row][col] = t
        ret = {
            'label': self.ret_label,
            'matrix': self.ret_matrix
        }
        return ret
