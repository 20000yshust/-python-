import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

from NeuralNet_1_2 import *

file_name = "housing.csv"

# 主程序
if __name__ == '__main__':
    reader = DataReader_1_3(file_name)
    reader.ReadData()
    reader.NormalizeX()

    num_input = 13
    num_output=1
    params = HyperParameters_1_1(num_input, num_output, eta=0.02, max_epoch=150, batch_size=10, eps=1e-5)
    net = NeuralNet_1_2(params)
    net.train(reader, checkpoint=0.1)