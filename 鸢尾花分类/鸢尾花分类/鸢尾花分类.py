import numpy as np

from NeuralNet_1_2 import *

file_name = "iris.data"

# 主程序
if __name__ == '__main__':
    num_category = 3
    reader = DataReader_1_3(file_name)
    reader.ReadData()
    reader.NormalizeX()
    reader.ToOneHot(num_category, base=1)

    num_input = 4
    params = HyperParameters_1_1(num_input, num_category, eta=0.1, max_epoch=400, batch_size=6, eps=1e-3, net_type=NetType.MultipleClassifier)
    net = NeuralNet_1_2(params)
    net.train(reader, checkpoint=1)
