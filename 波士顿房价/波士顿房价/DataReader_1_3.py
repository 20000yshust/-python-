import numpy as np
import pandas as pd
from pathlib import Path

"""
X:
    x1: feature1 feature2 feature3...
    x2: feature1 feature2 feature3...
    x3: feature1 feature2 feature3...
    ......
Y:  [if regression, value]
    [if binary classification, 0/1]
    [if multiple classification, e.g. 4 category, one-hot]
"""
#建立对象
class DataReader_1_3(object):
    def __init__(self,data_file):
        self.train_file_name=data_file
        self.num_train=0
        self.XTrain=None
        self.YTrain=None
        self.XRaw=None
        self.YRaw=None
#数据读取函数
    def ReadData(self):
        train_file=Path(self.train_file_name)
        if train_file.exists():
            data = pd.read_csv(self.train_file_name,header=None)
            data1=np.array(data)
            self.XRaw=data1[:,0:13]
            Yraw=data1[:,13]
            self.YRaw=Yraw.reshape((-1,1))
            self.XTrain=self.XRaw
            self.YTrain=self.YRaw
            self.num_train=self.XRaw.shape[0]
        else:
            raise Exception("can not find the file")
#归一化
    def NormalizeX(self):
        X_new = np.zeros(self.XRaw.shape)
        num_feature = self.XRaw.shape[1]
        self.X_norm = np.zeros((num_feature,2))
        for i in range(num_feature):
            # get one feature from all examples
            col_i = self.XRaw[:,i]
            max_value = np.max(col_i)
            min_value = np.min(col_i)
            # min value
            self.X_norm[i,0] = min_value 
            # range value
            self.X_norm[i,1] = max_value - min_value 
            new_col = (col_i - self.X_norm[i,0])/(self.X_norm[i,1])
            X_new[:,i] = new_col
        #end for
        self.XTrain = X_new

    def NormalizePredicateData(self, X_raw):
        X_new = np.zeros(X_raw.shape)
        n = X_raw.shape[1]
        for i in range(n):
            col_i = X_raw[:,i]
            X_new[:,i] = (col_i - self.X_norm[i,0]) / self.X_norm[i,1]
        return X_new

    def NormalizeY(self):
        self.Y_norm = np.zeros((1,2))
        max_value = np.max(self.YRaw)
        min_value = np.min(self.YRaw)
        # min value
        self.Y_norm[0, 0] = min_value 
        # range value
        self.Y_norm[0, 1] = max_value - min_value 
        y_new = (self.YRaw - min_value) / self.Y_norm[0, 1]
        self.YTrain = y_new

    def ToOneHot(self, num_category, base=0):
        count = self.YRaw.shape[0]
        self.num_category = num_category
        y_new = np.zeros((count, self.num_category))
        for i in range(count):
            n = (int)(self.YRaw[i,0])
            y_new[i,n-base] = 1
        self.YTrain = y_new

    def GetSingleTrainSample(self, iteration):
        x = self.XTrain[iteration]
        y = self.YTrain[iteration]
        return x, y

    def GetBatchTrainSamples(self, batch_size, iteration):
        start = iteration * batch_size
        end = start + batch_size
        batch_X = self.XTrain[start:end,:]
        batch_Y = self.YTrain[start:end,:]
        return batch_X, batch_Y

    def GetWholeTrainSamples(self):
        return self.XTrain, self.YTrain

    def Shuffle(self):
        seed = np.random.randint(0,100)
        np.random.seed(seed)
        XP = np.random.permutation(self.XTrain)
        np.random.seed(seed)
        YP = np.random.permutation(self.YTrain)
        self.XTrain = XP
        self.YTrain = YP