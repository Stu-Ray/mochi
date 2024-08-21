import numpy as np
import random
import time
from scipy.spatial.distance import cdist
from keras.models import Model
from keras.layers import Input, LSTM
import dataProcessor as dp

def build_data(logFile):
    transactions = dp.getWorkingsets(logFile)
    first_sqls = []
    for key in transactions:
        for i in range(2,6):
            if transactions[key][0][i] != 0:
                transactions[key][0][i] = 1
        first_sqls.append(transactions[key][0][0:6])
    return np.array(first_sqls)

class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, k=2, tolerance=0.001, max_iter=300):
        self.k_ = k
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    def fit(self, data):
        self.centers_ = {}
        for i in range(self.k_):
            self.centers_[i] = data[i]

        for i in range(self.max_iter_):
            self.clf_ = {}
            for i in range(self.k_):
                self.clf_[i] = []
            # print("质点:",self.centers_)
            for feature in data:
                # distances = [np.linalg.norm(feature-self.centers[center]) for center in self.centers]
                distances = []
                for center in self.centers_:
                    # 欧拉距离
                    # np.sqrt(np.sum((features-self.centers_[center])**2))
                    distances.append(np.linalg.norm(feature - self.centers_[center]))
                classification = distances.index(min(distances))
                self.clf_[classification].append(feature)

            # print("分组情况:",self.clf_)
            prev_centers = dict(self.centers_)
            for c in self.clf_:
                self.centers_[c] = np.average(self.clf_[c], axis=0)

            # '中心点'是否在误差范围
            optimized = True
            for center in self.centers_:
                org_centers = prev_centers[center]
                cur_centers = self.centers_[center]
                if abs(np.sum((cur_centers - org_centers) / org_centers * 100.0)) > self.tolerance_:
                    optimized = False
            if optimized:
                break

    def predict(self, p_data):
        distances = [np.linalg.norm(p_data - self.centers_[center]) for center in self.centers_]
        index = distances.index(min(distances))
        return index


if __name__ == '__main__':
    data = build_data("./Dataset/LOG_TPCC4.csv")
    # print(data)
    k_means = K_Means(k=2)
    k_means.fit(data)
    print(k_means.centers_)

    # for feature in predict:
    #     cat = k_means.predict(feature)