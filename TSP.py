import pandas as pd
import numpy as np
import math
import time

class tsp:
    def __init__(self):
        self.train_v =[]
        return

    def solve(self,w_k):
        self.train_v=w_k
        train_d = self.train_v
        dist = np.zeros((self.train_v.shape[0], train_d.shape[0]))
        for i in range(self.train_v.shape[0]):
            for j in range(train_d.shape[0]):
                dist[i, j] = math.sqrt(np.sum((self.train_v[i, :] - train_d[j, :]) ** 2))
        i = 1
        n = self.train_v.shape[0]
        j = 0
        sumpath = 0
        s = []
        s.append(0)
        start = time.clock()
        while True:
            k = 1
            Detemp = 10000000
            while True:
                l = 0
                flag = 0
                if k in s:
                    flag = 1
                if (flag == 0) and (dist[k][s[i - 1]] < Detemp):
                    j = k
                    Detemp = dist[k][s[i - 1]]
                k += 1
                if k >= n:
                    break
            s.append(j)
            i += 1
            sumpath += Detemp
            if i >= n:
                break
        #sumpath += dist[0][j]
        end = time.clock()
        print("result:")
        print(sumpath)
        for m in range(n):
            print("%s-> " % (s[m]))
        print("Running time:%s"% (end - start))
        return s,sumpath

