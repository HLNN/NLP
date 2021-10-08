import math
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils import *


class HMM:
    def __init__(self):
        self.trans_prop = {}
        self.emit_prop = {}
        self.start_prop = {}
        self.hidlist = []
        self.trans_sum = {}
        self.emit_sum = {}

    def __upd_trans(self, curhid, nxthid):
        """ update trans prep
        Args:
            curhid (string): current hidden state
            nxthid (string): next hidden state
        """
        if curhid in self.trans_prop:
            if nxthid in self.trans_prop[curhid]:
                self.trans_prop[curhid][nxthid] += 1
            else:
                self.trans_prop[curhid][nxthid] = 1
        else:
            self.trans_prop[curhid] = {nxthid: 1}

    def __upd_emit(self, hid, obs):
        """ update emit prep
        Args:
            hid (string): hidden state
            obs (string): observed process
        """
        if hid in self.emit_prop:
            if obs in self.emit_prop[hid]:
                self.emit_prop[hid][obs] += 1
            else:
                self.emit_prop[hid][obs] = 1
        else:
            self.emit_prop[hid] = {obs: 1}

    def __upd_start(self, hid):
        """update init prep
        Args:
            hid (string): start hidden state
        """
        if hid in self.start_prop:
            self.start_prop[hid] += 1
        else:
            self.start_prop[hid] = 1

    def fix(self, X, Y):
        """
        Args:
            X list of hidden state seq
            Y list of observed process seq
        """
        print('Training model...')
        start_time = time.thread_time()

        for x, y in zip(X, Y):
            self.__upd_start(y[0])
            for i in range(len(x) - 1):
                self.__upd_emit(y[i], x[i])
                self.__upd_trans(y[i], y[i + 1])
            i = len(x) - 1
            self.__upd_emit(y[i], x[i])
        self.hidlist = list(self.emit_prop.keys())
        self.hidlist.sort()

        num_trans = [
            sum(self.trans_prop[key].values()) for key in self.trans_prop
        ]
        self.trans_sum = dict(zip(self.trans_prop.keys(), num_trans))
        num_emit = [
            sum(self.emit_prop[key].values()) for key in self.emit_prop
        ]
        self.emit_sum = dict(zip(self.emit_prop.keys(), num_emit))

        end_time = time.thread_time()
        print(f'Training finish, using time {end_time - start_time:.3f}s')

    def predict(self, X):
        """ predict using viterbi
        Args:
            X input data
            (string): sentence split by ' '
            (list): list of split word
            (ndarray): sentences to predict
        Returns:
            ndarray: list of result
        """
        res = []
        for x in tqdm(X):
            hidnum = len(self.hidlist)
            dp = pd.DataFrame(index=self.hidlist)
            path = pd.DataFrame(index=self.hidlist)
            # 初始化 dp 矩阵（DP 矩阵: posnum * wordsnum 存储每个 word 每个 pos 的最大概率）
            start = []
            num_sentence = sum(self.start_prop.values()) + hidnum
            for hid in self.hidlist:
                sta_hid = self.start_prop.get(hid, 1e-16) / num_sentence
                sta_hid *= (self.emit_prop[hid].get(x[0], 1e-16) / self.emit_sum[hid])
                sta_hid = math.log(sta_hid)
                start.append(sta_hid)
            dp[0] = start
            # 初始化 path 矩阵
            path[0] = ['_start_'] * hidnum
            # 递推
            for t in range(1, len(x)):  # 句子中第 t 个词
                prob_hid, path_point = [], []
                for i in self.hidlist:  # i 为当前词的 pos
                    max_prob, last_point = float('-inf'), ''
                    emit = math.log(self.emit_prop[i].get(x[t], 1e-16) / self.emit_sum[i])
                    for j in self.hidlist:  # j 为上一次的 pos
                        tmp = dp.loc[j, t - 1] + emit
                        tmp += math.log(self.trans_prop[j].get(i, 1e-16) / self.trans_sum[j])
                        if tmp > max_prob:
                            max_prob, last_point = tmp, j
                    prob_hid.append(max_prob)
                    path_point.append(last_point)
                dp[t], path[t] = prob_hid, path_point
            # 回溯
            prob_list = list(dp[len(x) - 1])
            cur_pos = self.hidlist[prob_list.index(max(prob_list))]
            path_que = []
            path_que.append(cur_pos)
            for i in range(len(x) - 1, 0, -1):
                cur_pos = path[i].loc[cur_pos]
                path_que.append(cur_pos)
            # 返回结果
            obs = []
            for i in range(len(x)):
                obs.append(path_que[-i - 1])
            res.append(obs)
        return np.array(res + [[]])[:-1]


if __name__ == "__main__":
    # data_clean()
    hmm = HMM()
    _, Y = load_data()
    X, Y = Y[:, 0], Y[:, 1]
    hmm.fix(X, Y)
    print('Predict:')
    res = hmm.predict(X[0])
    print(res[0])
    print('Gt:')
    print(Y[0])

# 1. 语料库中有 26 个基本词类标记
#       形容词a、区别词b、连词c、副词d、叹词e、方位词f、语素g、前接成分h、成语i、
#       简称j、后接成分k、习惯用语l、数词m、名词n、拟声词o、介词p、量词q、代词r、
#       处所词s、时间词t、助词u、动词v、标点符号w、非语素字x、语气词y、状态词z、
#
#
# 2. 语料库中还有 74 个扩充标记：对于语素，具体区分为 Ag Bg Dg Mg Ng Rg Tg Vg Yg
#
#
# 3. 词性标注只标注基本词性，因此在数据清洗的过程中，将扩充标记归类到各个基本词类中，语素也归类到相应词类中

