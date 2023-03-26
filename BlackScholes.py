
import warnings

warnings.filterwarnings('ignore')


# import other useful libs
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import time
from tqdm import tqdm_notebook
# helper analytics
def bsPrice(spot, strike, vol, T):
    d1 = (np.log(spot / strike) + 0.5 * vol * vol * T) / vol / np.sqrt(T)
    d2 = d1 - vol * np.sqrt(T)
    return spot * norm.cdf(d1) - strike * norm.cdf(d2)


def bsDelta(spot, strike, vol, T):
    d1 = (np.log(spot / strike) + 0.5 * vol * vol * T) / vol / np.sqrt(T)
    return norm.cdf(d1)


def bsVega(spot, strike, vol, T):
    d1 = (np.log(spot / strike) + 0.5 * vol * vol * T) / vol / np.sqrt(T)
    return spot * np.sqrt(T) * norm.pdf(d1)


#

# main class
class BlackScholes:

    def __init__(self,
                 vol=0.2,
                 T1=1,
                 T2=2,
                 K=1.10,
                 volMult=1.5):

        self.spot = 1
        self.vol = vol
        self.T1 = T1
        self.T2 = T2
        self.K = K
        self.volMult = volMult

    def __init__(self,
                  vol=0.2,
                  T1=1,
                  T2=2,
                  K=1.10,
                  spot=1.0,
                  volMult=1.5):

            self.spot = spot
            self.vol = vol
            self.T1 = T1
            self.T2 = T2
            self.K = K
            self.volMult = volMult

    # training set: returns S1 (mx1), C2 (mx1) and dC2/dS1 (mx1)
    def trainingSet(self, m, anti=True, seed=None):

        np.random.seed(seed)

        # 2 sets of normal returns
        returns = np.random.normal(size=[m, 2])

        # SDE
        vol0 = self.vol * self.volMult
        R1 = np.exp(-0.5 * vol0 * vol0 * self.T1 + vol0 * np.sqrt(self.T1) * returns[:, 0])
        R2 = np.exp(-0.5 * self.vol * self.vol * (self.T2 - self.T1) \
                    + self.vol * np.sqrt(self.T2 - self.T1) * returns[:, 1])
        S1 = self.spot * R1
        S2 = S1 * R2

        # payoff
        pay = np.maximum(0, S2 - self.K)

        # two antithetic paths
        if anti:

            R2a = np.exp(-0.5 * self.vol * self.vol * (self.T2 - self.T1) \
                         - self.vol * np.sqrt(self.T2 - self.T1) * returns[:, 1])
            S2a = S1 * R2a
            paya = np.maximum(0, S2a - self.K)

            X = S1
            Y = 0.5 * (pay + paya)

            # differentials
            Z1 = np.where(S2 > self.K, R2, 0.0).reshape((-1, 1))
            Z2 = np.where(S2a > self.K, R2a, 0.0).reshape((-1, 1))
            Z = 0.5 * (Z1 + Z2)

        # standard
        else:

            X = S1
            Y = pay

            # differentials
            Z = np.where(S2 > self.K, R2, 0.0).reshape((-1, 1))

        return X.reshape([-1, 1]), Y.reshape([-1, 1]), Z.reshape([-1, 1])

    # test set: returns a grid of uniform spots
    # with corresponding ground true prices, deltas and vegas
    def testSet(self, lower=0.35, upper=1.65, num=100, seed=None):

        spots = np.linspace(lower, upper, num).reshape((-1, 1))
        # compute prices, deltas and vegas
        prices = bsPrice(spots, self.K, self.vol, self.T2 - self.T1).reshape((-1, 1))
        deltas = bsDelta(spots, self.K, self.vol, self.T2 - self.T1).reshape((-1, 1))
        vegas = bsVega(spots, self.K, self.vol, self.T2 - self.T1).reshape((-1, 1))
        return spots, spots, prices