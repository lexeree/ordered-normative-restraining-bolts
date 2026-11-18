import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from merchant import MerchantEnv
from agents import QLearner
import numpy as np
import random

if __name__ == "__main__":
    np.random.seed(7)
    random.seed(7)

    env = MerchantEnv(layout='basic', capacity=6, sunset=32, risk_fight= 1)
    a = QLearner(env, ntrain=100000, epsilon=0.25, alpha=0.1, gamma=0.925, max_episode_steps=35)
    a.train(save="[Q-learner] ctd")
    a.evaluate()
