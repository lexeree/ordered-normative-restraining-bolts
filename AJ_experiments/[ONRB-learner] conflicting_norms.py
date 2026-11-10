from merchant import MerchantEnv
from agents import ONRBAgent1
from dfas.dfas import DeliveryDFA, DangerDFA
import numpy as np
import random

from weight_computation import get_weights

if __name__ == "__main__":
    np.random.seed(7)
    random.seed(7)

    env = MerchantEnv(layout='basic', capacity=6, sunset=32, risk_fight= 1)

    dfa1 = DeliveryDFA(reward=1)
    dfa2 = DangerDFA(reward=1)

    ntrain = 100000
    epsilon = 0.25
    alpha = 0.1
    gamma = 0.925
    max_episode_steps = 35

    w = get_weights(env=env,
                    automata=[dfa1, dfa2], epsilon=epsilon, dist=True, discount_factor=gamma,
                    max_iterations=max_episode_steps, priority=[2, 1, 0])

    dfa1.reward = w[1]
    dfa2.reward = w[2]

    a = ONRBAgent1(env=env, dfa_list=[dfa1, dfa2], ntrain=100000, epsilon=0.25, alpha=0.1, gamma=0.925, max_episodes_steps=35)
    a.train(save="[ONRB-learner] conflicting_norms")
    a.evaluate()
