from environment import Environment
import qlearning as q
import qlLTL as rb
import epsilon_decay as rbd
from automaton import DFA
import argparse
from merchant import MerchantEnv
from agents import QLearner, RBAgent
from dfas.dfas import DangerDFA, PassiveDFA, DeliveryDFA

if __name__ == '__main__':
    env = MerchantEnv()

    #a = QLearner(env, ntrain=500000, epsilon=0.25)

    dfa1 = DeliveryDFA(1000.0)
    dfa2 = DangerDFA(25.0)
    dfa3 = PassiveDFA(200.0)

    a = RBAgent(env, dfa_list=[dfa1, dfa2, dfa3], ntrain=1000000, epsilon=0.3)

    a.train()
    #print(a.qvalues)
    a.evaluate()
 