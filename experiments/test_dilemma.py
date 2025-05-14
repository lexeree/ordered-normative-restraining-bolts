from environment import Environment
import qlearning as q
import qlLTL as rb
import epsilon_decay as rbd
from automaton import DFA
import argparse

if __name__ == '__main__':
    env = Environment('../layouts/basic.txt', risk=1, sunset=30, timeout=35)
    automata = []

    # "The merchant is forbidden from entering the dangerous area"
    # viol: F(at_danger)
    def trans0a(l):
        if 'at_danger' in l:
            return 1
        else:
            return 0
    def trans1a(l):
        return 1
    transa = {}
    transa[0] = trans0a
    transa[1] = trans1a
    automatona = DFA([0,1], 15.5, trans=transa, final=[1])

    # "if the merchant enters the dangerous area, they ought to unload their inventory"
    # viol: F(at_danger & attacked & !(Unload))
    def trans0b(l):
        if 'at_danger' in l and 'attacked' in l and 'Unload' not in l:
            return 1
        else:
            return 0
    def trans1b(l):
        return 1

    transb = {}
    transb[0] = trans0b
    transb[1] = trans1b
    automatonb = DFA([0, 1], 45, trans=transb, final=[1])


    # "The merchant ought to visit the market before sunset"
    # viol: !at_market U sundown
    def trans0c(l):
        if 'sundown' in l and 'at_home' in l:
            return 2
        elif 'at_home' in l and 'sundown' not in l and 'at_market' not in l:
            return 1
        else:
            return 0


    def trans1c(l):
        if 'sundown' in l:
            return 2
        elif 'at_market' not in l:
            return 1
        else:
            return 0


    def trans2c(l):
        return 2


    transa = {}
    transa[0] = trans0c
    transa[1] = trans1c
    transa[2] = trans2c
    automatonc = DFA([0, 1, 2], 57.5, trans=transa, final=[2], achievement=True)

    automata.append(automatonc)
    automata.append(automatona)
    agent = rb.MORLRBNormAgent(env, specs=automata, alpha=0.5, epsilon=0.25, gamma=0.9, ntrain=10000, seed='hi')
    agent.train()
    a, i, b, c = agent.test(rec=True)
