from environment import Environment
import epsilon_decay as rbd
from automaton import DFA

if __name__ == '__main__':
    env = Environment('../layouts/basic.txt', risk=1)
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
    automatona = DFA([0,1], 2.9, trans=transa, final=[1])

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
    automatonb = DFA([0, 1], 67.7, trans=transb, final=[1])

    automata.append(automatona)
    automata.append(automatonb)
    agent = rbd.DecayedMORLRBNormAgent(env, dist=True, specs=automata, alpha=0.5, epsilon_init=0.5, epsilon_end=0.01,
                                       gamma=0.9, ntrain=15000, plot=True, seed='hi')
    agent.train()
    a, i, b, c = agent.test(rec=True)
