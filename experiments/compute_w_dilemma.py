from environment import Environment
from automaton import DFA
from weight_computation import get_weights

if __name__ == "__main__":
    env = Environment('../layouts/basic.txt', risk=1, sunset=30, timeout=35)
    epsilon = 1
    discount_factor = 0.9
    max_iterations = 35
    automata = []

    priority = [2, 1, 0]

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
    automatona = DFA([0, 1], 1, trans=transa, final=[1])


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
    automatonc = DFA([0, 1, 2], 1, trans=transa, final=[2], achievement=True)

    automata.append(automatona)
    automata.append(automatonc)

    w_E = get_weights(env, automata, epsilon, dist=False, discount_factor=discount_factor,
                                       max_iterations=max_iterations, priority=priority)

    print("Ethical weight: ", w_E, flush=True)