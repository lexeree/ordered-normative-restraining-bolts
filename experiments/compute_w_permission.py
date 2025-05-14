from environment import Environment
from automaton import DFA
from weight_computation import get_weights

if __name__ == "__main__":
    env = Environment('../layouts/basic.txt', risk=1, sunset=20, timeout=50)
    epsilon = 0.1
    discount_factor = 0.9
    max_iterations = 50
    automata = []

    priority = [1, 0]

    def trans0a(l):
        if 'at_tree' in l and 'Extract' in l:
            return 1
        else:
            return 0


    def trans1a(l):
        return 1


    transa = {}
    transa[0] = trans0a
    transa[1] = trans1a
    automatona = DFA([0, 1], 1, trans=transa, final=[1])
    automata.append(automatona)

    w_E = get_weights(env, automata, epsilon, dist=True, discount_factor=discount_factor,max_iterations=max_iterations, priority=priority)

    print("Ethical weight: ", w_E, flush=True)