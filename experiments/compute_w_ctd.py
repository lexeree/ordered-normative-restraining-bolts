from environment import Environment
from automaton import DFA
from weight_computation import get_weights

if __name__ == "__main__":
    env = Environment('../layouts/basic.txt', risk=1, sunset=30, timeout=50)
    epsilon = 2
    discount_factor = 0.9
    max_iterations = 50
    automata = []

    priority = [2, 1, 0]


    # "If the merchant is currently at home, it ought to visit the market before sunset"
    # viol: !at_market U sundown
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
    automatonb = DFA([0, 1], 1, trans=transb, final=[1])

    automata.append(automatona)
    automata.append(automatonb)

    w_E = get_weights(env, automata, epsilon, dist=True, discount_factor=discount_factor,
                                       max_iterations=max_iterations, priority=priority)

    print("Ethical weight: ", w_E, flush=True)