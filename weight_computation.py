import numpy as np
#import pandas as pd
from mip import Model, xsum, maximize, minimize
from CHVI import convex_hull_value_iteration
from agents import Agent
from automaton import DFA
from environment import Environment
from state import State

def ethical_embedding_all_states(hull, priority, dist, epsilon):
    weights = {}

    for key in hull.keys():
        ag = Agent(env, dist)
        hull_state = hull.get(key)

        if len(ag.getPossibleActions(key[0])) > 1:
            best_ethical_indexes = list()
            best_ethical_index = lex_max(hull_state, priority, iWantIndex=True)
            best_ethical_indexes.append(best_ethical_index)
            w = minimal_weight_computation_all_states(hull, best_ethical_indexes, [key], epsilon)
            weights[key] = w
    return weights


def ethical_embedding_start_state(hull, priority, initial_states, epsilon):
    best_ethical_indexes = list()

    for state in initial_states:
        hull_state = hull.get(state)
        best_ethical_index = lex_max(hull_state, priority, iWantIndex=True)
        best_ethical_indexes.append(best_ethical_index)

    w = minimal_weight_computation_all_states(hull, best_ethical_indexes, initial_states, epsilon)
    return w

def minimal_weight_computation_all_states(hull, chosen_indexes, chosen_states=None, epsilon=0.0):
    chosen_hulls = list()

    if chosen_states is not None:

        # define size of the policies in the convex hull
        I = range(len(hull.get(chosen_states[0])[0]))

        # size of optimal policies found in the chosen states
        J = range(len(chosen_indexes))

        for state in chosen_states:
            chosen_hulls.append(hull.get(state))

        m = Model("knapsack")
        w = [m.add_var() for _ in I]

        # we minimize the weights
        m.objective = minimize(xsum(w[i] for i in I))

        for k in range(len(chosen_indexes)):
            hull_state = hull.get(chosen_states[k])

            optimal_policy = hull_state[chosen_indexes[k]]

            for j in range(len(hull_state)):
                if j != chosen_indexes[k]:
                    if hull_state[j][0] >= optimal_policy[0]:
                        m += xsum(w[i] * hull_state[j][i] for i in I) + epsilon <= xsum(w[i] * optimal_policy[i] for i in I)

        for i in I:
            m += w[i] >= 0

        m += w[0] == 1

        m.verbose = 0
        m.optimize()
        minimal_weights = [w[i].x for i in I]

        return minimal_weights
    else:
        return [-1]

# Retrieve the optimal policy based on the given lexicographic ordering
def lex_max(hull, lex=None, iWantIndex=False):

    # standard lexicographic ordering: individual objective > norm 1 > norm 2 > ...
    if lex == None:
        lex = [i for i in range(len(hull[0]))]

    hull = np.array(hull)

    ordered_hull = hull[:, lex]
    antilex = reverse_lex(lex)

    for i in range(len(lex)):
        first_max = np.max(ordered_hull, axis=0)[i]
        next_hull = []

        for j in ordered_hull:
            if j[i] >= first_max:
                next_hull.append(j)

        ordered_hull = next_hull

    lex_point = np.array(ordered_hull[0])
    lex_point = lex_point[antilex]

    if iWantIndex:
        for i in range(len(hull)):
            its_the_same = True
            for j in range(len(lex_point)):
                its_the_same *= lex_point[j] == hull[i][j]

            if its_the_same:
                return i

    return lex_point

def reverse_lex(lex):
    def find_n(n, lex):
        for i in range(len(lex)):
            if n == lex[i]:
                return i

    return [find_n(i, lex) for i in range(len(lex))]

def print_dict(weights):
    for key in weights.keys():
        if isinstance(weights[key], dict):
            for state in weights[key].keys():
                print(state[0].sPrint(), state[1:], ":", weights[key][state])
        else:
            print(key[0].sPrint(), key[1:], ":", weights[key])

def get_weights(env, automata, epsilon, dist=False, discount_factor=1.0, max_iterations=5,
                                 priority=None):
    # import pre-computed convex hull or compute it
    hull = convex_hull_value_iteration(env, automata, dist, discount_factor, max_iterations)

    # get the initial states (in our case just one)
    initial_states = [tuple([env.initialState()] + [x.state0 for x in automata])]

    # compute the ethical weights
    ethical_weight = ethical_embedding_start_state(hull, priority, initial_states, epsilon)

    #ethical_weights_all_states = ethical_embedding_all_states(hull, priority, dist, epsilon)
    #ethical_weights_separated = ethical_embedding_all_states_split(hull,priority,dist,epsilon)
    #print("Separated weights: \n")
    #print_dict(ethical_weights_separated)
    #print("Weights by state: \n")
    #print_dict(ethical_weights_all_states)

    return ethical_weight


if __name__ == "__main__":
    env = Environment('layouts/basic.txt', risk=1, sunset=20, timeout=50)
    epsilon = 1
    discount_factor = 0.9
    max_iterations = 50
    automata = []

    priority = [1, 2 , 0]


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
    automatonc = DFA([0, 1, 2], 1, trans=transa, final=[2], achievement=True)

    automata.append(automatona)
    automata.append(automatonb)
    #automata.append(automatonc)

    def trans0d(l):
        if 'at_tree' in l and 'Extract' in l:
            return 1
        else:
            return 0


    def trans1d(l):
        return 1


    transd = {}
    transd[0] = trans0d
    transd[1] = trans1d
    automatond = DFA([0, 1], 150, trans=transd, final=[1])
    #automata.append(automatond)

    w_E = get_weights(env, automata, epsilon, dist=True, discount_factor=discount_factor,
                                       max_iterations=max_iterations, priority=priority)

    print("Ethical weight: ", w_E, flush=True)
