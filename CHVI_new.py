import numpy as np

import convexhull
from agents import Agent
from dfas.dfas import DangerDFA
from merchant import MerchantEnv, decode_action, decode_state
import copy
from tqdm import tqdm


def create_state_dict(env, automata, dist):
    ag = Agent(env, dist)
    dict_list = []
    s, _ = env.reset()
    initial_state = (s,)
    pbar = tqdm(desc="Generating states for CHVI")

    # create extended state
    for i in range(0, len(automata)):
        initial_state = initial_state + (automata[i].state0,)

    dict_list.append((initial_state, []))
    add_next_state(initial_state, env, ag, automata, dict_list, 0, False, pbar)
    return dict_list


def add_next_state(state, env, agent, automata, dict_list, depth, is_terminal, pbar):
    pbar.update(1)

    # stop the recursion if no action is possible or if the state is terminal
    possible_actions = list(set(range(env.action_space.n)) - set(env.exclActions()))

    if len(possible_actions) == 0 or is_terminal:
        return

    curr_state_labels = env.get_labels().copy()

    for action in possible_actions:
        decoded_action = decode_action(action)

        # ignore states fulfilling the conditions below to avoid infinite recursion
        if "attack" in curr_state_labels and (
                decoded_action == 'south' or decoded_action == 'north' or decoded_action == 'west' or decoded_action == 'east'):
            continue

        env_new = copy.deepcopy(env)
        inpt = list(set(curr_state_labels + [decoded_action]))
        next_state, _, terminal, _, _ = env_new.step(action)

        trans = tuple()

        for i in range(len(automata)):
            astate = state[i + 1]
            if astate in automata[i].final:
                astate = automata[i].state0
            nst = automata[i].transition(inpt, astate)
            trans = trans + (nst,)

        nstate = (next_state,) + trans

        if (nstate, []) not in dict_list:
            dict_list.append((nstate, []))
            add_next_state(nstate, env_new, agent, automata, dict_list, depth + 1, terminal, pbar)

def convex_hull_value_iteration(env, automata, dist=False, discount_factor=1.0, max_iterations=5, model_used=None):
    """
       Convex Hull Value Iteration algorithm adapted from "Convex Hull Value Iteration" from
       Barret and Narananyan's 'Learning All Optimal Policies with Multiple Criteria' (2008)

       Calculates the convex hull for each state of the MOMDP

       :param env: the environment encoding the MOMDP
       :param discount_factor: discount factor of the environment, to be set at discretion
       :param max_iterations: convergence parameter, the more iterations the more probabilities of precise result
       :return: value function storing the partial convex hull for each state
       """
    dict_list = create_state_dict(env, automata, dist)
    iteration = 0
    dic = dict(dict_list)
    pbar = tqdm(total=max_iterations * len(dic.keys()), desc="CHVI algorithm running")

    while iteration < max_iterations:
        for key in dic.keys():
            pbar.update(1)
            dic[key] = Q_function_calculator(env, key, dic, automata, dist, discount_factor, model_used)
        iteration += 1
    return dic

def Q_function_calculator(env, state, V_state_dict, automata, dist=False, discount_factor=1.0, model_used=None):
    """
        Calculates the (convex hull)-value of applying each action to a given state.
        Heavily adapted to the merchant game

        :param env: the environment of the Markov Decision Process
        :param state: the current state
        :param V: value function to see the value of the next state V(s')
        :param dist:
        :param discount_factor: discount factor considered, a real number
        :return: the new convex obtained after checking for each action (this is the operation hull of unions)
    """
    hulls = list()
    s = state[0]
    env.reset(options={"state": s})
    possible_actions = list(set(range(env.action_space.n)) - set(env.exclActions()))

    if len(possible_actions) == 0:
        return []

    for action in possible_actions:
        decoded_action = decode_action(action)
        curr_state_labels = env.get_labels().copy()
        curr_inpt = list(set(curr_state_labels + [decoded_action]))
        next_state, reward_objective, is_terminal, _, _ = env.step(action)

        astates = state[1:]
        rwds = tuple()
        curr_astates = tuple()

        for i in range(len(automata)):
            if astates[i] in automata[i].final:
                curr_astates = curr_astates + (automata[i].transition(curr_inpt, automata[i].state0),)
            else:
                curr_astates = curr_astates + (automata[i].transition(curr_inpt, astates[i]),)

        for i in range(len(automata)):
            if curr_astates[i] in automata[i].final:
                rwds = rwds + (-1,)
            else:
                rwds = rwds + (0,)

        if is_terminal:
            return [list((reward_objective,) + rwds)]

        sts = (next_state,) + curr_astates
        try:
            V_state = V_state_dict[sts].copy()
        except:
            continue
        hull_sa = convexhull.translate_hull(np.array((reward_objective,) + rwds).astype(float), discount_factor,
                                            V_state)
        for point in hull_sa:
            hulls.append(point)

        env.reset(options={"state": s})

    hulls = np.unique(np.array(hulls), axis=0)
    new_hull = convexhull.get_hull(hulls)
    return new_hull

if __name__ == '__main__':
    env = MerchantEnv(layout="test", sunset=0)

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
    automatona = DangerDFA(dict(labels=['atDanger']), 1)
    hull = convex_hull_value_iteration(env, [automatona], False, 1, 25)

    for k in hull.keys():
        if k[0][6] == 7:
            print(len(hull[k]))
            print(decode_state(k[0]), hull[k])
