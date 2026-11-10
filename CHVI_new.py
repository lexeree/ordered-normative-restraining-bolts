import numpy as np
from dfas.dfas import NoExtractWoodIfAlreadyExtractedDFA
import convexhull
from merchant import MerchantEnv, decode_action, decode_state
from tqdm import tqdm


def create_state_dict(env, automata, dist):
    dict_list = []
    s, _ = env.reset()
    initial_state = (s,)
    pbar = tqdm(desc="Generating states for CHVI")

    # create extended state
    for i in range(0, len(automata)):
        initial_state = initial_state + (automata[i].state0,)

    # iterative search with a set for the visited check (a list lookup is O(n) and recursion overflows the stack)
    visited = {initial_state}
    dict_list.append((initial_state, []))
    stack = [initial_state]

    while stack:
        state = stack.pop()
        pbar.update(1)
        for nstate in next_states(state, env, automata):
            if nstate not in visited:
                visited.add(nstate)
                dict_list.append((nstate, []))
                stack.append(nstate)

    pbar.close()
    return dict_list


def next_states(state, env, automata):
    # the environment state is fully encoded in state[0], so resetting to it replaces a deepcopy of the env
    env.reset(options={"state": state[0]})
    possible_actions = list(set(range(env.action_space.n)) - set(env.exclActions()))
    curr_state_labels = env.get_labels().copy()
    result = []

    for action in possible_actions:
        decoded_action = decode_action(action)

        # ignore states fulfilling the conditions below to avoid infinite recursion
        if "attack" in curr_state_labels and (
                decoded_action == 'south' or decoded_action == 'north' or decoded_action == 'west' or decoded_action == 'east'):
            continue

        env.reset(options={"state": state[0]})
        inpt = list(set(curr_state_labels + [decoded_action]))
        next_state, _, terminal, _, _ = env.step(action)

        # terminal transitions have no successor state, their value is computed directly in the Q function
        if terminal:
            continue

        trans = tuple()

        for i in range(len(automata)):
            astate = state[i + 1]
            if astate in automata[i].final:
                astate = automata[i].state0
            nst = automata[i].transition(inpt, astate)
            trans = trans + (nst,)

        result.append((next_state,) + trans)

    return result

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

        print(f"Discovered states: {len(dic)}")
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
        # reset before every action, so that a skipped action cannot leave the env in a wrong state
        env.reset(options={"state": s})
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

        # a terminal action has no future value, but the other actions of this state still have to be considered
        if is_terminal:
            hulls.append(np.array((reward_objective,) + rwds).astype(float))
            continue

        sts = (next_state,) + curr_astates
        if sts not in V_state_dict:
            continue
        V_state = V_state_dict[sts].copy()
        hull_sa = convexhull.translate_hull(np.array((reward_objective,) + rwds).astype(float), discount_factor,
                                            V_state)
        for point in hull_sa:
            hulls.append(point)

    if len(hulls) == 0:
        return []

    hulls = np.unique(np.array(hulls), axis=0)
    new_hull = convexhull.get_hull(hulls)
    return new_hull

if __name__ == "__main__":

    env = MerchantEnv(layout="normchange", sunset=0)

    dfa = NoExtractWoodIfAlreadyExtractedDFA(reward=1)

    hull = convex_hull_value_iteration(env=env,automata=[dfa],dist=False,discount_factor=0.99,max_iterations=25)

    for k in hull.keys():
        if k[0][6] == 7:
            print(len(hull[k]))
            print(decode_state(k[0]), hull[k])
