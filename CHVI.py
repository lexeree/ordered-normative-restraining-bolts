import numpy as np
import convexhull as convexhull
from agents import Agent


def create_state_dict(env, automata, dist):
    # get all possible arguments for state objects
    ag = Agent(env, dist)
    # dictionary of all possible states
    dict_list = []
    initial_state = (env.initialState(),)
    for i in range(0, len(automata)):
        initial_state = initial_state + (0,)
    dict_list.append((initial_state, []))
    add_next_state(initial_state, env, ag, automata, dict_list, 0)
    return dict_list


def add_next_state(state, env, agent, automata, dict_list, depth):
    if len(agent.getPossibleActions(state[0])) == 0 or (state[0].label == "M" and not state[0].inventory):
        return
    for action in agent.getPossibleActions(state[0]):
        # ignore such states as the next state will be the previous state and will lead to infinite
        # recursion
        if state[0].attack == True and (
                action == 'South' or action == 'North' or action == 'West' or action == 'East'):
            continue
        env_new = env.copy()
        # it's important to compute the labels first
        inpt = agent.get_labels(state[0].copy()) + [action]
        next_state, _ = env_new.stateTransition(state=state[0], action=action)

        trans = tuple()
        for i in range(len(automata)):
            astate = state[i + 1]
            if astate in automata[i].final:
                astate = automata[i].state0
            nst = automata[i].transition(astate, inpt)
            trans = trans + (nst,)

        nstate = (next_state,) + trans
        if (nstate, []) not in dict_list:
            dict_list.append((nstate, []))
            add_next_state(nstate, env_new, agent, automata, dict_list, depth + 1)


#def extend_df(df, dic, iteration_string):
#    df[iteration_string] = df.index.map(dic)


#def df_to_excel(df, name):
#    df.to_excel(name, index=True)


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

    # dictionary of all possible states
    dict_list = create_state_dict(env, automata, dist)
    print("Number of States:", len(dict_list), flush=True)
    iteration = 0
    dic = dict(dict_list)

    while iteration < max_iterations:
        iteration += 1
        print("Iteration:", iteration, "-------")
        count = 0
        for key in dic.keys():
            count = count + 1
            dic[key] = Q_function_calculator(env, key, dic, automata, dist, discount_factor, model_used)
        print("Values s0:", list(dic.values())[0], flush=True)
    return dic

# TODO: find out what model_used does
def Q_function_calculator(env, states, V_state_dict, automata, dist=False, discount_factor=1.0, model_used=None):
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
    ag = Agent(env, dist)
    hulls = list()
    state = states[0]
    possible = ag.getPossibleActions(state)
    for action in possible:
        env.reset()
        curr_inpt = ag.get_labels(state) + [action]
        next_state, reward_objective = env.stateTransition(state=state, action=action)
        astates = states[1:]
        rwds = tuple()
        curr_astates = tuple()

        for i in range(len(automata)):
            if astates[i] in automata[i].final:
                curr_astates = curr_astates + (automata[i].transition(automata[i].state0, curr_inpt),)
            else:
                curr_astates = curr_astates + (automata[i].transition(astates[i], curr_inpt),)
        for i in range(len(automata)):
            if curr_astates[i] in automata[i].final:
                rwds = rwds + (-1,)
            else:
                rwds = rwds + (0,)

        # The transition function subtracts from the damage even if it is below 0, which will lead to
        # 'incorrect states' not existent in the dictionary. Thus, the following block is used to skip
        # those cases
        # TODO: adapt the environment for such situation
        sts = (next_state,) + curr_astates
        try:
            V_state = V_state_dict[sts].copy()
        except:
            continue
        hull_sa = convexhull.translate_hull(np.array((reward_objective,) + rwds).astype(float), discount_factor,
                                            V_state)
        for point in hull_sa:
            hulls.append(point)
    hulls = np.unique(np.array(hulls), axis=0)
    new_hull = convexhull.get_hull(hulls)
    return new_hull