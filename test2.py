from environment import Environment
import qlearning as q
import qlLTL as rb
import epsilon_decay as rbd
from automaton import DFA
import argparse
from merchant import MerchantEnv, labels, actions
from agents import QLearner, RBAgent, ONRBAgent1, ONRBAgent2
from dfas.dfas import DangerDFA, PassiveDFA, DeliveryDFA
import pickle
import pandas as pd
import ast

def decode_state(state):
    s = state[0]
    return f'x={s[0]}, y={s[1]}, label={labels[s[2]]}, coll_wood={s[3]}, coll_ore={s[4]}, sundown={s[5]}, last_action={"-" if s[6] == 0 else actions[s[6]]}'

def split_key(key):
    state, a, b, c = key
    return list(state) + [a, b, c]

if __name__ == '__main__':
    env = MerchantEnv()

    #a = QLearner(env, ntrain=100000, epsilon=0.3, gamma=0.99)

    dfa1 = DeliveryDFA(500.0)
    dfa2 = DangerDFA(50.0)
    dfa3 = PassiveDFA(350.0)

    #a = RBAgent(env, dfa_list=[dfa1, dfa2, dfa3], ntrain=200000, epsilon=0.5, gamma=0.999)

    a = ONRBAgent1(env, dfa_list=[dfa1, dfa2], ntrain=1000000, epsilon=0.1, gamma=0.999)

    a.train(save="model1")
    #init_state, _ = env.reset()
    #init_state = (init_state, 0, 0, 0)
    #print(list(zip(actions, a.qvalues[init_state])))
    a.evaluate()

    #with open("radu_test.p", "rb") as f:
    #    data = pickle.load(f)

    #df = pd.DataFrame.from_dict(data, orient="index")

    # Move index (keys) into a column
    #df.reset_index(inplace=True)

    # Rename column
    #df.rename(columns={"index": "id"}, inplace=True)
    #expanded = df["id"].apply(split_key).apply(pd.Series)
    #expanded = expanded.drop(expanded.columns[7:13], axis=1)
    #expanded = pd.concat(
    #    [expanded, df.reset_index(drop=True).iloc[:, 1:]],
    #    axis=1
    #)
    #expanded.columns = [
    #    "x",
    #    "y",
    #    "label",
    #    "wood_collected",
    #    "ore_collected",
    #    "sundown",
    #    "last_action",
    #    "aut_1",
    #    "aut_2",
    #    "aut_3",
    #] + actions
    #expanded["last_action"] = expanded["last_action"].map(lambda x: "No act" if x == 7 else actions[x])
    #expanded["label"] = expanded["label"].map(lambda x: labels[x])
    # Save to CSV
    #expanded.to_csv("output.csv", index=False)

 