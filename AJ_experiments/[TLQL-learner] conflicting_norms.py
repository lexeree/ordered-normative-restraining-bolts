from merchant import MerchantEnv
from agents import ONRBAgent2
from dfas.dfas import DeliveryDFA, DangerDFA
import numpy as np
import random
from run_optuna import Optuna

def tune_optuna(
    scenario: str,
    alpha: float,
    gamma: float,
    epsilon: float,
    training_steps : int,
    ep_length: int
) -> tuple[float,float]:

    seed_scores = []

    for seed in [0, 1, 2]:
        random.seed(seed)
        np.random.seed(seed)

        env = MerchantEnv(layout=scenario, capacity=6, sunset=32, risk_fight=1)
        dfa1 = DeliveryDFA(reward=1)
        dfa2 = DangerDFA(reward=1)

        a = ONRBAgent2(env=env, dfa_list=[dfa1, dfa2], ntrain=training_steps, epsilon=epsilon, alpha=alpha, gamma=gamma,
                       max_episodes_steps=ep_length)
        a.train()

        rewards = a.evaluate(runs=20)

        seed_scores.append(np.mean(rewards))

    return float(np.mean(seed_scores)), float(np.std(seed_scores))

def run_optuna():
    opt = Optuna(train_test_func=tune_optuna)
    opt.run_optuna(db_name="[TLQL-learner] conflicting_norms",
                   scenario = "basic",
                   trial=1,
                   alpha_interval=[0.005, 0.3],
                   gamma_interval=[0.90, 0.999],
                   epsilon_interval=[0.05, 0.4],
                   training_step_list=[100000],
                   episode_length_list=[30, 32, 35]
                   )

def train_and_evaluate_tuned():
    np.random.seed(7)
    random.seed(7)

    env = MerchantEnv(layout='basic', capacity=6, sunset=32, risk_fight=1)

    dfa1 = DeliveryDFA(reward=1)
    dfa2 = DangerDFA(reward=1)

    a = ONRBAgent2(env=env, dfa_list=[dfa1, dfa2], ntrain=100000, epsilon=0.25, alpha=0.1, gamma=0.9,
                   max_episodes_steps=35)
    a.train(save="[TLQL-learner] conflicting_norms")
    a.evaluate()

if __name__ == "__main__":
    run_optuna()
