import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
) -> tuple[np.ndarray, np.ndarray]:
    """Returns per-component (mean, std) across seeds.
    Components: [-delivery_violations, -danger_violations, env_reward], all to be maximized."""

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

        rewards = np.asarray(a.evaluate(runs=20, record=False))  # shape (runs, n_components)

        seed_scores.append(rewards.mean(axis=0))  # per-component mean over runs

    seed_scores = np.asarray(seed_scores)  # shape (n_seeds, n_components)
    return seed_scores.mean(axis=0), seed_scores.std(axis=0)

def run_optuna():
    opt = Optuna(train_test_func=tune_optuna)
    opt.run_optuna(db_name="[TLQL-learner] conflicting_norms",
                   scenario = "basic",
                   trial=2,
                   alpha_interval = [0.005, 0.3],
                   gamma_interval = [0.97, 0.999],
                   epsilon_interval = [0.1, 0.5],
                   training_step_list = [100000, 500000, 1000000],
                   episode_length_list = [50, 75, 100],
                   n_objectives=3
                   )

def train_and_evaluate_tuned():
    np.random.seed(7)
    random.seed(7)

    env = MerchantEnv(layout='basic', capacity=6, sunset=32, risk_fight=1)

    dfa1 = DeliveryDFA(reward=1)
    dfa2 = DangerDFA(reward=1)
    dfas= [dfa1, dfa2]

    a = ONRBAgent2(env=env, dfa_list=dfas, ntrain=100000, epsilon=0.2851, alpha=0.2127, gamma=0.9747,
                   max_episodes_steps=30)
    a.train(save="[TLQL-learner] conflicting_norms")
    a.evaluate()

if __name__ == "__main__":
    run_optuna()
