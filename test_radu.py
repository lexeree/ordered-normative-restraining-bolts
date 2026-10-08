from merchant import MerchantEnv, labels, actions
from agents import QLearner, ONRBAgent1
import optuna
import numpy as np
import random
import argparse
from dfas.dfas import OreAfterTreeDFA, NoExtractWoodIfAlreadyExtractedDFA, NoExtractOreDFA
from weight_computation import get_weights

CAPACITY = 100

def parse_args():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # -------------------------
    # Run Optuna
    # -------------------------
    optuna_parser = subparsers.add_parser("optuna")

    optuna_parser.add_argument("--scenario", type=str, required=True)
    optuna_parser.add_argument("--trial", type=int, required=True)

    optuna_parser.add_argument(
        "--alpha-interval",
        type=float,
        nargs=2,
        required=True
    )

    optuna_parser.add_argument(
        "--gamma-interval",
        type=float,
        nargs=2,
        required=True
    )

    optuna_parser.add_argument(
        "--epsilon-interval",
        type=float,
        nargs=2,
        required=True
    )

    optuna_parser.add_argument(
        "--training-steps",
        type=int,
        nargs="+",
        required=True
    )

    optuna_parser.add_argument(
        "--episode-lengths",
        type=int,
        nargs="+",
        required=True
    )

    # -------------------------
    # Test Optuna parameters
    # -------------------------
    test_parser = subparsers.add_parser("test")

    test_parser.add_argument("--agent", type=str, required=True)
    test_parser.add_argument("--scenario", type=str, required=True)
    test_parser.add_argument("--ntrain", type=int, required=True)
    test_parser.add_argument("--epsilon", type=float, required=True)
    test_parser.add_argument("--alpha", type=float, required=True)
    test_parser.add_argument("--gamma", type=float, required=True)
    test_parser.add_argument(
        "--max-episode-steps",
        type=int,
        required=True
    )

    return parser.parse_args()

def train_and_evaluate_agent(
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

        env = MerchantEnv(layout=scenario, capacity=CAPACITY)

        agent = QLearner(env, ntrain=training_steps, epsilon=epsilon, alpha=alpha, gamma=gamma, max_episode_steps=ep_length)

        agent.train()

        rewards = agent.evaluate(runs=20)

        seed_scores.append(np.mean(rewards))

    return float(np.mean(seed_scores)), float(np.std(seed_scores))

def objective(trial: optuna.Trial,
              scenario: str,
              alpha_interval: list[float],
              gamma_interval: list[float],
              epsilon_interval: list[float],
              training_step_list: list[int],
              episode_length_list: list[int],
              ) -> float:
    alpha = trial.suggest_float(
        "alpha", alpha_interval[0], alpha_interval[1]
    )
    gamma = trial.suggest_float(
        "gamma", gamma_interval[0], gamma_interval[1]
    )

    epsilon = trial.suggest_float(
        "epsilon", epsilon_interval[0], epsilon_interval[1]
    )

    training_steps = trial.suggest_categorical(
        "training_steps",
        training_step_list
    )

    episode_length = trial.suggest_categorical(
        "episode_length",
        episode_length_list
    )

    mean_score, std_score = train_and_evaluate_agent(
        scenario=scenario,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        training_steps=training_steps,
        ep_length=episode_length
    )

    trial.set_user_attr("reward_std", std_score)
    return mean_score


def decode_state(state):
    s = state[0]
    return f'x={s[0]}, y={s[1]}, label={labels[s[2]]}, coll_wood={s[3]}, coll_ore={s[4]}, sundown={s[5]}, last_action={"-" if s[6] == 0 else actions[s[6]]}'


def split_key(key):
    state, a, b, c = key
    return list(state) + [a, b, c]

def run_optuna(
        scenario: str,
        trail: int,
        alpha_interval: list[float],
        gamma_interval: list[float],
        epsilon_interval: list[float],
        training_step_list: list[int],
        episode_length_list: list[int]
):
    study = optuna.create_study(
        study_name= str(scenario) + "_" + str(trail),
        direction="maximize",
        storage="sqlite:///optuna_new.db",
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial,
                                scenario,
                                alpha_interval,
                                gamma_interval,
                                epsilon_interval,
                                training_step_list,
                                episode_length_list),
        n_trials=100)

    print("Best reward:", study.best_value)

    print("Best hyperparameters:", study.best_params)

def test_q_learner(scenario, ntrain, epsilon, alpha, gamma, max_episode_steps):

    env = MerchantEnv(layout=scenario, capacity=CAPACITY)

    a = QLearner(env, ntrain=ntrain, epsilon=epsilon, alpha=alpha, gamma=gamma, max_episode_steps=max_episode_steps)

    a.train(save="model1")

    a.evaluate()


def test_onrb_learner_norm_change(ntrain, epsilon, alpha, gamma, max_episode_steps):
    # Obliged to extract wood before extracting ore
    #dfa1 = OreAfterTreeDFA(reward=1)

    # Forbidden to collect wood after already collecting wood
    dfa2 = NoExtractWoodIfAlreadyExtractedDFA(reward=1)

    # After some time: Forbidden to collect ore
    #dfa3 = NoExtractOreDFA(reward=1)

    def revise(step):
            if step > 15:
                return [False, True, True]
            else:
                return [True, True, True]

    
    env = MerchantEnv(layout='normchange', capacity=CAPACITY)

    w = get_weights(env=env, 
                automata= [dfa2], epsilon=epsilon, dist=True, discount_factor=gamma,
                                       max_iterations=max_episode_steps, priority=[1, 0])

    print('Weights ', w)

    #a = ONRBAgent1(env=env, dfa_list=[dfa2], ntrain=ntrain, gamma=gamma, alpha=alpha, epsilon=epsilon, max_episodes_steps=max_episode_steps)

    #a.train(save="model1")

    #a.evaluate()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "optuna":
        run_optuna(
            scenario=args.scenario,
            trail=args.trial,
            alpha_interval=args.alpha_interval,
            gamma_interval=args.gamma_interval,
            epsilon_interval=args.epsilon_interval,
            training_step_list=args.training_steps,
            episode_length_list=args.episode_lengths
        )

    elif args.mode == "test":
        if args.agent == 'Q_learner':
            test_q_learner(
                scenario=args.scenario,
                ntrain=args.ntrain,
                epsilon=args.epsilon,
                alpha=args.alpha,
                gamma=args.gamma,
                max_episode_steps=args.max_episode_steps
            )
        elif args.agent == 'ONRB_scalarized':
            if args.scenario == "normchange":
                test_onrb_learner_norm_change(
                    ntrain=args.ntrain,
                    epsilon=args.epsilon,
                    alpha=args.alpha,
                    gamma=args.gamma,
                    max_episode_steps=args.max_episode_steps
                )

