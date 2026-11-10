import optuna

class Optuna:
    def __init__(self, train_test_func):
        self.train_test_func = train_test_func

    def objective(self, trial: optuna.Trial,
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

        mean_score, std_score = self.train_test_func(
            scenario=scenario,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            training_steps=training_steps,
            ep_length=episode_length
        )

        trial.set_user_attr("reward_std", std_score)
        return mean_score

    def run_optuna(self,
            db_name: str,
            scenario: str,
            trial: int,
            alpha_interval: list[float],
            gamma_interval: list[float],
            epsilon_interval: list[float],
            training_step_list: list[int],
            episode_length_list: list[int]
    ):
        study = optuna.create_study(
            study_name= str(scenario) + "_" + str(trial),
            direction="maximize",
            storage="sqlite:///" + db_name + ".db",
            load_if_exists=True,
        )

        study.optimize(
            lambda trial: self.objective(trial,
                                    scenario,
                                    alpha_interval,
                                    gamma_interval,
                                    epsilon_interval,
                                    training_step_list,
                                    episode_length_list),
            n_trials=100)

        print("Best reward:", study.best_value)

        print("Best hyperparameters:", study.best_params)