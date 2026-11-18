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
                  ) -> tuple[float, ...]:

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

        mean_vec, std_vec = self.train_test_func(
            scenario=scenario,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            training_steps=training_steps,
            ep_length=episode_length
        )

        # store as plain lists: numpy arrays are not JSON-serializable for the SQLite storage
        mean_list = [float(x) for x in mean_vec]
        std_list = [float(x) for x in std_vec]
        trial.set_user_attr("reward_mean", mean_list)
        trial.set_user_attr("reward_std", std_list)

        # one objective value per reward component (multi-objective study)
        return tuple(mean_list)

    def run_optuna(self,
            db_name: str,
            scenario: str,
            trial: int,
            alpha_interval: list[float],
            gamma_interval: list[float],
            epsilon_interval: list[float],
            training_step_list: list[int],
            episode_length_list: list[int],
            n_objectives: int
    ):
        # multi-objective study: every reward component is maximized
        # ("_mo" suffix avoids clashing with older single-objective studies in the same db)
        study = optuna.create_study(
            study_name= str(scenario) + "_" + str(trial) + "_mo",
            directions=["maximize"] * n_objectives,
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

        # Pareto front: all non-dominated trials
        pareto = study.best_trials
        print(f"Pareto front ({len(pareto)} trials):")
        for t in pareto:
            print(f"  trial {t.number}: mean={t.user_attrs['reward_mean']} "
                  f"std={t.user_attrs['reward_std']} params={t.params}")

        # lexicographic optimum (objectives are ordered by priority);
        # it is always Pareto-optimal, so searching the front suffices
        best = max(pareto, key=lambda t: tuple(t.values))
        print("Lexicographic best trial:", best.number)
        print("  mean reward:", best.user_attrs["reward_mean"])
        print("  std reward:", best.user_attrs["reward_std"])
        print("  hyperparameters:", best.params)

        return best