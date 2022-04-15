from hiive.mdptoolbox.mdp import QLearning
import numpy as np
import pandas as pd
import os.path
import matplotlib.pyplot as plt

from scripts.problem import FrozenLakeSimulator, TreeSimulator, simulate_policy, render_lake, render_forest
from scripts.processing import CommonRunner

seed = 42


class QRunner(CommonRunner):
    def __init__(self, name, mdp_dict, base_directory, **kwargs):
        super().__init__(learner="q", name=name, mdp_dict=mdp_dict, directory=base_directory, **kwargs)

        self.iter_callback = None
        if self.name == 'lake':
            self.lake_sim = FrozenLakeSimulator(self.T, self.R, None, self.map)
            self.iter_callback = self.lake_callback
        else:
            self.tree_sim = TreeSimulator(self.T, self.R, None)
            self.iter_callback = self.tree_callback

        self.grid_mode = True

    def lake_callback(self, s, a, s_new):
        return self.lake_sim.terminate(state=s_new)

    def tree_callback(self, s, a, s_new):
        return self.tree_sim.terminate(state=s_new)

    def run(self):
        if self.name == 'lake':
            gamma_range = [0.9]
        else:
            gamma_range = [0.999]

        base_params = {
            'lake': {
                'gamma': gamma_range[0],
                'alpha': 0.1,
                'epsilon': 1.0,
                'alpha_decay': 1.0,
                'epsilon_decay': 0.999,
            },
            'forest': {
                'gamma': gamma_range[0],
                'alpha': 0.20,
                'epsilon': 1.0,
                'alpha_decay': 1.0,
                'epsilon_decay': 0.999,
            }
        }[self.name]
        grids = [
            ('alpha', [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]),
            ('epsilon', [0.75, 0.85, 0.95, 0.99, 0.999, 0.9999, 1.0]),
            ('alpha_decay', [0.9, 0.99, 0.999, 0.9999, 1.0][::-1]),
            ('epsilon_decay', [0.9, 0.99, 0.999, 0.9999, 1.0][::-1]),
        ]

        self.dir = os.path.join(self.directory, 'grids')
        self.data_dir = os.path.join(self.dir, '_data')
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

        bucket_results = {
            k: []
            for k in [g[0] for g in grids]
        }
        all_dfs = []
        last_rows = []
        sim_results = []
        for grid in grids:
            param, values = grid
            for value in values:
                params = base_params.copy()
                params[param] = value
                gamma = params['gamma']
                alpha = params['alpha']
                epsilon = params['epsilon']
                alpha_decay = params['alpha_decay']
                epsilon_decay = params['epsilon_decay']

                run_stats = None
                q_values = None
                values = None
                policy = None
                df = None
                sim_results_df = None

                run_stats_file = None
                q_file = None
                values_file = None
                policy_file = None
                df_file = None
                policy_img_file = None
                base_file = None
                sim_results_file = None
                if self.dir is not None:
                    base_file = f"{param}={value}"
                    run_stats_file = os.path.join(self.data_dir, f"{base_file}.json")
                    q_file = os.path.join(self.data_dir, f"{base_file}.q.npy")
                    values_file = os.path.join(self.data_dir, f"{base_file}.values.npy")
                    policy_file = os.path.join(self.data_dir, f"{base_file}.policy.npy")
                    df_file = os.path.join(self.data_dir, f"{base_file}.csv")

                    policy_img_file = os.path.join(self.dir, f"{base_file}.policy.png")
                    sim_results_file = os.path.join(self.data_dir, f"{base_file}.sim.csv")

                if run_stats is None or values is None or policy is None or df is None:

                    alpha_min = 0.001
                    epsilon_min = 0.1
                    n_iter = 1000000
                    if self.name == 'forest':
                        n_iter = 1000000

                    np.random.seed(seed)

                    learner = QLearning(transitions=self.T, reward=self.R, gamma=gamma, alpha=alpha, alpha_min=alpha_min,
                                        alpha_decay=alpha_decay, epsilon=epsilon, epsilon_min=epsilon_min,
                                        epsilon_decay=epsilon_decay, n_iter=n_iter, iter_callback=self.iter_callback,
                                        skip_check=False)
                    learner.run()

                    run_stats = learner.run_stats
                    self.save_file(run_stats_file, run_stats)

                    values = learner.V
                    if values_file is not None:
                        np.save(values_file, values)

                    policy = learner.policy
                    if policy_file is not None:
                        np.save(policy_file, policy)

                    df = pd.DataFrame(data=run_stats)
                    df['mdp'] = self.name
                    df['alg'] = self.learner
                    df['alpha'] = alpha
                    df['alpha_decay'] = alpha_decay
                    df['alpha_min'] = alpha_min
                    df['epsilon'] = epsilon
                    df['epsilon_min'] = epsilon_min
                    df['epsilon_decay'] = epsilon_decay
                    df['n_iter'] = n_iter
                    if self.name == 'forest':
                        df['cut_count'] = np.sum(policy)
                        df['wait_p'] = 1 - (np.sum(policy) / len(policy))
                    if df_file:
                        df.to_csv(df_file)

                if sim_results_df is None:
                    simulator = None
                    if self.name == 'lake':
                        simulator = FrozenLakeSimulator(self.T, self.R, policy, self.map)
                    else:
                        simulator = TreeSimulator(self.T, self.R, policy)
                    sim_results_df = simulate_policy(simulator=simulator)
                    sim_results_df['mdp'] = self.name
                    sim_results_df['alg'] = self.learner
                    sim_results_df['gamma'] = gamma
                    if sim_results_file is not None:
                        sim_results_df.to_csv(sim_results_file)

                sim_results_df['gamma'] = gamma
                sim_results_df['alpha'] = alpha
                sim_results_df['alpha_decay'] = alpha_decay
                sim_results_df['epsilon'] = epsilon
                sim_results_df['epsilon_decay'] = epsilon_decay
                r_key = 'Reward'
                if self.name == 'forest':
                    sim_results_df['cut_count'] = np.sum(policy)
                    sim_results_df['wait_p'] = 1 - (np.sum(policy) / len(policy))
                    r_key = 'Cum Reward'
                sim_results.append(sim_results_df)
                bucket_results[param].append(sim_results_df)

                all_dfs.append(df)

                if policy_img_file is not None and self.render_policy:
                    if self.name == 'lake':
                        render_lake(self.map, policy_img_file, policy)
                    else:
                        render_forest(policy_img_file, policy, title=f"Q: {self.name} Policy ({param}={value})")

            conj = ','.join([
                str(x)
                for x in [gamma, alpha, alpha_decay, epsilon, epsilon_decay]
            ])

        sim_df = pd.concat(sim_results)
        sim_df_last_gammas = sim_df.groupby(['gamma', 'alpha', 'alpha_decay', 'epsilon', 'epsilon_decay']).mean().reset_index()
        sorter = 'Reward'
        if self.name == 'forest':
            sorter = 'Cum Reward'
        sim_df_last_gammas.sort_values(sorter, ascending=False).to_csv(
            os.path.join(self.dir, f"{self.learner}_{self.name}_sim.csv"))

        df = pd.concat(all_dfs)

        plt.clf()
        nrows = 1
        ncols = 3
        width = 16
        height = 4
        plt.clf()
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, height))
        plt.autoscale()

        colors = ['blue', 'green', 'blue', 'green']
        for index, param in enumerate(['alpha', 'epsilon', 'alpha_decay', 'epsilon_decay']):
            sub_df = pd.concat(bucket_results[param]).groupby(param).mean().reset_index()
            ax = axes[min(index, ncols - 1)]
            y = 'Cum Reward'
            ylabel = "Simulated Reward"
            xlabel = param
            if index >= ncols - 1:
                xlabel = "decay rate"
            if self.name == 'forest':
                y = 'wait_p'
                ylabel = "Wait %"

            sub_df.plot(x=param, y=y, ax=ax, color=colors[index])
            if index == 1:
                ax.set_title(f"{self.name}: Effect of Q-Learning parameters on {ylabel}")
            if index > 0:
                ylabel = ""
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)

            if index >= ncols - 1:
                ax.legend(["alpha_decay", "epsilon decay"])
            else:
                ax.legend([])

        if self.dir is not None:
            img_file = os.path.join(self.dir, f"{self.learner}_{self.name}_EFFECT_PARAMS.png")
            plt.savefig(img_file)

    def plot_iter(self, df, x, y, xlabel, ylabel, title, save_file_base):
        plt.clf()
        df.plot(x=x, y=y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"Q:{self.name}: {title}")
        plt.savefig(os.path.join(self.dir, save_file_base))
