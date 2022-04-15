from hiive.mdptoolbox.mdp import PolicyIteration, ValueIteration
import numpy as np
import pandas as pd
import os.path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scripts.problem import FrozenLakeSimulator, TreeSimulator, simulate_policy, render_lake, render_forest, plot_forest
from scripts.processing import CommonRunner

seed = 42


class IterationRunner(CommonRunner):

    def __init__(self, name, mdp_dict, directory, learner_type=None, **kwargs):
        super().__init__(learner=learner_type, name=name, mdp_dict=mdp_dict, directory=directory, **kwargs)
        self.learner_type = learner_type
        self.long_name = 'Value Iteration' if self.learner == 'value' else 'Policy Iteration'

    def run(self):
        if self.name == 'lake':
            gamma_range = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
        else:
            gamma_range = [0.1, 0.5, 0.7, 0.9, 0.95, 0.999]

        all_dfs = []
        simulated_dfs = []
        policies = []
        for gamma in gamma_range:
            print(f"   gamma={gamma}")

            run_stats = None
            values = None
            policy = None
            df = None
            sim_results_df = None

            run_stats_file = None
            values_file = None
            policy_file = None
            df_file = None
            policy_img_file = None
            sim_results_file = None
            if self.directory is not None:
                base_file = f"gamma_{gamma}"
                run_stats_file = os.path.join(self.data_dir, f"{base_file}.json")
                values_file = os.path.join(self.data_dir, f"{base_file}.values.npy")
                policy_file = os.path.join(self.data_dir, f"{base_file}.policy.npy")
                df_file = os.path.join(self.data_dir, f"{base_file}.csv")
                policy_img_file = os.path.join(self.directory, f"{base_file}.policy.png")
                sim_results_file = os.path.join(self.data_dir, f"{base_file}.sim.csv")
                print(sim_results_file)

            if run_stats is None or values is None or policy is None or df is None or sim_results_df is None:
                np.random.seed(seed)

                if self.learner_type == 'value':
                    learner = ValueIteration(transitions=self.T, reward=self.R, gamma=gamma, epsilon=0.001,
                                             max_iter=None if gamma < 1 else int(1e5), skip_check=False,
                                             run_stat_frequency=1)
                elif self.learner_type == 'policy':
                    if gamma >= 1:
                        continue
                    learner = PolicyIteration(transitions=self.T, reward=self.R, gamma=gamma, skip_check=False,
                                              max_iter=int(1e8), run_stat_frequency=1)
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
                df['gamma'] = gamma
                df['mdp'] = self.name
                df['alg'] = self.learner
                if df_file is not None:
                    df.to_csv(df_file)

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

            simulated_dfs.append(sim_results_df)
            policies.append(policy)
            if policy_img_file is not None and self.render_policy:
                if self.name == 'lake':
                    render_lake(self.map, policy_img_file, policy)
                else:
                    render_forest(policy_img_file, policy, title=f"{self.long_name}: {self.name} Policy (g={gamma})")

            all_dfs.append(df)

        row = 2
        col = 6
        if self.name == 'lake':
            width = 17
            height = 7
        else:
            width = 17
            height = 4
        plt.clf()
        fig, axes = plt.subplots(nrows=row, ncols=col, figsize=(width, height))
        fig.suptitle(f"{self.name}: {self.long_name} Found Policies For Various γ", fontsize=12)
        fig.tight_layout()
        index = -1
        for index, (gamma, policy) in enumerate(zip(gamma_range, policies)):
            if index >= len(gamma_range):
                print(f"  WARN: OVER GAMMA RANGE INDEX")
                break
            i, j = np.unravel_index(index, axes.shape)
            ax = axes[i, j]
            ax.set_title(f"γ={gamma}")

            if self.name == 'lake':
                f = os.path.join(self.directory, f"gamma_{gamma}.policy.png")
                img = mpimg.imread(f)
                ax.imshow(img)
                ax.axis('off')
            else:
                plot_forest(policy, ax)

        index += 1
        while index < axes.size:
            i, j = np.unravel_index(index, axes.shape)
            ax = axes[i, j]
            ax.axis('off')
            index += 1

        plt.savefig(os.path.join(self.directory, f"{self.learner}_{self.name}_ALL_POLICY.png"))

        df = pd.concat(all_dfs)
        for sim_df in simulated_dfs:
            sim_df['Burned'] = sim_df['Cum Reward'] == 0
        sim_df = pd.concat(simulated_dfs)
        sim_df_last_gammas = sim_df.groupby(['gamma']).mean().reset_index()
        sim_df_last_gammas.to_csv(os.path.join(self.directory, f"{self.learner}_{self.name}_sim.csv"))

        plt.clf()
        ax = None
        y = 'Reward' if self.name == 'lake' else 'Cum Reward'
        sim_df_last_gammas.plot(x='gamma', y=y)
        plt.xlabel('Gamma')
        plt.ylabel('Simulated Reward')
        plt.legend(['Simulated Reward'])
        plt.title(f'{self.long_name}: ({self.name}) Gamma vs Average Reward for 75 epochs')
        plt.tight_layout()
        if self.name == 'forest':
            plt.yscale('log')

        if self.directory is not None:
            img_file = os.path.join(self.directory, f"{self.learner}_{self.name}_gamma_sim.png")
            plt.savefig(img_file)

        self.plot_over_gamma(df=df, gamma_range=gamma_range, x='Iteration', y='Time', xlabel='Iteration',
                             ylabel='Time', title_postfix='Time vs Iteration', img_base_file='time_iteration.png')

        df_last_gammas = df.sort_values('Iteration').groupby(['gamma']).last().reset_index()

        self.plot_against_gamma(df=df_last_gammas, y='Time', ylabel='Time',
                                title_postfix='Gamma vs Time Until Convergence',
                                img_base_file='gamma_time.png')

        self.plot_against_gamma(df=df_last_gammas, y='Error', ylabel='Error',
                                title_postfix='Gamma vs Error At Convergence', img_base_file='gamma_error.png')

        plt.clf()
        row = 1
        col = 3
        width = 16
        height = 5
        plt.clf()
        fig, axes = plt.subplots(nrows=row, ncols=col, figsize=(width, height))
        fig.suptitle(f"{self.name}: {self.long_name} Effects of γ", fontsize=12)
        fig.tight_layout()

        for index, file_part in enumerate(["gamma_sim", "gamma_error", "gamma_time"]):
            f = os.path.join(self.directory, f"{self.learner}_{self.name}_{file_part}.png")
            img = mpimg.imread(f)
            axes[index].imshow(img)
            axes[index].axis('off')

        if self.directory is not None:
            img_file = os.path.join(self.directory, f"{self.learner}_{self.name}_EFFECT_GAMMA.png")
            plt.savefig(img_file)

        df_last_gammas.to_csv(os.path.join(self.directory, f"{self.learner}_{self.name}_results.csv"))
        return df

    def plot_over_gamma(self, df, gamma_range, x, y, xlabel, ylabel, title_postfix, img_base_file):
        plt.clf()
        ax = None
        for gamma in gamma_range:
            sub_df = df[df['gamma'] == gamma].sort_values(['Iteration'])
            _ax = sub_df.plot(x=x, y=y, ax=ax)
            if ax is None:
                ax = _ax
        plt.legend(gamma_range)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f'{self.long_name}: ({self.name}) {title_postfix}')

        if self.directory is not None:
            img_file = os.path.join(self.directory, f"{self.learner}_{self.name}_{img_base_file}")
            plt.savefig(img_file)

    def plot_against_gamma(self, df, y, ylabel, title_postfix, img_base_file):
        plt.clf()
        df.plot(x='gamma', y=y)
        plt.xlabel('Gamma')
        plt.ylabel(ylabel)
        plt.title(f'{self.long_name}: ({self.name}) {title_postfix}')
        plt.tight_layout()

        if self.directory is not None:
            img_file = os.path.join(self.directory, f"{self.learner}_{self.name}_{img_base_file}")
            plt.savefig(img_file)
