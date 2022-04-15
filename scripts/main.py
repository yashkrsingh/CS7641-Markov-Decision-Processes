import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.ilearner import IterationRunner
from scripts.problem import create_lake, create_forest
from scripts.qlearner import QRunner

directory = "misc"

AllMDPs = ['lake', 'forest']
AllSections = ['value', 'policy', 'q', 'mvp']


class MainRunner:
    def __init__(self, problem, sections, base_directory=None, misc_kwargs=None):

        if problem is None or 'all' in problem:
            problem = AllMDPs.copy()
        self.mdp = problem

        if sections is None or 'all' in sections:
            sections = AllSections.copy()
        self.sections = sections

        self.base_directory = base_directory

        lake_size = 8
        lake_p = 0.85
        forest_size = 2000
        forest_max_wait = forest_size * 2
        forest_max_cut = forest_size
        forest_burn_p = 0.001

        self.misc_kwargs = misc_kwargs
        self.runner_kwargs = {
            'directory': self.base_directory,
            'mdp_dict': {
                'lake': create_lake(size=lake_size, p=lake_p),
                'forest': create_forest(size=forest_size, wait_max_reward=forest_max_wait,
                                        cut_max_reward=forest_max_cut, p_fire=forest_burn_p),
            },
            'misc_kwargs': self.misc_kwargs,
        }

    def run(self):
        for section in self.sections:
            {
                'value': self.run_value_iteration,
                'policy': self.run_policy_iteration,
                'mvp': self.merge_results,
                'q': self.run_rl,
            }[section]()

    def run_value_iteration(self):
        for mdp in self.mdp:
            print(f"Part 1. Value Iteration :: MDP = {mdp}")
            sub_runner = IterationRunner(learner_type='value', name=mdp, **self.runner_kwargs)
            sub_runner.run()

    def run_policy_iteration(self):
        for mdp in self.mdp:
            print(f"Part 2. Policy Iteration :: MDP = {mdp}")
            sub_runner = IterationRunner(learner_type='policy', name=mdp, **self.runner_kwargs)
            sub_runner.run()

    def run_rl(self):
        for mdp in self.mdp:
            print(f"Part 3. Reinforcement Learning: MDP = {mdp}")
            sub_runner = QRunner(name=mdp, **self.runner_kwargs)
            sub_runner.run()

    def merge_results(self):
        for mdp_name in self.mdp:
            print(f"Part 4. Merge VI / PI / Q = {mdp_name}")

            params = {
                'lake': {
                    'gamma': 0.9,
                    'value': {},
                    'policy': {},
                    'q': {
                        'alpha': 0.10,
                        'epsilon': 1.0,
                        'alpha_decay': 0.9999,
                        'epsilon_decay': 0.9999,
                    }
                },
                'forest': {
                    'gamma': 0.999,
                    'value': {},
                    'policy': {},
                    'q': {
                        'alpha': 0.10,
                        'epsilon': 1.0,
                        'alpha_decay': 0.9999,
                        'epsilon_decay': 0.9999,
                    }
                }
            }

            sub_params = params[mdp_name]
            gamma = sub_params['gamma']

            sim_dfs = []
            result_dfs = []
            policies = []
            for alg in ['value', 'policy', 'q']:
                mdp_alg_dir = os.path.join(self.base_directory, alg, mdp_name, "_data")
                sim_file = None
                results_file = None
                if alg != 'q':
                    b = os.path.join(mdp_alg_dir, f"gamma_{gamma}")
                    sim_file = f"{b}.sim.csv"
                    results_file = f"{b}.json"
                    policy_file = f"{b}.policy.npy"
                else:
                    mdp_alg_dir = os.path.join(self.base_directory, alg, mdp_name, "grids", "_data")
                    b = os.path.join(mdp_alg_dir, f"alpha_decay=1.0")
                    sim_file = f"{b}.sim.csv"
                    results_file = f"{b}.json"
                    policy_file = f"{b}.policy.npy"

                sim_df = pd.read_csv(sim_file)
                result_df = None
                with open(results_file, 'r') as file_handle:
                    result_df = pd.DataFrame(json.load(file_handle))

                policy = np.load(policy_file)

                sim_dfs.append(sim_df)
                result_dfs.append(result_df)
                policies.append(policy)

            plt.clf()
            nrows = 1
            ncols = 3
            width = 16
            height = 4
            plt.clf()
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, height))
            plt.autoscale()

            colors = ['green', 'red', 'blue']
            for index, df in enumerate(result_dfs):
                ax = axes[index]
                ax.set_title(mdp_name + ' ' + ['value', 'policy', 'q'][index] + ": Iteration vs Error")
                df.plot(x='Iteration', y='Error', ax=ax, color=colors[index])

            plt.suptitle(mdp_name + " Delta Convergence")
            plt.savefig(os.path.join(self.base_directory, f"{mdp_name}_all_iter_error.png"))

            plt.clf()
            nrows = 1
            ncols = 3
            width = 16
            height = 4
            plt.clf()
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, height))
            plt.autoscale()

            for index, df in enumerate(result_dfs):
                ax = axes[index]
                ax.set_title(mdp_name + ' ' + ['value', 'policy', 'q'][index] + ": Iteration vs Mean Value")
                df.plot(x='Iteration', y='Mean V', ax=ax, color=colors[index])

                print(f"{mdp_name} :: {['value', 'policy', 'q'][index]} Total time: {df['Time'].iloc[-1]} Iteration: {df['Iteration'].iloc[-1]}")

            plt.savefig(os.path.join(self.base_directory, f"{mdp_name}_all_iter_value.png"))

            plt.clf()
            nrows = 1
            ncols = 3
            width = 16
            height = 4
            plt.clf()
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, height))
            plt.autoscale()

            for index, df in enumerate(result_dfs):
                ax = axes[index]
                ax.set_title(mdp_name + ' ' + ['value', 'policy', 'q'][index] + ": Iteration vs Reward Value")
                df.plot(x='Iteration', y='Reward', ax=ax, color=colors[index])

            plt.savefig(os.path.join(self.base_directory, f"{mdp_name}_all_iter_reward.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MDP Assignment Runner')

    parser.add_argument('-s', '--sections', type=str.lower, nargs='*', help='processes to run', default=['all'])
    parser.add_argument('-m', '--mdps', type=str.lower, nargs='*', help='mdp problems to run', default=['all'])
    parser.add_argument('-b', '--base_dir', type=str, dest='base_directory', help='directory for data saving / loading', default=directory)
    parser.add_argument('-k', '--kwargs', type=str, dest='misc_kwargs', help='misc kwargs to pass to a runner', default=None)

    args = parser.parse_args()
    runner = MainRunner(
        problem=args.mdps,
        sections=args.sections,
        base_directory=args.base_directory,
        misc_kwargs=args.misc_kwargs
    )
    runner.run()
