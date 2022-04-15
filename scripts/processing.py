import os.path
import json
import numpy as np


class CommonRunner:
    def __init__(self, learner, name, mdp_dict, directory, **kwargs):
        self.learner = learner
        self.name = name
        self.kwargs = kwargs
        self.T = mdp_dict[self.name][0].copy()
        self.R = mdp_dict[self.name][1].copy()
        self.map = mdp_dict[self.name][2].copy()
        self.render_policy = True
        self.directory = None
        self.data_dir = None

        if self.kwargs is not None and 'misc_kwargs' in self.kwargs:
            print(self.kwargs['misc_kwargs'])
            if self.kwargs['misc_kwargs'] is not None and 'nogame' in self.kwargs['misc_kwargs']:
                self.render_policy = False

        if directory is not None:
            self.directory = os.path.join(directory, learner, name)
            os.makedirs(self.directory, exist_ok=True)
            self.data_dir = os.path.join(self.directory, "_data")
            os.makedirs(self.data_dir, exist_ok=True)

    @staticmethod
    def save_file(file_name, data):
        f = open(file_name, 'w')
        json.dump(data, f, indent=2, default=CommonRunner.json_serializer)
        f.close()

    @staticmethod
    def json_serializer(value):
        return int(value) if isinstance(value, np.int64) else float(value)

