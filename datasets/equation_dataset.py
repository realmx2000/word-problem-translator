import torch
import numpy as np
import json
import gensim

from torch.utils.data import Dataset

class EquationDataset(Dataset):
    def __init__(self, names: list):
        """
        Args:
            names: Names of the JSON files to load as the dataset.
        """
        PREFIX = 'data/microsoft/'
        self.questions = []
        self.equations = []
        for filename in names:
            with open(PREFIX + filename, 'r') as file:
                data = json.load(file)
            for example in data:
                question = data['sQuestion'].split(' ')
                equation = data['lEquations']

                num_index = 0
                for idx, word in enumerate(question):
                    if word.isnumeric():
                        question[idx] = 'n' + str(num_index)

    def __getitem__(self, idx):
        return self.questions[idx], self.equations[idx]
if __name__ == '__main__':
    EquationDataset(['kushman.json'])
