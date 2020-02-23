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
        questions = []
        equations = []
        #embeddings = gensim.models.Word2Vec.load_word2vec_format('data/embeddings/GoogleNews-vectors-negative300.bin', binary=True)
        for filename in names:
            with open(PREFIX + filename, 'r') as file:
                data = json.load(file)
            for example in data:
                questions.append(data['sQuestion'].split(' '))
                equations.append(data['lEquations'].split(' '))


if __name__ == '__main__':
    EquationDataset(['kushman.json'])
