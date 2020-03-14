import torch
import json
import re
import numpy as np

from torch.utils.data import Dataset
from .vocab import VocabEntry

class EquivalenceDataset(Dataset):
    def __init__(self, names):
        self.OPERATORS = ['+', '-', '*', '/', '=']

        self.equations = []
        self.constants = []
        self.variables = []

        PREFIX = 'data/microsoft/'
        for filename in names:
            with open(PREFIX + filename, 'r') as file:
                data = json.load(file)
            for example in data:
                equation_system = example['lEquations']
                constants = set()
                variables = set()
                for idx, equation in enumerate(equation_system):
                    constants.update(self.find_constants(equation))
                    variables.update(self.find_variables(equation))
                self.equations.append(equation_system)
                self.constants.append(constants)
                self.variables.append(variables)

    @staticmethod
    def find_constants(equation):
        constant_re = re.compile('[0-9][0-9.]*')
        return set(constant_re.finditer(equation))

    @staticmethod
    def find_variables(equation):
        variable_re = re.compile('[^()+\-*/^=]+')
        return set(variable_re.finditer(equation))

    def __len__(self):
        return len(self.equations)

    def __getitem__(self, idx):
        equation_system = self.equations[idx]
        label = torch.randint(2, 1)
        if label[0] == 0:

    @staticmethod
    def get_expression_subtree_prefix(equation, idx, operators):
        if equation[idx] != '(':
            while idx != len(equation) and equation[idx] not in operators:
                idx += 1
        else:
            depth = 1
            while idx != len(equation) and depth != 0:
                while idx != len(equation) and equation[idx] != ')':
                    idx += 1
                depth -= 1
        return idx

    @staticmethod
    def get_expression_subtree_suffix(equation, idx, operators):
        if equation[idx] != ')':
            while idx != -1 and equation[idx] not in operators:
                idx -= 1
        else:
            depth = 1
            while idx != -1 and depth != 0:
                while idx != -1 and equation[idx] != '(':
                    idx -= 1
                depth -= 1
        return idx + 1

    @staticmethod
    def invariant_perturbation(equation_system, constants, variables, operators):
        permuted_system = np.permute(equation_system)
        equation = equation_system.join(',')
        all_vars = [chr(i) for i in range(97, 123)]
        for var in variables():
            replacement = np.random.choice(all_vars)
            all_vars.remove(replacement)
            equation = equation.replace(var, replacement)
            for i, char in enumerate(equation):
                if char in ['+', '-']:
                    left_begin = EquivalenceDataset.get_expression_subtree_suffix(equation, i - 1, operators)
                    right_end = EquivalenceDataset.get
