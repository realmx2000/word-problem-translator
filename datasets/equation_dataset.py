import torch
import numpy as np
import json
import re

from torch.utils.data import Dataset

class TokenizeDataset(Dataset):
    def __init__(self, names: list):
        """
        Args:
            names: Names of the JSON files to load as the dataset.
        """
        PREFIX = 'data/microsoft/'
        self.questions = []
        self.equations = []
        self.max_num_variables = 0
        self.max_num_constants = 0

        for filename in names:
            with open(PREFIX + filename, 'r') as file:
                data = json.load(file)
            for example in data:
                question = example['sQuestion']
                equation_system = example['lEquations']

                question, const_dict = self.replace_constants(question, {})
                question = question.split(' ')

                var_label = 'a'
                var_dict = {}
                for idx, equation in enumerate(equation_system):
                    equation, var_dict = self.replace_variables(equation, var_dict, var_label)

                    equation, _ = self.replace_constants(equation, const_dict)
                    equation_system[idx] = equation

                self.questions.append(question)
                self.equations.append(equation_system)
                self.max_num_constants = max(self.max_num_constants, len(const_dict))
                self.max_num_variables = max(self.max_num_variables, len(var_dict))

    @staticmethod
    def replace_constants(string: str, const_dict: dict, const_label: int=0) -> dict:
        '''
        Replace all numeric constants in the string with number tokens.
        Args:
            string: String to process.
            const_dict: Dictionary of constant -> number token mappings, to be used
                where possible.
            const_label: The constant token to start from for unseen constants

        Returns:
            string: the string with constants replaced.
            const_dict: updated dictionary of constant -> number token mappings.
        '''
        # Replace all numeric constants with number tokens.
        constant_re = re.compile('[0-9][0-9.]*')
        constants = reversed(list(constant_re.finditer(string)))
        for match in constants:
            const_val = float(match.group())
            if const_val in const_dict:
                constant =  const_dict[const_val]
            else:
                constant = '<n' + str(const_label) + '>'
                const_label += 1
                const_dict[const_val] = constant
            string = string[:match.start()] + constant + string[match.end():]

        return string, const_dict

    @staticmethod
    def replace_variables(equation: str, var_dict: dict, var_label: str):
        """
        Replace the variables in an equation with variable tokens.
        Args:
            equation: The equation to process.
            var_dict: Dictionary of var -> var token mappings, to be followed where possible.
            var_label: the label to start from for unseen variables.

        Returns:
            equation: the equation with variables replaced
            var_dict: the mapping of variables to variable tokens.
        """
        variable_re = re.compile('[^0-9.()+\-*/^=]+')

        variables = reversed(list(variable_re.finditer(equation)))
        for match in variables:
            var_name = match.group()
            if var_name in var_dict:
                var = var_dict[var_name]
            else:
                var = var_label
                var_dict[var_name] = var
                var_label = chr(ord(var_label) + 1)
            equation = equation[:match.start()] + var + equation[match.end():]
        return equation, var_dict

    def __getitem__(self, idx):
        return self.questions[idx], self.equations[idx]

    def __len__(self):
        return len(self.questions)

if __name__ == '__main__':
    prob = input('Enter problem: ')
    eq = input('Enter equation: ')
    prob, const_dict = EquationDataset.replace_constants(prob, {})
    print(prob)
    print(const_dict)
    eq, var_dict = EquationDataset.replace_variables(eq, {}, 'a')
    eq, _ = EquationDataset.replace_constants(eq, const_dict, 0)
    print(eq)
    print(var_dict)
