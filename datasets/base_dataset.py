import torch
import json
import re
from itertools import chain

from torch.utils.data import Dataset
from .vocab import VocabEntry

class BaseDataset(Dataset):
    def __init__(self, names: list):
        """
        Args:
            names: Names of the JSON files to load as the dataset.
        """
        self.questions = []
        self.equations = []
        self.alignments = []
        self.solutions = []
        self.const_idxs = {}
        self.var_dicts = []
        self.const_dicts = []

        self.src_vocab = None
        self.tgt_vocab = None
        self.src_pad_token = None
        self.tgt_pad_token = None
        self.max_num_variables = 0
        self.max_num_constants = 0

        self.create_dataset(names)
        self.create_vocabs()

    def create_dataset(self, names: list):
        """
        Args:
            names: Names of the JSON files to load as dataset.

        End State:
            self.questions will be a list of lists, the outer list containing questions, and
            the inner lists containing the individual words/tokens in those questions.
            self.equations will also be a list of lists, the outer list containing systems of
            equations, delimited by commas, and the inner list containing individual characters.
            self.alignments will be a list of tensors, each tensor representing a map from a token
            representing a constant to the constant's position in the question text.
        """
        PREFIX = 'data/microsoft/'
        for filename in names:
            with open(PREFIX + filename, 'r') as file:
                data = json.load(file)
            for example in data:
                question = example['sQuestion']
                equation_system = example['lEquations']
                solution = example['lSolutions']

                # Remove commas in numbers
                question = question.replace('-', ' ').split(' ')
                for i, token in enumerate(question):
                    if ',' in token and len(token) > 1:
                        question[i] = token.replace(',', '')
                question = ' '.join(question)
                question = '<s> ' + question + ' </s>'

                # Replace constants and tokenize by word
                question, const_dict = self.replace_constants(question, {})
                question = question.split(' ')

                # Replace variables and constants in equation system
                var_label = 'a'
                var_dict = {}
                keep = True
                for idx, equation in enumerate(equation_system):
                    prev_size = len(const_dict)
                    equation, var_dict = self.replace_variables(equation, var_dict, var_label, const_dict)

                    equation, updated_consts = self.replace_constants(equation, const_dict)
                    # Drop the example if there are constants in the equation we didn't identify.
                    # TODO: Find a better way to process these.
                    if len(updated_consts) != prev_size:
                        keep = False
                    equation_system[idx] = list(equation)

                if not keep or (len(const_dict.items()) == 0):
                    continue

                # Compute the alignments between constants and tokens
                alignment_dict = self.compute_alignments(question, const_dict)

                concat_equation = ''
                for equation in equation_system:
                    concat_equation += ''.join(equation) + ','

                concat_equation = ['<s>'] + list(concat_equation[:-1]) + ['</s>']

                self.questions.append(question)
                self.equations.append(concat_equation)
                self.alignments.append(alignment_dict)
                self.solutions.append(torch.tensor(solution))
                self.var_dicts.append(var_dict)
                self.const_dicts.append(const_dict)
                self.max_num_constants = max(self.max_num_constants, len(const_dict))
                self.max_num_variables = max(self.max_num_variables, len(var_dict))

    @staticmethod
    def compute_alignments(question, const_dict):
        #const_alignment_vec = torch.zeros(len(const_dict))
        alignment_dict = {}
        for name in const_dict.values():
            matches = [i for i, x in enumerate(question) if x == name]
            #const_alignment_vec[int(name)] = float(matches[0])
            alignment_dict[int(name)] = torch.tensor(matches).long()
        return alignment_dict#const_alignment_vec

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
        question_mode = len(const_dict) == 0
        # Replace all numeric constants with number tokens. Spaces need to be there in the question, but
        # not in the equation.
        constant_re = re.compile(' [0-9][0-9.]* ') if question_mode else re.compile('[0-9][0-9.]*')
        constants = reversed(list(constant_re.finditer(string)))
        for match in constants:
            const_val = float(match.group())
            if const_val in const_dict:
                constant =  const_dict[const_val]
            else:
                constant = str(int(const_label))
                const_label += 1
                const_dict[const_val] = constant
            if question_mode:
                string = string[:match.start() + 1] + constant + string[match.end() - 1:]
            else:
                string = string[:match.start()] + constant + string[match.end():]

        return string, const_dict

    @staticmethod
    def replace_variables(equation: str, var_dict: dict, var_label: str, const_dict: dict):
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
        variable_re = re.compile('[^()+\-*/^=]+')

        variables = reversed(list(variable_re.finditer(equation)))
        for match in variables:
            var_name = match.group()
            try:
                if float(var_name) in const_dict:
                    continue
                else:
                    const_dict[var_name] = 0  # This example will be rejected
            except:
                pass
            if var_name in var_dict:
                var = var_dict[var_name]
            else:
                var = var_label
                var_dict[var_name] = var
                var_label = chr(ord(var_label) + 1)
            equation = equation[:match.start()] + var + equation[match.end():]
        return equation, var_dict

    def create_vocabs(self):
        """
        Creates the index vocabularies for the data and converts self.questions and self.equations to
        be lists of tensors of indices instead of lists of lists of tokens.
        """
        question_corpus = list(chain.from_iterable(self.questions))
        self.src_vocab = VocabEntry.from_corpus(question_corpus, len(question_corpus), 1)
        equation_corpus = list(chain.from_iterable(self.equations))
        self.tgt_vocab = VocabEntry.from_corpus(equation_corpus, len(equation_corpus), 1)

        self.src_pad_token = self.src_vocab['<pad>']
        self.tgt_pad_token = self.tgt_vocab['<pad>']

        for const in range(self.max_num_constants):
            self.const_idxs[const] = self.tgt_vocab[str(const)]

    def convert(self):
        self.questions = [torch.tensor(question) for question in self.src_vocab.words2indices(self.questions)]
        self.equations = [torch.tensor(equation) for equation in self.tgt_vocab.words2indices(self.equations)]

    def __getitem__(self, idx):
        return self.questions[idx], self.equations[idx], self.alignments[idx], \
               self.solutions[idx], self.var_dicts[idx], self.const_dicts[idx]

    def __len__(self):
        return len(self.questions)

if __name__ == '__main__':
    '''
    prob = input('Enter problem: ')
    eq = input('Enter equation: ')
    prob, const_dict = BaseDataset.replace_constants(prob, {})
    print(prob)
    print(const_dict)
    eq, var_dict = BaseDataset.replace_variables(eq, {}, 'a')
    eq, _ = BaseDataset.replace_constants(eq, const_dict, 0)
    print(eq)
    print(var_dict)
    '''
    dataset = BaseDataset(['kushman.json'])
    import ipdb
    ipdb.set_trace()
    print(dataset.max_num_constants)
    print(dataset.max_num_variables)
    print(len(dataset))
