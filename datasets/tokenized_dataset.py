import torch
from transformers import BertTokenizer

from .base_dataset import BaseDataset

class TokenizedDataset(BaseDataset):
    def __init__(self, names: list, embeddings: str='bert'):
        super().__init__(names)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.tokenized_questions = []
        self.tokenized_equations = []
        for i, question in enumerate(self.questions):
            tokenized_question = tokenizer.encode(question, add_special_tokens=True)
            tokenized_system = []
            system = ''
            for equation in self.equations[i]:
                system += equation
                #tokenized_equation = tokenizer.encoder(equation, add_special_tokens=True)
            tokenized_system.append(tokenizer.encoder(system, add_special_tokens=True))
            self.tokenized_questions.append(tokenized_question)
            self.tokenized_equations.append(tokenized_system)

    def __getitem__(self, idx):
        question = torch.tensor(self.tokenized_questions[idx])
        eq_system = torch.tensor(self.tokenized_equations[idx])
        return question, eq_system
