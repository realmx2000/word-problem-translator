import torch
from transformers import BertTokenizer

from .base_dataset import BaseDataset

class TokenizedDataset(BaseDataset):
    def __init__(self, names: list, embeddings: str='bert'):
        super().__init__(names)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.src_pad_token = tokenizer.encode('[PAD]')[0]

        self.tokenized_questions = []
        #self.tokenized_equations = []
        for i, question in enumerate(self.questions):
            tokenized_question = tokenizer.encode(question, add_special_tokens=True)
            self.tokenized_questions.append(torch.tensor(tokenized_question))
            prev_alignment_vec = self.alignments[i]
            new_alignment_vec = self.compute_alignments(question, {str(x):str(x) for x in range(prev_alignment_vec.shape[0])})
            self.alignments[i] = new_alignment_vec

    def convert(self):
        self.equations = [torch.tensor(equation) for equation in self.tgt_vocab.words2indices(self.equations)]

    def __getitem__(self, idx):
        return self.tokenized_questions[idx], self.equations[idx], self.alignments[idx], self.solutions[idx]
