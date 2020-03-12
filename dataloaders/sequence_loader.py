import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader


class SequenceLoader(DataLoader):
    def __init__(self, dataset, batch_size: int, phase: str):
        self.pad_val_src = dataset.src_pad_token
        self.pad_val_tgt = dataset.tgt_pad_token
        super().__init__(dataset,
                         batch_size,
                         shuffle=(phase == 'train'),
                         collate_fn=self.collate_fn,
                         num_workers=0,
                         pin_memory=True)

    def pack_and_pad(self, sequence, pad_val):
        lengths = torch.tensor([item.shape[0] for item in sequence])
        batch = [torch.tensor(item) for item in sequence]
        batch = torch.nn.utils.rnn.pad_sequence(batch, padding_value=pad_val)
        packed = pack_padded_sequence(batch, lengths, enforce_sorted=False)
        return packed

    def collate_fn(self, batch):
        questions = []
        equations = []
        alignments = []
        solutions = []
        for example in batch:
            questions.append(example[0])
            equations.append(example[1])
            alignments.append(example[2])
            solutions.append(example[3])

        packed_questions = self.pack_and_pad(questions, self.pad_val_src)
        packed_equations = self.pack_and_pad(equations, self.pad_val_tgt)
        packed_alignments = self.pack_and_pad(alignments, -1)
        packed_solutions = self.pack_and_pad(solutions, float('nan'))
        return packed_questions, packed_equations, packed_alignments
