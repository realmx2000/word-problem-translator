from datasets import BaseDataset
from dataloaders import SequenceLoader
from models import RNNModel

from torch.nn.functional import cross_entropy
from torch.nn.utils.rnn import pad_packed_sequence


if __name__ == '__main__':
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    ENC_HID_DIM = 512
    DEC_HID_DIM = 512
    DROPOUT = 0.5
    FORCING_RATIO = 0.5

    dataset = BaseDataset(['kushman.json'])
    dataloader = SequenceLoader(dataset, 8, 'train')
    model = RNNModel(len(dataset.src_vocab),
                    len(dataset.tgt_vocab),
                    ENC_EMB_DIM,
                    ENC_HID_DIM,
                    DEC_EMB_DIM,
                    DEC_HID_DIM,
                    DROPOUT,
                    FORCING_RATIO)
    for i, batch in enumerate(dataloader):
        equations, lengths = pad_packed_sequence(batch[1])
        logits, equations = model.forward(batch)
        logits = logits[1:].view(-1, logits.shape[-1])
        labels = equations[1:].view(-1)
        loss = cross_entropy(logits, labels)
        print(loss)
        break
