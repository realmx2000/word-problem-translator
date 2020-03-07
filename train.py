"""
Usage:
    train.py --dataset=<file>

Options:
    --dataset=<file>                      dataset file, such as "kushman.json"
"""

from datasets import BaseDataset
from dataloaders import SequenceLoader
from models import RNNModel

import torch.optim as optim
import torch
import time
import math

from docopt import docopt

from torch.nn.functional import cross_entropy
from torch.nn.utils.rnn import pad_packed_sequence

def train(model, dataloader, optimizer, clip):
    epoch_loss = 0
    for i, batch in enumerate(dataloader):
        #print(i)
        optimizer.zero_grad()
        #equations, lengths = pad_packed_sequence(batch[1])
        logits, equations = model.forward(batch)
        logits = logits[1:].view(-1, logits.shape[-1])
        labels = equations[1:].view(-1)
        loss = cross_entropy(logits, labels)
        #print(loss)

        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        preds = logits.argmax(1).tolist()
        labels_list = labels.tolist()
        print(' '.join(dataloader.dataset.tgt_vocab.indices2words(preds)))
        print(' '.join(dataloader.dataset.tgt_vocab.indices2words(labels_list)))

        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def validate(model, dataloader):
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            equations, lengths = pad_packed_sequence(batch[1])
            logits, equations = model.forward(batch)
            logits = logits[1:].view(-1, logits.shape[-1])
            labels = equations[1:].view(-1)
            loss = cross_entropy(logits, labels)
            
            epoch_loss += loss.item()
        return epoch_loss / len(dataloader)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

if __name__ == '__main__':

    args = docopt(__doc__)

    dataset_file = args['--dataset']

    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    ENC_HID_DIM = 512
    DEC_HID_DIM = 512
    DROPOUT = 0
    FORCING_RATIO = 1
    BATCH_SIZE = 1

    dataset = BaseDataset([dataset_file])
    dataloader = SequenceLoader(dataset, BATCH_SIZE, 'train')
    model = RNNModel(len(dataset.src_vocab),
                    len(dataset.tgt_vocab),
                    ENC_EMB_DIM,
                    ENC_HID_DIM,
                    DEC_EMB_DIM,
                    DEC_HID_DIM,
                    DROPOUT,
                    FORCING_RATIO)

    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(model):,} trainable parameters')
    
    N_EPOCHS = 100 #max number of epochs to train for
    CLIP = 1

    best_validation_loss = float('inf')
    
    print("BEGINNING MODEL TRAINING")
    for epoch in range(N_EPOCHS):
        start_time = time.time()

        training_loss = train(model, dataloader, optimizer, CLIP)
        validation_loss = validate(model, dataloader) #TODO: don't use same dataloader for training and validation

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), dataset_file + '_model.pt')
            #TODO: early stopping
        
        print(f'Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {training_loss:.3f} | Train PPL: {math.exp(training_loss):7.3f}')
        print(f'\t Val. Loss: {validation_loss:.3f} |  Val. PPL: {math.exp(validation_loss):7.3f}')

model.load_state_dict(torch.load(dataset_file + '_model.pt'))

test_loss = validate(model, dataloader) #TODO: don't use same dataloader for training and testing

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
