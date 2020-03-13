"""
Usage:
    train.py --dataset=<file>

Options:
    --dataset=<file>                      dataset file, such as "kushman.json"
"""

from datasets import BaseDataset, TokenizedDataset
from dataloaders import SequenceLoader
from models import RNNModel, ContextualEmbeddingModel

import torch.optim as optim
import torch
import time
import math
from tqdm import tqdm

from docopt import docopt

import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence

def train(model, dataloader, optimizer):
    epoch_loss = 0
    epoch_acc = 0
    loss_fn = nn.NLLLoss()
    total = len(dataloader.dataset) // dataloader.batch_size
    pbar = tqdm(enumerate(dataloader), total=total)
    for i, batch in pbar:
        optimizer.zero_grad()
        probs, equations = model.forward(batch)
        probs_flat = probs[1:].view(-1, probs.shape[-1])
        labels = equations[1:].view(-1)
        loss = loss_fn(torch.log(probs_flat), labels)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        '''
        preds = probs[1:].argmax(2).squeeze(1).tolist()
        labels_list = labels.tolist()
        print(' '.join(dataloader.dataset.tgt_vocab.indices2words(preds)))
        print(' '.join(dataloader.dataset.tgt_vocab.indices2words(labels_list)))
        '''

        epoch_loss += loss.item()
        pbar.set_description("Loss: {:3f}".format(loss))
        pbar.refresh()
    return epoch_loss / len(dataloader)

def validate(model, dataloader, VERBOSE=False):
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            equations, lengths = pad_packed_sequence(batch[1])
            logits, outputs = model.forward(batch)
            predicted = logits.argmax(-1)
            logits = logits[1:].view(-1, logits.shape[-1])
            labels = equations[1:].view(-1)
            loss = cross_entropy(logits, labels)
            predicted_list = torch.split(predicted, 1, dim=1)
            tgt_vocab = dataloader.dataset.tgt_vocab

            predicted_list = [tgt_vocab.indices2words(i.flatten().tolist()) for i in predicted_list]
            if VERBOSE:
                print("".join(predicted_list[0]))
            epoch_loss += loss.item()
            epoch_acc += (predicted == equations).float().sum()
        return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

if __name__ == '__main__':

    args = docopt(__doc__)

    dataset_file = args['--dataset']
    VERBOSE = True

    ENC_EMB_DIM = 256#768
    DEC_EMB_DIM = 256
    ENC_HID_DIM = 512
    DEC_HID_DIM = 512
    DROPOUT = 0
    FORCING_RATIO = 1
    BATCH_SIZE = 8

    #dataset = BaseDataset([dataset_file])
    #dataloader = SequenceLoader(dataset, BATCH_SIZE, 'train')
    dataset = BaseDataset([dataset_file])
    dataset.convert()
    dataloader = SequenceLoader(dataset, BATCH_SIZE, 'train')
    #embedding_model = ContextualEmbeddingModel('bert', dataset.max_num_variables, dataset.max_num_constants)

    '''
    print(dataset.questions)
    questions = [q for q in dataset.src_vocab.indices2words(dataset.questions[0].tolist())]
    equations = [e for e in dataset.tgt_vocab.indices2words(dataset.equations[0].tolist())]
    print(questions)
    print(equations)
    print(dataset.alignments)
    print(dataset.solutions)
    '''

    #criterion = nn.CrossEntropyLoss() #TODO: add ignore index for pad token?
    model = RNNModel(len(dataset.src_vocab),
                    len(dataset.tgt_vocab),
                    ENC_EMB_DIM,
                    ENC_HID_DIM,
                    DEC_EMB_DIM,
                    DEC_HID_DIM,
                    DROPOUT,
                    FORCING_RATIO,
                    src_embed_model=None,
                    const_mapping=dataset.const_idxs)

    optimizer = optim.Adam(model.parameters(), lr=1e-3) # For RNN - 1e-2

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(model):,} trainable parameters')
    
    N_EPOCHS = 100 #max number of epochs to train for
    CLIP = 1

    best_validation_loss = float('inf')
    
    print("BEGINNING MODEL TRAINING")
    for epoch in range(N_EPOCHS):
        start_time = time.time()

        training_loss = train(model, dataloader, optimizer)
        #validation_loss, validation_acc = validate(model, dataloader, VERBOSE) #TODO: don't use same dataloader for training and validation

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        '''
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            print("New lowest validation loss! Saving model in " + dataset_file + "_model.pt")
            torch.save(model.state_dict(), dataset_file + '_model.pt')
            #TODO: early stopping
        '''
        
        print(f'Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {training_loss:.3f} | Train PPL: {math.exp(training_loss):7.3f}')
        #print(f'\t Val. Loss: {validation_loss:.3f} |  Val. PPL: {math.exp(validation_loss):7.3f}')
        #print(f'\t Val. Acc: {validation_acc:.3f}')

model.load_state_dict(torch.load(dataset_file + '_model.pt'))

test_loss = validate(model, dataloader) #TODO: don't use same dataloader for training and testing

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
