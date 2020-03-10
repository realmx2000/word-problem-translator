import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .copy import Copy

class RNNModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, src_embed_dim, tgt_embed_dim, enc_hid_dim,
                 dec_hid_dim, dropout, teacher_forcing_ratio):
        super().__init__()
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.tgt_vocab_size = tgt_vocab_size

        # Encoder
        self.src_embedding = nn.Embedding(src_vocab_size, src_embed_dim)
        self.encoder = nn.GRU(src_embed_dim, enc_hid_dim, bidirectional=True)
        self.enc_fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.enc_dropout = nn.Dropout(dropout)

        # Attention
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)

        # Decoder
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, tgt_embed_dim)
        self.decoder = nn.GRU((enc_hid_dim * 2) + tgt_embed_dim, dec_hid_dim)
        self.dec_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + tgt_embed_dim, tgt_vocab_size)
        self.dec_dropout = nn.Dropout(dropout)

        #Copy
        self.copy = Copy(enc_hid_dim * 2, enc_hid_dim * 2, tgt_embed_dim)

    def forward(self, x):
        questions, question_lengths = pad_packed_sequence(x[0])
        equations, equation_lengths = pad_packed_sequence(x[1])
        alignments, alignment_lengths = pad_packed_sequence(x[2])

        # Encoder
        embeddings = self.src_embedding(questions)
        encoded, enc_hidden = self.encoder(embeddings)
        enc_projection = torch.tanh(self.enc_fc(torch.cat((enc_hidden[-2,:,:], enc_hidden[-1,:,:]), dim=1)))
        enc_projection = self.enc_dropout(enc_projection)

        # Decoding
        all_logits = torch.zeros(equations.shape[0], equations.shape[1], self.tgt_vocab_size)
        prev_output = equations[0].unsqueeze(0)
        curr_hidden = enc_projection
        for idx in range(1, equations.shape[0]):
            # Attention
            att_hidden = curr_hidden.unsqueeze(1).repeat(1, encoded.shape[0], 1)
            att_encoded = encoded.permute(1, 0, 2)
            energy = torch.tanh(self.attn(torch.cat((att_hidden, att_encoded), dim=2)))
            attention = self.v(energy).squeeze(2)
            attention_scores = F.softmax(attention, dim=1).unsqueeze(1)
            weighted = torch.bmm(attention_scores, att_encoded)
            weighted = weighted.permute(1, 0, 2)

            # Decoder

            embedding = self.tgt_embedding(prev_output)
            prob = self.copy(torch.cat((enc_projection, curr_hidden, embedding.squeeze(0)), dim=1))
            decoder_input = torch.cat((embedding, weighted), dim=2)
            output, dec_hidden = self.decoder(decoder_input, enc_projection.unsqueeze(0))

            logits = self.dec_out(torch.cat((output.squeeze(0), weighted.squeeze(0), embedding.squeeze(0)), dim=1))
            #print(logits)
            logits = self.dec_dropout(logits)
            all_logits[idx,:,:] = logits

            # Create next state
            curr_hidden = dec_hidden.squeeze(0)
            teacher_force = random.random() < self.teacher_forcing_ratio
            prev_output = equations[idx] if teacher_force else logits.argmax(1)
            prev_output = prev_output.unsqueeze(0)

        return all_logits, equations
