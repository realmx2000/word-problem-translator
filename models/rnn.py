import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .copy import Copy

class RNNModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, src_embed_dim, tgt_embed_dim, enc_hid_dim,
                 dec_hid_dim, dropout, teacher_forcing_ratio, src_embed_model=None):
        super().__init__()
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.tgt_vocab_size = tgt_vocab_size

        # Encoder
        if src_embed_model is None:
            self.src_embedding = nn.Embedding(src_vocab_size, src_embed_dim)
            self.src_embedding_lookup = self._src_embedding_wrapper
        else:
            self.src_embedding_lookup = src_embed_model
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
        self.softmax = nn.Softmax(dim=2)

        #Copy
        self.copy = Copy(enc_hid_dim * 2, enc_hid_dim * 2, tgt_embed_dim)

    def _src_embedding_wrapper(self, x, mask):
        return self.src_embedding(x)

    def _tgt_embedding_lookup(self, x, mask):
        return self.tgt_embedding(x)

    def forward(self, x):
        questions, question_lengths = pad_packed_sequence(x[0])
        equations, equation_lengths = pad_packed_sequence(x[1])
        alignments, alignment_lengths = pad_packed_sequence(x[2])

        question_masks = torch.zeros_like(questions)
        for i, length in enumerate(question_lengths):
            question_masks[:length, i] = 1

        # Encoder
        embeddings = self.src_embedding_lookup(questions, question_masks)
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
            embedding = self._tgt_embedding_lookup(prev_output, None)

            #prob_copy = self.copy(torch.cat((enc_projection, curr_hidden, embedding.squeeze(0)), dim=1))
            #token_attn = torch.index_select(attention_scores.flatten(), 0, alignments.view(-1).long())
            #token_attn = token_attn.view(-1, alignments.shape[0])
            #token_attn_sum = torch.sum(token_attn, dim=1)
            #copy_weight = token_attn / token_attn_sum # [batch size, num token length]

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

        return self.softmax(all_logits), all_logits, equations
