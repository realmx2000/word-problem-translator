import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .copy import Copy

class RNNModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, src_embed_dim, tgt_embed_dim, enc_hid_dim,
                 dec_hid_dim, dropout, teacher_forcing_ratio, src_embed_model=None, const_mapping=None):
        super().__init__()
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.tgt_vocab_size = tgt_vocab_size
        self.const_mapping = const_mapping

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
        self.softmax = nn.Softmax(dim=1)

        #Copy
        self.copy = Copy(enc_hid_dim * 2, enc_hid_dim * 2, tgt_embed_dim)

    def _src_embedding_wrapper(self, x, mask):
        return self.src_embedding(x)

    def _tgt_embedding_lookup(self, x, mask):
        return self.tgt_embedding(x)

    def forward(self, x):
        questions, question_lengths = pad_packed_sequence(x[0])
        equations, equation_lengths = pad_packed_sequence(x[1])
        alignments = x[2]

        question_masks = torch.zeros_like(questions)
        for i, length in enumerate(question_lengths):
            question_masks[:length, i] = 1

        # Encoder
        embeddings = self.src_embedding_lookup(questions, question_masks)
        encoded, enc_hidden = self.encoder(embeddings)
        enc_projection = torch.tanh(self.enc_fc(torch.cat((enc_hidden[-2,:,:], enc_hidden[-1,:,:]), dim=1)))
        enc_projection = self.enc_dropout(enc_projection)

        # Decoding
        all_probs = torch.zeros(equations.shape[0], equations.shape[1], self.tgt_vocab_size)
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
            num_constants = max([len(x) for x in alignments])
            token_attentions = torch.zeros((attention_scores.shape[0], num_constants))
            for batch_idx in range(attention_scores.shape[0]):
                att_scores = attention_scores[batch_idx, :, :].squeeze(0)
                for token, aligns in alignments[batch_idx].items():
                    token_attentions[batch_idx, token] = torch.index_select(att_scores, 0, aligns).sum()
            copy_weights = token_attentions / token_attentions.sum(dim=1, keepdim=True)

            copy_probs = torch.zeros((attention_scores.shape[0], self.tgt_vocab_size))
            for i in range(copy_weights.shape[0]):
                copy_probs[:, self.const_mapping[i]] = copy_weights[:, i]

            prob_gen = self.copy(torch.cat((enc_projection, curr_hidden, embedding.squeeze(0)), dim=1))\

            decoder_input = torch.cat((embedding, weighted), dim=2)
            output, dec_hidden = self.decoder(decoder_input, enc_projection.unsqueeze(0))

            logits = self.dec_out(torch.cat((output.squeeze(0), weighted.squeeze(0), embedding.squeeze(0)), dim=1))
            logits = self.dec_dropout(logits)
            probs = self.softmax(logits)
            all_probs[idx,:,:] = prob_gen * probs + (1 - prob_gen) * copy_probs

            # Create next state
            curr_hidden = dec_hidden.squeeze(0)
            teacher_force = random.random() < self.teacher_forcing_ratio
            prev_output = equations[idx] if teacher_force else logits.argmax(1)
            prev_output = prev_output.unsqueeze(0)

        return all_probs, equations
