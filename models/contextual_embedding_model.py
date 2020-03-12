import torch
import torch.nn as nn
from transformers import BertModel


class ContextualEmbeddingModel(nn.Module):
    def __init__(self, embedding_type: str, max_num_variables: int, max_num_constants: int,
                 freeze_embeddings: bool=False):
        """
        Args:
            embedding_type: What kind of pretrained embeddings to use. Current options: 'bert'.
            max_num_variables: The most variables that appear in any example. Used to determine the
            vocabulary.
            max_num_constants: The most constants that appear in any example. Used to determine the
            vocabular.
            freeze_embeddings: Whether or not to freeze the pretrained embeddings.
        """
        super().__init__()

        additional_variables = [chr(i) for i in range(ord('a'), ord('a') + max_num_variables)]
        additional_constants = list(range(1, max_num_constants + 1))
        self.vocabulary = ['(', ')', '+', '-', '*', '/', ',', 'END'] + additional_variables + additional_constants
        self.vocabulary_size = len(self.vocabulary)

        if embedding_type == 'bert':
            self.embedding_model = BertModel.from_pretrained('bert-base-uncased')

        if freeze_embeddings:
            for p in self.embedding_model.parameters():
                p.requires_grad = False

    def forward(self, x: torch.LongTensor, mask: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: Indices of the tokens within the model's voacbulary. Shape (batch x sequence_length)
            mask: Mask indicating which tokens are padding. Shape (batch x sequence_length)

        Returns:
            embeddings: FloatTensor of the model's embeddings for the input batch.
            Shape (batch, sequence_length, hidden_size)
        """
        # TODO: consider relative positional embeddings
        embeddings, _ = self.embedding_model(x, mask)
        return embeddings
