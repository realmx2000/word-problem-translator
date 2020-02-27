import torch
import torch.nn as nn

from .contextual_embedding_model import ContextualEmbeddingModel

class ContextualRNN(ContextualEmbeddingModel):
    def __init__(self, embedding_type: str, model_type: str, max_num_variables: int, max_num_constants: int,
                 hidden_size=200, freeze_embeddings: bool=False):
        """
        Args:
            embedding_type: What kind of pretrained embeddings to use. Current options: 'bert'.
            model_type: What kind of model to use for the encoder/decoder. Current options: 'lstm'.
            max_num_variables: The most variables that appear in any example. Used to determine the
            vocabulary.
            max_num_constants: The most constants that appear in any example. Used to determine the
            vocabular.
            freeze_embeddings: Whether or not to freeze the pretrained embeddings.
        """
        super().__init__(embedding_type, max_num_variables, max_num_constants, freeze_embeddings)

        if model_type == 'lstm':
            self.encoder =
            self.decoder = 