from __future__ import annotations

import torch
import Levenshtein
import itertools
import numpy as np
import torch.nn.functional as F

from typing import Literal
from numpy import ndarray
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

from mse.utils.list import to_batch, flatten, unflatten


class SimilarityModel:

    def __init__(
        self,
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'
    ) -> None:
        """
        Parameters
        ----------
        model_name : str, optional
            The name of the pre-trained transformer model to use for similarity computations,
            by default 'sentence-transformers/all-MiniLM-L6-v2'.
        """
        self.model_name = model_name

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    # | Distance
    # |-------------------
    def distance(
        self,
        a: str | list[str],
        b: str | list[str],
        method: Literal['matrix', 'pairwise'] = 'matrix',
        progress: bool = False,
        desc: str | None = None,
        leave: bool = False
    ) -> float | ndarray[float] | ndarray[ndarray[float]]:
        """
        Compute Levenshtein distances between two string sequences.

        Parameters
        ----------
        a : str | list[str]
            First sequence of strings.
        b : str | list[str]
            Second sequence of strings.
        method : Literal['matrix', 'pairwise']
            'matrix' computes distances for all pairs; 'pairwise' for corresponding elements.
        progress : bool | tqdm
            Displays a progress bar or uses provided tqdm object.

        Returns
        -------
        float | ndarray[float] | ndarray[ndarray[float]]
            Distances as a 2D array for 'matrix', and 1D array for 'pairwise'.

        Raises
        ------
        ValueError
            If lengths of a and b differ with 'pairwise' method.
        """
        a = to_batch(a)
        b = to_batch(b)

        # Compute distances
        match method:
            case 'matrix':
                pairs = tqdm([[(ai, bi) for bi in b] for ai in a], disable=not progress, desc=desc, leave=leave)
                distances = [[Levenshtein.distance(ai, bi) for ai, bi in row] for row in pairs]
            case 'pairwise':

                # Validate input dimensions
                if len(a) != len(b):
                    raise ValueError('Lengths of sequences must be equal for pairwise distance')

                pairs = tqdm(list(zip(a, b)), disable=not progress, desc=desc, leave=leave)
                distances = [Levenshtein.distance(ai, bi) for ai, bi in pairs]

        # Remove unnecessary dimensions
        distances = np.array(distances).squeeze()
        return distances

    # | Similarity
    # |-------------------
    @staticmethod
    def _mean_pooling(
        model_output: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean pool the transformer model's output.

        Parameters
        ----------
        model_output : torch.Tensor
            Transformer model output.
        attention_mask : torch.Tensor
            Transformer model attention mask.

        Returns
        -------
        torch.Tensor
            Tensor with values averaged over sequence length per attention mask.
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(token_embeddings).float()

        # Compute mean pooling
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    @torch.no_grad()
    def encode(
        self,
        *inputs: str | list[str],
        batch_size: int | None = 64,
        progress: bool | tqdm = False,
        desc: str | None = None
    ) -> torch.Tensor | list[torch.Tensor]:
        """
        Encode input sequences into fixed-dimensional embeddings.

        Parameters
        ----------
        *inputs : str | list[str]
            Strings or lists of strings to be encoded.
        batch_size : int, optional
            Batch size for encoding. Default is 64.
        progress : bool | tqdm, optional
            Display a progress bar or use provided progress bar.
        desc : str | None, optional
            Progress bar description if `progress` is `True`.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, ...]
            A tensor of embeddings or tuple of tensors if multiple inputs provided.
        """
        # Turn all inputs into batches
        inputs = [to_batch(x) for x in inputs]

        # Flatten inputs
        flattened = flatten(inputs)

        # Create batch iterator
        batch_size = batch_size or len(flattened)
        batches = itertools.batched(flattened, batch_size)

        # Create progress bar
        progress = progress if isinstance(progress, tqdm) else tqdm(total=len(flattened), disable=not progress, desc=desc)

        # Process batches
        embeddings = []

        for batch in batches:
            # Convert batch to list to maintain contents
            batch = list(batch)

            # Tokenize batch and get outputs
            tokens = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(self.device)
            outputs = self.model(**tokens)

            # Mean pooling + L2 normalization
            embeddings_batch = self._mean_pooling(outputs, tokens['attention_mask'])
            embeddings_norm = F.normalize(embeddings_batch, p=2, dim=1)

            # Add to embeddings
            embeddings.append(embeddings_norm)

            # Update the progress bar after completing batch
            progress.update(len(batch))

        # Concatenate all batch tensors
        embeddings_flat = torch.cat(embeddings, dim=0)

        # Reconstruct the nested lists
        embeddings = unflatten(embeddings_flat, inputs)

        # Squeeze output embeddings
        if len(embeddings) == 1:
            embeddings = embeddings[0]

        return embeddings

    def similarity(
        self,
        a: str | list[str] | torch.Tensor | torch.Tensor[torch.Tensor],
        b: str | list[str] | torch.Tensor | torch.Tensor[torch.Tensor],
        method: Literal['matrix', 'pairwise'] = 'matrix',
        encode: bool = True,
        **kwargs
    ) -> float | np.ndarray[float] | np.ndarray[np.ndarray[float]]:
        """
        Calculate cosine similarity between sequences or embeddings.

        Parameters
        ----------
        a : str | list[str] | torch.Tensor | torch.Tensor[torch.Tensor]
            First string or embedding sequence.
        b : str | list[str] | torch.Tensor | torch.Tensor[torch.Tensor]
            Second string or embedding sequence.
        method : Literal['matrix', 'pairwise'], optional
            'matrix' for matrix multiplication, 'pairwise' for element-wise similarity.
        encode : bool, optional
            Set True to encode sequences; False assumes embeddings are provided.

        Returns
        -------
        float | np.ndarray[float] | np.ndarray[np.ndarray[float]]
            Cosine similarity of sequences or embeddings.
        """    
        # Turn sequences into batches
        a = to_batch(a)
        b = to_batch(b)

        # Validate input sequences
        if encode and (isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor)):
            raise ValueError('To encode, input sequences must be strings or lists of strings.')

        if method == 'pairwise' and len(a) != len(b):
            raise ValueError('Pairwise method requires sequences of equal length.')

        # Compute embeddings
        a_emb, b_emb = self.encode(a, b, **kwargs) if encode else (a, b)

        # Compute cosine similarity
        match method:
            case 'matrix':
                # Calculate matrix multiplication of both embeddings
                scores = torch.mm(a_emb, b_emb.transpose(0, 1))

            case 'pairwise':
                # Find dot product between sequences
                scores = (a_emb * b_emb).sum(dim=-1)

        # Remove empty dimensions
        return scores.squeeze().numpy()