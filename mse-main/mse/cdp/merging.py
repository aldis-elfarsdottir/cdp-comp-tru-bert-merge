from __future__ import annotations

import os
import torch
import itertools
import numpy as np
import polars as pl
import networkx as nx

from dataclasses import dataclass
from tqdm.auto import tqdm
from typing import Sequence, Any, Literal
from sklearn.cluster import KMeans

import mse.utils.cdp as cdp
from mse.utils.list import flatten, unflatten
from mse.cdp.similarity import SimilarityModel


@dataclass
class Fingerprint:
    columns: list[str]
    column_embs: torch.Tensor
    context_embs: torch.Tensor
    sample_embs: list[torch.Tensor]


class Merger:

    def __init__(
        self,
        model: SimilarityModel,
        w_D_col: float = 1.0,
        w_S_col: float = 1.0,
        w_S_field: float = 1.0,
        w_S_context: float = 1.0,
        n_fields: int = 3,
        win_S_context: int = 4,
        thresh_C_method: Literal['fixed', 'cluster', 'spread'] = 'fixed',
        thresh_C: float = 1.0,
        thresh_C_z_factor: float = 1.0
    ) -> None:
        # Set thresh_C to 1.0 to allow everything to be matched
        # Similarity model for cost calculation
        # Weights for matrix components of cost matrix
        # Parameters for calculating the cost matrix
        # Define the threshold.
        # The parameters thresh_C, z_factor are only used when the correct
        # thresh_C_method is specified

        # Disable tokenizers parallelism
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        self.model = model

        self.w_D_col = w_D_col
        self.w_S_col = w_S_col
        self.w_S_field = w_S_field
        self.w_S_context = w_S_context

        self.n_fields = n_fields
        self.win_S_context = win_S_context

        self.thresh_C_method = thresh_C_method
        self.thresh_C = thresh_C
        self.thresh_C_z_factor = thresh_C_z_factor

    # | Helpers
    # |-------------------
    @staticmethod
    def _normalize(A: np.ndarray) -> np.ndarray:
        min_A = np.min(A)
        spread = np.max(A) - min_A or np.inf
        A_norm = (A - min_A) / spread
        return A_norm

    @staticmethod
    def _sample(a: list[list | Any], n: int) -> list[Any]:
        # Sample all possible values in a uniform manner with replacement
        # Expand list elements resulting from the process of
        # joining the sheets of a workbook
        # Convert empty string fields to null so they get dropped
        # Drop null entries to only get usable values
        # Convert all values to strings to make unique work
        # Only get unique values to ensure broader coverage.
        # Maintain order to ensure that the unique method
        # returns the same values for each run.
        # Randomly sample from the series with replacement, and
        # set a constant seed to make sure sampling is deterministic.        
        samples = (
            pl.Series(a)
            .explode()
            .drop_nulls()
            .unique(maintain_order=True)
            .sample(n, shuffle=True, with_replacement=True, seed=0)
            .to_list()
        )
        return samples

    @staticmethod
    def _contexts(
        series: list[torch.Tensor | Any],
        window: int,
        groups: list[Any] = None,
        mean: bool = False
    ) -> list[torch.Tensor | list[Any]]:
        # i = position of the column from which to get context
        # If groups, remove the elements from the context window that do not
        # fall in the same group as the current element.
        # Average should only be used if each element in a is torch.Tensor
        contexts = []

        for i, current in enumerate(series):
            # Calculate window slices
            l_window = (max(i - window, 0), i + 1)
            r_window = (i + 1, i + window + 1)

            # Gather both sides of context window
            l_context = series[l_window[0] : l_window[1]]
            r_context = series[r_window[0] : r_window[1]]

            # Combine into single context
            context = l_context + r_context

            # Filter the context window for groups
            if groups:
                # Gather both sides of groups window
                l_groups = groups[l_window[0] : l_window[1]]
                r_groups = groups[r_window[0] : r_window[1]]

                # Combine into single groups
                context_groups = l_groups + r_groups

                # Get the current group
                current_group = groups[i]

                # Select elements whose groups match the current group
                context = [x for x, group in zip(context, context_groups) if group == current_group]

            # Fall back to current element if no context
            if not context:
                context = [current]
            
            # Average embeddings to create single embedding
            if mean:
                context = torch.stack(context, dim=0).mean(dim=0)

            contexts.append(context)

        return contexts

    @staticmethod
    def _make_unique(a: Sequence[str]) -> pl.Series:
        # Make any repeated values unique
        seen = {}
        unique = []
        for value in a:
            if value not in seen:
                seen[value] = 0
                unique.append(value)
            else:
                seen[value] += 1
                unique.append(f'{value}_{seen[value]}')
        
        return pl.Series(unique)

    # | Metrics
    # |-------------------
    def _build_D_col(
        self,
        columns_A: list[str],
        columns_B: list[str]
    ) -> np.ndarray:
        D_col = self.model.distance(columns_A, columns_B, method='matrix')
        return D_col

    def _build_S_col(
        self,
        column_embs_A: list[torch.Tensor],
        column_embs_B: list[torch.Tensor]
    ) -> np.ndarray:
        S_col = self.model.similarity(column_embs_A, column_embs_B, method='matrix', encode=False)
        return S_col

    def _build_S_context(
        self,
        context_embs_A: list[torch.Tensor],
        context_embs_B: list[torch.Tensor]
    ) -> np.ndarray:
        S_context = self.model.similarity(context_embs_A, context_embs_B, method='matrix', encode=False)
        return S_context

    def _build_S_field(
        self,
        sample_embs_A: list[torch.Tensor[torch.Tensor]],
        sample_embs_B: list[torch.Tensor[torch.Tensor]]
    ) -> np.ndarray:
        # Compute the similarity matrix
        S_field = np.zeros((len(sample_embs_A), len(sample_embs_B)))

        for i, embs_A in enumerate(sample_embs_A):
            for j, embs_B in enumerate(sample_embs_B):
                if embs_A.numel() and embs_B.numel():
                    scores = self.model.similarity(embs_A, embs_B, method="pairwise", encode=False)
                    
                    # Take simple average of scores
                    score = scores.mean().item()
                    S_field[i, j] = score

        return S_field

    # | Embeddings
    # |-------------------
    def _build_fingerprint(
        self,
        A: cdp.DataFrame,
        progress: tqdm = None
    ) -> Fingerprint:
        cols = A.columns_

        samples = [self._sample(col, self.n_fields) for col in A]
        samples_flat = flatten(samples)

        # Encode column names and samples with progress tracking
        col_embs = self.model.encode(cols, progress=progress, batch_size=64)
        sample_embs_flat = self.model.encode(samples_flat, progress=progress, batch_size=64)

        # Generate context embeddings from column embeddings
        context_embs = self._contexts(col_embs, self.win_S_context, groups=A.sheets, mean=True)

        # Reconstruct nested sample embeddings
        sample_embs = unflatten(sample_embs_flat, samples)

        # Convert all embeddings to torch tensors
        col_embs = torch.stack(col_embs)
        context_embs = torch.stack(context_embs)
        sample_embs = [torch.stack(x) if x else torch.tensor([]) for x in sample_embs]

        # Create fingerprint
        fingerprint = Fingerprint(
            columns=cols,
            column_embs=col_embs,
            context_embs=context_embs,
            sample_embs=sample_embs)

        return fingerprint

    # | Cost matrix
    # |-------------------
    def _build_cost_matrix(
        self,
        fp_A: Fingerprint,
        fp_B: Fingerprint
    ) -> np.ndarray:
        # Compute distance and similarity matrices
        D_col = self._build_D_col(fp_A.columns, fp_B.columns)
        S_col = self._build_S_col(fp_A.column_embs, fp_B.column_embs)
        S_context = self._build_S_context(fp_A.context_embs, fp_B.context_embs)
        S_field = self._build_S_field(fp_A.sample_embs, fp_B.sample_embs)

        # Lower similarity scores should correspond to higher costs
        S_col = -S_col
        S_context = -S_context
        S_field = -S_field

        # Normalize matrices
        D_col_norm = self._normalize(D_col)
        S_col_norm = self._normalize(S_col)
        S_context_norm = self._normalize(S_context)
        S_field_norm = self._normalize(S_field)

        # Compute weighted sum
        C = (
            self.w_D_col * D_col_norm
            + self.w_S_col * S_col_norm
            + self.w_S_context * S_context_norm
            + self.w_S_field * S_field_norm
        )

        # Normalize cost matrix
        C_norm = self._normalize(C)
        return C_norm

    # | Threshold
    # |-------------------
    @staticmethod
    def _threshold_cluster(C: np.array) -> float:
        # Assumes the threshold is the maximum value of the good cluster,
        # given that the good cluster is the cluster with the smallest center
        C = C.reshape(-1, 1)

        # Use K-means to find cluster centers
        kmeans = KMeans(n_clusters=2, random_state=0).fit(C)
        centers = kmeans.cluster_centers_.flatten()

        # Find the good cluster
        label_0 = np.argmin(centers)

        # Find the threshold
        threshold = C[kmeans.labels_ == label_0].max().item()
        return threshold

    @staticmethod
    def _threshold_spread(C: np.array, z_factor: float = 1.0) -> float:
        C = C.flatten()

        # Compute statistics
        mean = np.mean(C)
        std = np.std(C)

        # Calculate threshold
        threshold = mean + z_factor * std
        return threshold

    def _threshold(self, C: np.array) -> float:
        match self.thresh_C_method:
            case 'fixed':
                return self.thresh_C
            case 'cluster':
                return self._threshold_cluster(C)
            case 'spread':
                return self._threshold_spread(C, self.thresh_C_z_factor)

    # | Mapping
    # |-------------------
    def _map(
        self,
        A: pl.DataFrame,
        fp_A: Fingerprint,
        B: pl.DataFrame,
        fp_B: Fingerprint
    ) -> pl.DataFrame:
        # Build the cost matrix
        C = self._build_cost_matrix(fp_A, fp_B)

        # Create a bipartite graph
        G = nx.Graph()

        # Add column names as nodes
        node_map = {}

        for i, col_A in enumerate(A.columns):
            node_A = f'A_{i}'
            node_map[node_A] = col_A
            G.add_node(node_A, label=col_A, bipartite=0)

        for j, col_B in enumerate(B.columns):
            node_B = f'B_{j}'
            node_map[node_B] = col_B
            G.add_node(node_B, label=col_B, bipartite=1)

        # Connect columns from A to B with costs
        for i, col_A in enumerate(A.columns):
            for j, col_B in enumerate(B.columns):
                G.add_edge(f'A_{i}', f'B_{j}', weight=C[i, j])

        # Run Hungarian algorithm
        matches = nx.algorithms.bipartite.minimum_weight_full_matching(G)

        # Create node mapping
        mapping = {'A': [], 'B': [], 'cost': []}

        # Collect matched nodes
        matched_A = set()
        matched_B = set()

        # Calculate the cost threshold
        threshold = self._threshold(C)

        # Process matched nodes
        for node in matches:
            if node.startswith('A'):
                node_A = node
                node_B = matches[node_A]

                cost = G[node_A][node_B]['weight']
                if cost <= threshold:
                    col_A = node_map[node_A]
                    col_B = node_map[node_B]
                    
                    mapping['A'].append(col_A)
                    mapping['B'].append(col_B)
                    mapping['cost'].append(cost)

                    matched_A.add(col_A)
                    matched_B.add(col_B)

        # Handle unmatched nodes in A
        for col_A in A.columns:
            if col_A not in matched_A:
                mapping['A'].append(col_A)
                mapping['B'].append(None)
                mapping['cost'].append(None)

        # Handle unmatched columns in B
        for col_B in B.columns:
            if col_B not in matched_B:
                mapping['A'].append(None)
                mapping['B'].append(col_B)
                mapping['cost'].append(None)

        # Remove completely null rows
        mapping = pl.DataFrame(mapping)
        mapping = mapping.filter(~pl.all_horizontal(pl.all().is_null()))
        return mapping

    def map(
        self,
        dfs: Sequence[cdp.DataFrame],
        costs: bool = False
    ) -> pl.DataFrame:

        if len(dfs) < 2:
            raise ValueError('The number of dataframes must be at least 2')
        if len(set(df.name for df in dfs)) < len(dfs):
            raise ValueError('The dataframe names must be unique')
        
        # Compute the total work for the progress bar
        total = 0
        for df in dfs:
            cols = df.columns_
            samples = [self._sample(col, self.n_fields) for col in df]
            samples_flat = flatten(samples)
            total += len(cols) + len(samples_flat)

        # Build the progress bar
        with tqdm(total=total, desc='Encoding dataframes', leave=False) as progress:

            # Build the fingerprints
            fingerprints = []

            # Encode each dataframe
            for df in dfs:
                fingerprint = self._build_fingerprint(df, progress=progress)
                fingerprint = (df, fingerprint)
                fingerprints.append(fingerprint)

        # Build mapping chain
        chain = pl.DataFrame()

        with tqdm(list(itertools.pairwise(fingerprints)), desc='Mapping dataframes', leave=False) as progress:

            # Loop through each pair
            for (A, fp_A), (B, fp_B) in progress:
                # Map the two dataframes
                mapping = self._map(A, fp_A, B, fp_B)
                mapping = mapping.rename({'A': A.name, 'B': B.name, 'cost': f'{A.name}__{B.name}__cost'})

                chain = mapping if chain.is_empty() else chain.join(mapping, how='full', on=A.name, coalesce=True, join_nulls=False)

        # Remove cost columns
        if not costs:
            chain = chain.drop(pl.selectors.ends_with('__cost'))

        # Remove completely null mappings
        return chain.filter(~pl.all_horizontal((~pl.selectors.ends_with('__cost')).is_null()))

    # | Merging
    # |-------------------
    def merge(
        self,
        dfs: Sequence[cdp.DataFrame],
        mapping: pl.DataFrame = None
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        # Create the mapping
        mapping = self.map(dfs) if mapping is None else mapping

        if not all(df.name in mapping.columns for df in dfs):
            raise ValueError('The mapping must have the dataframe names as columns')

        # Coalesce mapping values from left to right
        # ┌───────┬───────┐     ┌───────┐
        # │ A     │ B     │     │ B     │
        # │═══════╪═══════│     │═══════│
        # │ A1    │ B1    │     │ B1    │
        # │ A2    │ Null  │ --> │ A2    │
        # │ Null  │ B3    │     │ B3    │
        # └───────────────┘     └───────┘
        mapping_coalesced = (
            mapping
            .select(mapping.columns[::-1])
            .with_columns(pl.coalesce(pl.all()).alias('_'))
            .select(pl.col('_'))
            .to_series()
        )

        # Prevent column name conflicts
        mapping_coalesced = self._make_unique(mapping_coalesced)

        # Vertical stack the dataframes
        merged = pl.DataFrame()

        for df in dfs:
            mapping_column = mapping.get_column(df.name)
            column_mapping = dict(zip(mapping_column, mapping_coalesced))
            df = df.rename(lambda old: column_mapping[old])

            merged = pl.concat((merged, df), how='diagonal_relaxed')

        # Cast to standard type
        merged = merged.cast(pl.List(pl.String))

        # Drop completely null columns
        merged = merged.drop(column.name for column in merged if column.is_null().all())

        return merged, mapping