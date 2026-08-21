from __future__ import annotations

import gc
import torch
import more_itertools as itertools
import polars as pl
import numpy as np

from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from mse.utils.list import to_batch, flatten, unflatten
from mse.utils.io import write_parquet
from mse.cdp.processing import Processor


class _Classifier:

    def __init__(
        self,
        name: str,
        model_name: str,
        tokenizer_name: str,
        half: bool = True
    ) -> None:
        self.name = name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load classification model
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        
        # Use half precision for GPU inference
        if half and torch.cuda.is_available():
            self.model.half()
        
        # Set evaluation mode
        self.model.eval()
        
        # Prepare tokenizer
        self.tokenizer_name = tokenizer_name
        self._tokenizer = None
        
        # Get the model label configuration
        self.id2label = self.model.config.id2label
        self.labels = list(self.id2label.values())
    
    @property
    def tokenizer(self) -> AutoTokenizer:
        if not self._tokenizer:
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        return self._tokenizer

    def unload(self) -> None:
        """Explicitly unload model from GPU"""
        del self.model
        self._tokenizer = None


class Classifier:

    def __init__(self, models: tuple[str, str, str]) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.models = models

        # Create cache to store predictions: (model, text) -> (label, scores)
        self._cache = {}

    def _clear_cache(self) -> None:
        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Clear cache
        del self._cache
        self._cache = {}
        
    @torch.no_grad()
    @torch.inference_mode()
    def classify(
        self,
        texts: str | list[str],
        model: _Classifier,
        progress: bool = True,
        batch_size: int = 128
    ) -> tuple[list[str], list[dict[str, float]]]:
        # Convert texts to a batch
        texts = np.array(to_batch(texts), dtype=object)

        # Identify which texts are already in cache
        texts_cached = [text for (name, text) in self._cache if name == model.name]

        # Find unique texts
        unique = np.unique(texts[texts.astype(bool)])

        # Masks for unique inputs to process
        mask_filled = unique.astype(bool)
        mask_cached = np.isin(unique, texts_cached)
        mask_unique = mask_filled & ~mask_cached

        # Get unique inputs for inference
        unique_inputs = unique[mask_unique]
        unique_cached = unique[mask_cached]

        # Build label and score mappings from cache
        mapping_results = {(model.name, text): self._cache[(model.name, text)] for text in unique_cached}

        # Create batches progress bar
        progress = tqdm(total=len(unique_inputs), desc='Predicting texts', disable=not progress)

        # Process inputs in batches
        for batch in itertools.batched(unique_inputs, n=batch_size):
            batch_inputs = list(batch)

            # Encode batch inputs
            batch_encoded = model.tokenizer(
                batch_inputs,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            batch_encoded = {k: v.to(self.device) for k, v in batch_encoded.items()}
            
            # Run inference over tkoens
            batch_outputs = model.model(**batch_encoded)
            batch_scores = batch_outputs.logits.softmax(dim=1).cpu().numpy()
            
            # Process results
            batch_indices = np.argmax(batch_scores, axis=1)
            batch_labels = [model.id2label[i] for i in batch_indices]
            batch_scores = [dict(zip(model.labels, scores)) for scores in batch_scores]

            # Add predictions to mappings
            batch_mapping_results = {(model.name, text): (label, scores) for text, label, scores in zip(batch_inputs, batch_labels, batch_scores)}
            mapping_results.update(batch_mapping_results)
            
            # Update the progress bar
            progress.update(len(batch_inputs))

            # Clean up memory
            del batch_encoded, batch_outputs

        # Add results to cache
        self._cache.update(mapping_results)

        # Build output arrays
        mapping_labels = {text: label for (name, text), (label, scores) in mapping_results.items()}
        mapping_scores = {text: scores for (name, text), (label, scores) in mapping_results.items()}

        labels = [mapping_labels.get(text) for text in texts]
        scores = [mapping_scores.get(text) for text in texts]

        return labels, scores

    def classify_df(
        self,
        df: pl.DataFrame,
        processor: Processor,
        extends: bool = True,
        progress: bool = True,
        batch_size: int = 512,
        chunk_path: str | None = None,
        chunk_makedirs: bool = True
    ) -> pl.DataFrame:
        # Create output mapping
        output = []

        # Gather columns as lists
        col_lists = [col.to_list() for col in df]
        col_names = [col for col in df.columns]

        # Combine texts across the columns
        texts_flat = flatten(col_lists)

        # Clear cache and memory first
        self._clear_cache()

        # Process texts using each model
        for i, model in enumerate(self.models):

            # Gather data for this model
            data = {}

            # Create model only when needed
            model = _Classifier(*model)

            # Run predictions
            print(f'Model {i + 1}/{len(self.models)}: {model.name}')
            labels_flat, scores_flat = self.classify(texts_flat, model=model, batch_size=batch_size, progress=progress)

            # Reconstruct nested lists
            labels_nested = unflatten(labels_flat, col_lists)
            scores_nested = unflatten(scores_flat, col_lists)

            # Add predictions for each column
            progress_insert = tqdm(total=len(col_names), desc='Inserting columns')
            
            for col_name, labels, score_mappings in zip(col_names, labels_nested, scores_nested):

                # Add predicted labels
                col = processor.update_column(col_name, {'model': model.name, 'value': 'label'})
                data[col] = labels
                
                # Flatten scores and reconstruct
                score_mappings_flat = flatten(score_mappings)

                # Add scores for each label
                for label in model.labels:
                    scores_flat = [m[label] if m else None for m in score_mappings_flat]
                    scores = unflatten(scores_flat, labels)

                    col = processor.update_column(col_name, {'model': model.name, 'label': label, 'value': 'score'})
                    data[col] = scores

                progress_insert.update(1)

            print('Converting data to dataframe → ', end='')
            # Convert data to a polars dataframe
            data = pl.DataFrame(data, schema={col: pl.List(pl.String) for col in data}, strict=False)

            # Add data to output data
            output.append(data)

            print('Writing chunked data → ', end='')
            # Write chunk to parquet
            if chunk_path:
                write_parquet(data, chunk_path.format(model.name), makedirs=chunk_makedirs)

            # Unload the model to free GPU memory
            model.unload()

            print('Clearing cache\n')
            # Clear memory
            self._clear_cache()

        # Add original dataframe if specified
        if extends:
            output = [df, *output]

        # Convert output data to dataframe
        output = pl.concat(output, how='horizontal')

        return output