from __future__ import annotations

import torch
import more_itertools as itertools
import polars as pl
import numpy as np

from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from fast_langdetect import detect_language

from mse.utils.list import to_batch, flatten, unflatten
from mse.cdp.processing import Processor


class Translator:

    def __init__(
        self,
        target: str = 'en',
        model_name: str = 'alirezamsh/small100',
    ) -> None:
        """
        Parameters
        ----------
        target : str, optional
            The ISO 639-1 language code representing the language to translate input text to.
        model_name : str, optional
            The name of the model to use.
        """
        self.target = target
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load the model with optimizations
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        
        if torch.cuda.is_available():
            self.model.device_map = 'auto',
            self.model.low_cpu_mem_usage = True,
            self.model.quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
            # Compile the model
            self.model = torch.compile(self.model)
            
        # Lazy-load the tokenizer
        self._tokenizer = None

        # Cache translation results
        self._cache = {}

    @property
    def tokenizer(self) -> AutoTokenizer:
        if not self._tokenizer:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, tgt_lang=self.target)

        return self._tokenizer

    def detect(self, text: str) -> str:
        if not text:
            return self.target
        
        # Get detected language of text
        text = text.replace('\n', ' ')[:100]
        lang = detect_language(text, low_memory=False).lower()

        # Fall back to target if no language detected
        if not lang:
            return self.target

        return lang

    @torch.no_grad()
    def translate(
        self,
        texts: str | list[str],
        progress: bool = True,
        batch_size: int = 64
    ) -> tuple[list[str], list[str]]:
        # Convert texts to a batch
        texts = np.array(to_batch(texts), dtype=object)

        # Find unique texts and languages
        unique = np.unique(texts[texts.astype(bool)])
        unique_langs = np.array([self.detect(text) for text in tqdm(unique, desc='Detecting texts', disable=not progress)])
        # Masks for unique inputs to process
        mask_trans = unique.astype(bool) & (unique_langs != self.target)
        mask_cached = np.isin(unique, list(self._cache))
        mask_unique = mask_trans & ~mask_cached

        # Get unique inputs to translate
        unique_inputs = unique[mask_unique]
        unique_cached = unique[mask_cached]

        # Build translations mapping
        mapping_trans = {text: self._cache[text] for text in unique_cached}
        mapping_langs = {text: lang for text, lang in zip(unique, unique_langs)}

        # Clear memory
        del unique_langs
        del mask_trans
        del mask_cached
        del mask_unique
        del unique_cached

        # Create batches progress bar
        progress = tqdm(total=len(unique_inputs), desc='Translating texts', disable=not progress)

        # Process inputs in batches
        for batch in itertools.batched(unique_inputs, n=batch_size):
            batch_inputs = list(batch)

            # Tokenize and translate the batch inputs
            batch_encodings = self.tokenizer(batch_inputs, return_tensors='pt', padding=True, truncation=True)
            batch_encodings = {k: v.to(self.device) for k, v in batch_encodings.items()}
            
            batch_sequences = self.model.generate(**batch_encodings, num_beams=1)
            batch_trans = self.tokenizer.batch_decode(batch_sequences, skip_special_tokens=True)

            # Add translations to mapping
            batch_mapping_trans = {text: trans for text, trans in zip(batch_inputs, batch_trans)}
            mapping_trans.update(batch_mapping_trans)

            # Update the progress bar
            progress.update(len(batch_inputs))

        # Add translations to cache
        self._cache.update(mapping_trans)

        # Build translations array
        trans = [mapping_trans.get(text, text) for text in texts]
        langs = [mapping_langs.get(text, self.target) for text in texts]

        return trans, langs

    def translate_df(
        self,
        df: pl.DataFrame,
        processor: Processor,
        extends: bool = True,
        detected: bool = True,
        progress: bool = True,
        batch_size: int = 512
    ) -> pl.DataFrame:
        # Flatten input columns
        col_names = df.columns
        cols = [col.to_list() for col in df]

        inputs_flat = flatten(cols)

        # Translate combined texts
        trans_flat, langs_flat = self.translate(inputs_flat, batch_size=batch_size)

        # Reconstruct flattened arrays into nested arrays
        trans_nested = unflatten(trans_flat, cols)
        langs_nested = unflatten(langs_flat, cols)

        # Create progress bar
        progress = tqdm(zip(col_names, cols, trans_nested, langs_nested), desc='Adding columns', disable=not progress)

        # Add results to output dataframe
        output = {}
        for col_name, col, col_trans, col_langs in progress:
            
            # Create column names
            col_trans_name = processor.update_column(col_name, {'translated': 'true'}) if extends else col_name
            col_langs_name = processor.update_column(col_name, {'value': 'response_language'})

            # Add translations and languages
            output[col_trans_name] = pl.Series(col_trans, strict=False).cast(pl.List(pl.String))

            if detected:
                output[col_langs_name] = pl.Series(col_langs, strict=False).cast(pl.List(pl.String))
        
        # Convert output to dataframe
        output = pl.DataFrame(output)

        if extends:
            output = pl.concat((df, output), how='horizontal')
        
        return output



