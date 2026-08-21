from __future__ import annotations

import polars as pl

from typing import Union, Dict, Any, List, Optional, Tuple

from mse.cdp.processing import Processor


class Workbook(dict):

    def __init__(
        self,
        data: Any,
        year: int,
        label: str,
        join: str,
        sheets: Optional[list[Union[list[int], int]]] = None,
        merges: Optional[Dict[str, Tuple[int, int]]] = None,
        redundant: Optional[list[str]] = None,
        drop: Optional[list[str]] = None,
        renames: Optional[Dict[str, str]] = None
    ) -> None:
        super().__init__(data)

        self.year = year
        self.label = label
        self.name = f'{year}_{label}'

        self.join = join
        self.sheets = sheets or []
        self.merges = merges or {}
        self.redundant = redundant or []
        self.drop = drop or []
        self.renames = renames or {}


class DataFrame(pl.DataFrame):

    def __init__(
        self,
        data: Union[Dict, pl.DataFrame],
        year: Optional[int] = None,
        label: Optional[str] = None,
        processor: Optional[Processor] = None
    ) -> None:
        super().__init__(data)

        self.year = year
        self.label = label
        self.name = f'{year}_{label}'

        self.processor = processor

    def __getattribute__(self, name: str) -> Any:
        # Intercept method calls that return DataFrames and wrap them in the subclass.
        attr = super().__getattribute__(name)

        if callable(attr):

            def wrapped_method(*args, **kwargs):
                result = attr(*args, **kwargs)
                # Ensure the subclass is used for DataFrame-returning methods
                if isinstance(result, pl.DataFrame) and not isinstance(result, DataFrame):
                    return DataFrame(result, year=self.year, label=self.label, processor=self.processor)

                return result

            return wrapped_method

        return attr

    @property
    def columns_(self) -> List[str]:
        # Property to compute and store sheets once.
        return [self.processor.get_column_property(column, 'column') for column in self.columns]

    @property
    def sheets(self) -> List[str]:
        # Property to compute and store sheets once.
        return [self.processor.get_column_property(column, 'sheets') for column in self.columns]