from __future__ import annotations

import polars as pl
import re

from frozendict import frozendict
from tqdm.auto import tqdm
from typing import Sequence

import mse.utils.cdp as cdp


class Processor:

    def __init__(
        self,
        merger,
        column_property_join: str = '=',
        column_property_separator: str = '|',
        column_suffix_special: str = '*'
    ) -> None:
        """
        Initialize the Processor with necessary configurations.

        Parameters
        ----------
        merger : Merger
            An instance of the Merger class used for merging dataframes.
        column_id_separator : str, optional
            The string used to separate column identifiers, by default '__'.
        column_suffix_special : str, optional
            The suffix added to the join column, by default '*'.
        """
        self.merger = merger

        self.column_property_join = column_property_join
        self.column_property_separator = column_property_separator
        self.column_suffix_special = column_suffix_special

    # | Columns
    # |-------------------
    def create_column(self, props: dict[str, str]) -> str:
        props = [self.column_property_join.join((k, v)) for k, v in props.items()]
        return self.column_property_separator.join(props)

    def create_column_standard(self, column: str, sheet: str, value: str = 'response') -> str:
        return self.create_column({'column': column, 'sheet': sheet, 'value': value})

    def create_column_classification_label(self, column: str, sheet: str, model: str, value: str = 'label') -> str:
        return self.create_column({'column': column, 'sheet': sheet, 'model': model, 'value': value})

    def create_column_classification_score(self, column: str, sheet: str, model: str, label: str, value: str = 'score') -> str:
        return self.create_column({'column': column, 'sheet': sheet, 'model': model, 'label': label, 'value': value})


    def get_column_properties(self, col: str) -> frozendict:
        props = [pair.split(self.column_property_join, 1) for pair in col.split(self.column_property_separator)]
        return frozendict(props)

    def get_column_property(self, col: str, key: str, default: str | None = None) -> str:
        props = self.get_column_properties(col)
        return props.get(key, default)

    def filter_columns(self, cols: list[str], props: dict[str, str]) -> list[str]:
        return [col for col in cols if all(self.get_column_property(col, key) == value for key, value in props.items())]

    def update_column(self, col: str, props: dict[str, str]) -> str:
        # Merge old properties with new properties
        current_props = self.get_column_properties(col)
        new_props = dict(current_props) | dict(props)

        # Move the value property to the end
        value = new_props.pop('value')
        new_props['value'] = value
        return self.create_column(new_props)


    def _clean_column_name(self, col: str) -> str:
        """
        Cleans a column identifier by removing unnecessary whitespace.

        Parameters
        ----------
        col : str
            The column identifier to be cleaned.

        Returns
        -------
        str
            The cleaned column identifier.
        """
        clean = (
            ' '.join(col.split())
            .strip()
            .replace('\u2028', '')
            .replace('\u2029', '')
        )
        return clean

    # | Workbooks
    # |-------------------
    def _filter_workbook(
        self,
        workbook: dict[str, pl.DataFrame],
        sheet_numbers: list[int | list[int]]
    ) -> dict[str, pl.DataFrame]:
        """
        Filters a workbook by the specified sheet ranges.

        Parameters
        ----------
        workbook : dict[str, pl.DataFrame]
            The workbook to be filtered.
        sheet_numbers : list[int | list[int]]
            A list of sheet numbers or ranges to be filtered.

        Returns
        -------
        dict[str, pl.DataFrame]
            The filtered workbook.
        """
        # Filters the workbook by the specified sheet ranges.
        filtered = {}
        sheet_names = list(workbook)
        for e in sheet_numbers:
            start, end = (e, e) if isinstance(e, int) else e
            for i in range(start, end + 1):
                sheet_name = sheet_names[i]
                filtered[sheet_name] = workbook[sheet_name]

        return filtered

    def _join_workbook(
        self,
        workbook: dict[str, pl.DataFrame],
        join_column: str
    ) -> pl.DataFrame:
        """
        Consolidates a tabbed workbook horizontally based on a join column.

        Parameters
        ----------
        workbook : dict[str, pl.DataFrame]
            A dictionary of dataframes, where each dataframe represents a sheet in a workbook.
        join_column : str
            The column to perform the join on.

        Returns
        -------
        pl.DataFrame
            The consolidated dataframe, where fields of rows with the same join column are
            aggregated into lists. All column names are appended with sheet names to avoid
            uniqneness conflicts, and the join column also has a suffix for consistency.
        """
        join_column_new = self.create_column({'column': join_column, 'sheet': self.column_suffix_special, 'value': 'response'})
        
        output = pl.DataFrame()
        for sheet_name, sheet in workbook.items():
            # Clean the sheet name
            sheet_name = self._clean_column_name(sheet_name)

            # Skip this sheet if no join column
            if join_column not in sheet.columns:
                continue

            # Build renames
            renames = {}
            for column in sheet.columns:
                if column == join_column:
                    new_column = join_column_new
                else:
                    # Clean the column name
                    new_column = self._clean_column_name(column)
                    new_column = self.create_column({'column': new_column, 'sheet': sheet_name, 'value': 'response'})
                
                renames[column] = new_column

            # Rename the columns
            sheet = (
                sheet
                .cast(pl.String)
                .group_by(join_column, maintain_order=True).agg(pl.all())
                .rename(renames)
            )

            # Join with the output dataframe
            output = (
                sheet if output.is_empty() else
                output.join(sheet, how='full', on=join_column_new, coalesce=True)
            )

        return output.cast(pl.List(pl.String))

    def process_workbook(
        self,
        workbook: cdp.Workbook
    ) -> cdp.DataFrame:
        """
        Processes a workbook by filtering, joining, cleaning, renaming, and dropping columns.

        Parameters
        ----------
        workbook : cdp.Workbook
            The workbook to be processed.

        Returns
        -------
        cdp.DataFrame
            The processed dataframe.
        """
        wb = self._filter_workbook(workbook, workbook.sheets)
        df = self._join_workbook(wb, workbook.join)
        df = self._clean_fields(df)
        df = self._rename_columns(df, workbook.renames)
        df = self._drop_columns(df, workbook.redundant, keep_first=True)
        df = self._drop_columns(df, workbook.drop, keep_first=False)
        # df = self._merge_sections(df, workbook.merges, list(workbook), workbook.redundant)

        # Create new column labels
        year = self.create_column({'column': 'year', 'sheet': '*', 'value': 'response'})
        label = self.create_column({'column': 'label', 'sheet': '*', 'value': 'response'})

        # Add year and label columns
        df = (
            df
            .with_columns(pl.lit(workbook.year).alias(year), pl.lit(workbook.label).alias(label))
            .select(year, label, pl.exclude(year, label))
            .cast(pl.List(pl.String))
        )

        # Convert to a custom dataframe
        df = cdp.DataFrame(df, year=workbook.year, label=workbook.label, processor=self)
        return df

    def process_workbooks(
        self,
        workbooks: Sequence[cdp.Workbook],
        progress: bool = True
    ) -> Sequence[cdp.DataFrame]:
        """
        Processes a sequence of workbooks into dataframes.

        Parameters
        ----------
        workbooks : Sequence[cdp.Workbook]
            A sequence of workbooks to be processed.
        progress : bool, optional
            If True, displays a progress bar during processing, by default True.

        Returns
        -------
        Sequence[cdp.DataFrame]
            A sequence of processed dataframes corresponding to the input workbooks.
        """
        dfs = []
        with tqdm(workbooks, disable=not progress, desc='Workbooks processed', leave=True) as progress:
            for workbook in progress:
                df = self.process_workbook(workbook)
                dfs.append(df)

        return dfs

    # | Dataframes
    # |-------------------
    def _clean_column_names(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Cleans the column names of a dataframe.

        Parameters
        ----------
        df : pl.DataFrame
            The dataframe to be cleaned.

        Returns
        -------
        pl.DataFrame
            The dataframe with cleaned column names.
        """
        df = df.rename(lambda column_name: self._clean_column_name(column_name))
        return df

    def _clean_fields(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Cleans the fields of a dataframe.

        Parameters
        ----------
        df : pl.DataFrame
            The dataframe to be cleaned.

        Returns
        -------
        pl.DataFrame
            The dataframe with cleaned fields.
        """
        # Cleans the fields of a dataframe
        df = df.with_columns(
            pl.all().list.eval(
                pl.when(pl.element().is_null())
                .then(None)
                .otherwise(
                    pl.element()
                    .str.strip_chars()
                    .str.replace_all(r'\s+', ' ')
                    .str.replace_all(r'[\u2028-\u2029]+', '')
                )
            )
        )
        return df

    def _rename_columns(
        self,
        df: pl.DataFrame,
        mapping: dict[str, str],
        strict: bool = False
    ) -> pl.DataFrame:
        """
        Renames columns in the DataFrame based on a given mapping.

        Parameters
        ----------
        df : pl.DataFrame
            The DataFrame whose columns are to be renamed.
        mapping : dict[str, str]
            A dictionary mapping existing column names to new column names.
        strict : bool, optional
            If True, only columns present in the DataFrame are renamed; otherwise,
            columns not present in the DataFrame are ignored without raising an error,
            by default False.

        Returns
        -------
        pl.DataFrame
            The DataFrame with renamed columns.
        """
        revised = mapping.copy()
        if not strict:
            for original in mapping:
                if original not in df.columns:
                    revised.pop(original)

        df = df.rename(revised)
        return df

    def _drop_columns(
        self,
        df: pl.DataFrame,
        to_drop: Sequence[str],
        keep_first: bool = False
    ) -> pl.DataFrame:
        """_summary_

        Parameters
        ----------
        df : pl.DataFrame
            _description_
        to_drop : Sequence[str]
            _description_
        keep_first : bool, optional
            _description_, by default False

        Returns
        -------
        pl.DataFrame
            _description_
        """    
        # If keep_first, remove the explicitly provided columns such that, for each
        # column, we keep the first occurrence but drop all subsequent occurrences.
        # The fields in to_drop should be provided in regex.
        # Map patterns to the columns they match.
        # Drop completely null columns.
        matched = []
        for pattern in to_drop:
            matches = []
            pattern = re.compile(pattern, flags=re.IGNORECASE)

            for column_id in df.columns:
                match = pattern.match(column_id)
                if match:
                    matches.append(column_id)

            matched.append(matches)

        to_drop = []
        for column_ids in matched:
            if keep_first:
                column_ids = column_ids[1:]
            to_drop.extend(column_ids)

        df = (
            df.drop(to_drop)
            .drop(pl.selectors.contains('__UNNAMED__', 'column_'))
            .drop(column.name for column in df if column.is_null().all()))

        return df

    def _merge_sections(
        self,
        df: pl.DataFrame,
        merges: dict[str, tuple[int, int]],
        sheets: list[str],
        redundant: list[str]
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """_summary_

        Parameters
        ----------
        df : pl.DataFrame
            _description_
        merges : Dict[str, tuple[int, int]]
            _description_
        sheets : list[str]
            _description_
        redundant : list[str]
            _description_

        Returns
        -------
        tuple[pl.DataFrame, pl.DataFrame]
            _description_
        """        
        # Merge the two sheet ranges with each other and removes redundant columns for each slice.

        # Create label for chunk
        chunk = self.create_column({'column': 'chunk', 'sheet': '*'})

        # Process dataframe
        if merges and sheets:

            slices = []
            for label, span in merges.items():
                # Get the sheets section
                sheets_ = sheets[span[0] : span[1] + 1]

                # Get the slice from the sheets
                sliced = df.select(col for col in df.columns if self.get_column_property(col, 'sheet') in sheets_)
                sliced = cdp.DataFrame(sliced, label=label, processor=self)

                # Drop redundant columns
                sliced = self._drop_columns(sliced, redundant, keep_first=True)

                # Remove completely null rows
                sliced = sliced.filter(~pl.all_horizontal(pl.all().is_null()))

                # Add slice label column
                sliced = (
                    sliced.with_columns(pl.lit(label).alias(chunk))
                    .select(chunk, pl.exclude(chunk))
                    .cast(pl.List(pl.String)))

                slices.append(sliced)

            df, mapping = self.merger.merge(slices)
        
        else:
            # Drop redundant columns
            df = self._drop_columns(df, redundant, keep_first=True)

            # Fill added column with None
            df, _ = df.with_columns(pl.lit(None).alias(chunk)), None

        return df