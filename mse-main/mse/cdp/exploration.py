from __future__ import annotations

import re
import polars as pl

from collections import defaultdict, Counter

from mse.utils.cdp import Workbook


class Exploration:

    def __init__(
        self,
        bin: float | None = None,
        min: float | None = None
    ) -> None:
        self.bin = bin
        self.min = min

        if not (bin or min):
            raise ValueError('bin or min must be provided')

    def duplicates(self, workbooks: list[Workbook]) -> pl.DataFrame:
        # Finds any columns that have the same column name for each sheet in each workbook.
        duplicates = defaultdict(list)
        for A in workbooks:
            for sheet_name, sheet in A.items():

                columns = []
                for column in sheet.columns:
                    # Clean up the column to facilitate matching
                    pattern = r'(\s+)|(_duplicated_\d+$)'
                    column = re.sub(pattern, '', column)
                    columns.append(column)

                # Add the duplicate column names
                for column in columns:
                    if columns.count(column) > 1:
                        duplicates['name'].append(A.name)
                        duplicates['sheet'].append(sheet_name)
                        duplicates['column'].append(column)
            
            duplicates = pl.DataFrame(duplicates)
            return duplicates

    def redundant(self, workbooks: list[Workbook]) -> pl.DataFrame:
        # Finds the columns repeated across bin proportion of sheets
        output = defaultdict(list)
        for A in workbooks:
            num_sheets = len(A)

            columns = []
            for sheet in A.values():
                for column in sheet.columns:
                    column = ' '.join(column.split()).strip()
                    if '__UNNAMED__' not in column:
                        columns.append(column)
            
            threshold = self.bin * num_sheets if self.bin else self.min
            
            counts = Counter(columns)
            for column, count in counts.items():
                if count >= threshold:
                    output[A.name].append(column)
                    output[f'{A.name}__count'].append(count)

        height = max(len(column) for column in output.values())
        for name, column in output.items():
            output[name] += [None] * (height - len(column))

        return pl.DataFrame(output)

    def mismatches(self, workbooks: list[Workbook]) -> pl.DataFrame:
        # Finds the column names that are only in one workbook
        n = len(workbooks)
        mismatches = defaultdict(list)

        for i in range(n):
            for j in range(i + 1, n):
                A = workbooks[i]
                B = workbooks[j]
                cols_A = A.columns
                cols_B = B.columns

                mismatches = []
                for col_A in cols_A:
                    if cols_A not in cols_B:
                        col = f'{A.name} -> {B.name}'
                        mismatches[col].append(col_A)
        
        mismatches = pl.DataFrame(mismatches)
        return mismatches

    def sheets(self, workbooks: list[Workbook]) -> pl.DataFrame:
        # Enumerates the sheet names for each provided workbook
        height = max(len(A) for A in workbooks)
        output = pl.DataFrame(pl.int_range(0, height, eager=True).alias('#'))

        for A in workbooks:
            sheets = pl.Series(list(A))
            padding = pl.repeat(None, n=height - sheets.len(), eager=True)
            sheets = sheets.append(padding)

            output = output.with_columns(sheets.alias(A.name))
        
        return output

    # List the unique column names inside of each workbook
    def columns(self, workbooks: list[Workbook]) -> pl.DataFrame:
        output = defaultdict(list)

        for A in workbooks:
            columns = []
            for sheet in A.values():
                for column in sheet.columns:
                    column = ' '.join(column.split()).strip()
                    if '__UNNAMED__' not in column:
                        columns.append(column)

            columns = pl.Series(columns).unique(maintain_order=True).to_list()
            output[A.name] = columns

        height = max(len(column) for column in output.values())

        for name, column in output.items():
            column = column + [None] * (height - len(column))
            output[name] = column
        
        output = pl.DataFrame(output)
        return output