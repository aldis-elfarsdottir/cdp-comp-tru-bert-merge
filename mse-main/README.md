
# configs.json

"year": "The year of the dataset",
"label": "The program, either investor or supply_chain",
"join": "The column on which to join all sheets of the tabbed worksheet",
"sheets": "The zero-indexed ranges (i, j) of sheet indices to keep,  where the sheets i through j are kept",
"merges": "The two zero-indexed ranges of sheet indices to merge",
"redundant": "Duplicate columns in regex whose first occurrence should be kept",
"drop": "Columns in regex to drop",
"renames": "Maps old column names to new column names"



1. Read tabbed .xlsx raw CDP data files

3) Consolidate each sheet
4) Manually rename columns from RowName to longform question names for 2018 - 2020
   a) Also rename the XXX (3) columns in 2010, for example, to XXX (SME)
5) Automatically remove Attachments, Documents, and Table Question columns
   a) But be careful, beacuse for some sheets [FIND OUT WHICH ONES], the Table Question columns
   have actual questions in them. In that case, take the question and put it before each column name
6) Then merge
7) Validate, find % of matchings that are right (check against 2010 and 2020 datasets)
8) Change parameter values accordingly

8. Get rid of RowName and Row, which only appear 2018 +
   a) Rename the RowNames and Rows that actually contain useful information and not just 1 / Row 1,
   and make sure when you are processing the workbooks that you rename before you drop columns because
   you don't want to drop the Row/RowNames that are important before renaming.

::::::::::::::::::::: Merging :::::::::::::::::::::

1) Procedure

---

This module contains functionality to create a 1:1 mapping between the
columns of two datasets (which can be generalized to n > 2 datasets).

Dataset A | m_A x n_A
┌───────┬───────┬───────┬───────┐
│ A1    │ A2    │ A3    │ ...   │
│═══════╪═══════╪═══════╪═══════│
│ a11   │ a12   │ a13   │ ...   │
│ a21   │ a22   │ a23   │ ...   │
│ a31   │ a32   │ a33   │ ...   │
│ ...   │ ...   │ ...   │ ...   │
└───────────────────────────────┘

Dataset B | m_B x n_B
┌───────┬───────┬───────┬───────┐
│ B1    │ B2    │ B3    │ ...   │
│═══════╪═══════╪═══════╪═══════│
│ b11   │ b12   │ b13   │ ...   │
│ b21   │ b22   │ b23   │ ...   │
│ b31   │ b32   │ b33   │ ...   │
│ ...   │ ...   │ ...   │ ...   │
└───────────────────────────────┘

a) Cost matrix calculation

---

Cost Matrix (C) | n_A x n_B
┌───────┬───────┬───────┬───────┬───────┐
│       │ B1    │ B2    │ B3    │ ...   │
│═══════╪═══════╪═══════╪═══════╪═══════│
│ A1    │ A1@B1 │ A1@B2 │ A1@B3 │ ...   │
│ A2    │ A2@B1 │ A2@B2 │ A2@B3 │ ...   │
│ A3    │ A3@B1 │ A3@B2 │ A3@B3 │ ...   │
│ ...   │ ...   │ ...   │ ...   │ ...   │
└───────────────────────────────────────┘

Where AX@BY represents the weighted sum of several normalized metrics:
|
|  w_coldist * norm(coldist(AX, BY))
|  + w_colsim * norm(-colsim(AX, BY))
|  + w_fieldsim * norm(-fieldsim(AX, BY))
|  + w_colcontext * ...

1) w_colsim, w_coldist, w_fieldsim :: the weights for the metrics in the calculation
2) norm() :: linearly normalizes the values to be between 0 and 1, inclusive
3) coldist() :: the levenshtein distance between column names AX and BY
4) colsim() :: The cosine similarity between column names AX and BY. This is subsequently
   negated because a higher similarity score should correspond to a lower cost, and a lower
   score should correspond to a higher cost.
5) fieldsim() :: The sum of the cosine similarity scores of n pairs of sample fields
   a) n non-null fields are uniquely sampled from columns AX and BY to form the
   tuples S_AX = (s1_AX, s2_AX, ..., sn_AX), S_BY = (s1_BY, s2_BY, ..., sn_BY)
   b) This sampling occurs with replacement if n > the number of unique non-null fields
   in the column
   c) These fields are evaluated for their similarities pairwise. That is, similarity scores
   are calculated for (s1_AX, s1_BY), (s2_AX, s2_BY), ..., (sn_AX, sn_BY)
   d) These similarity scores are then summed to create the final similarity score
6) Minimum assignment

---

Once the cost matrix C is calculated, the problem then becomes finding a set of
elements S = {c_ab | 0 ≤ a ∈ Z ≤ n_A, 0 ≤ b ∈ Z ≤ n_B} such that all of the following
criteria are fulfilled:

1) |S| ≤ min(n_A, n_B)
2) No two elements overlap in terms of their rows or columns. In other words:
   a1 ≠ a2, b1 ≠ b2 ∀ pairs of elements (c_a1b1, c_a2b2) ∈ S
3) The sum of the cost is globally minimized

This is a bipartite matching problem, so we use the Hungarian algorithm. Intuitively, we
use this approach because when the columns of A are being mapped to the columns of B,
we expect only one column in A to map to only one column in B. Thus, there is expected
to be no duplicate column names within datasets A and B.

A cost threshold thresh_C is then applied to filter out any matchings that have a cost
greater than the specified threshold. This post-processing guarantees that the overall set of
matches is globally optimal before applying the threshold. In other words, the algorithm considers
the entire structure and relationships in the graph.

a) Mapping

---

We create a weighted (n_B, n_A)-biregular bipartite graph G = (U, V, E):

1) U: the set of column names in A, {A1, B1, ...}
2) V: the set of column names in B, {B1, B2, ...}
3) E: the set of edges {(u_i, v_j) | 0 ≤ i ∈ Z ≤ n_A, 0 ≤ j ∈ Z ≤ n_B}
   a) |u_i| = n_B ∀ i
   b) |v_j| = n_A ∀ j
4) The weight of the edge (u_i, v_j) = C[i, j] = c_ij ∀ i, j

The Hungarian algorithm is then used to find the minimum assignment.
Below is an example of a mapping that may result from the procedure
detailed above given the dimensions n_A = 5, n_B = 7:

Mapping (M_AB) | n_B x 2
┌───────┬───────┐
│ A     │ B     │
│═══════╪═══════│
│ A1    │ B4    │
│ A2    │ B3    │
│ A3    │ B7    │
│ A4    │ B1    │
│ A5    │ B2    │
│ Null  │ B5    │
│ Null  │ B6    │
└───────────────┘

3) Map chaining

---

Once the mapping M_AB is constructed, it might be desired to add column names
from a third dataset C to this mapping. To achieve this, we can calculate the
mapping M_BC. Then, we create a temporary key mapping M_AB' (n_B x 1) such that
for 1 ≤ i ≤ n_B, the element i of M_AB' is the last non-null element of row i of M_AB.
Then, M_AB' is outer-joined with M_BC to form the dataframe M_AB'_BC, the M_AB' column
is dropped from M_AB'_BC, and M_AB'_BC is horizontally concatenated with M_AB with
rows that are added being filled with Null.

4) Manual validation

---

The resulting mapping M_ABC[...] is then manually validated as
a final check to ensure the validity of its mappings.

5) Considerations

---

1) We don't incorporate response type (float/int, str) into the matrix calculation
   because some columns include a currency unit, for example, and in another year they
   don't include the unit, so in this case, the str column likely wouldn't be matched
   to the int/float column even though they are supposed to be matched.
2) This procedure maps all the columns in A to a portion of columns in B (assuming
   n_A < n_B) and hence assumes that this accurately models reality. In some cases,
   however (as is the case with CDP), some columns in year X's dataset may be dropped in
   year (X + 1)'s dataset, and hence M_X(X+1) may not be completely valid. This is why
   the manual validation step is necessary, as well as why there is a cost threshold parameter.
3) The chaining procedure also ignores cases in which a column from year X is dropped in year
   X+1 and reappears in year X+2, but potentially with different wording.

TODO

1. We may have to take SME, merge them, then take normal investor, merge those, then take normal sc, merge those,
   then merge all of the (three) resulting mappings.
2. Add docstrings
3. Have an example of a complicated merge
4. Think about that dummy variable/column problem for regression
5. After each mapping, tally how many mappings are right and how many are wrong (by comparing with the manual mapping)
6. A mapping can contain the same column across two years exactly, but they might not be matched,
   so when we are mapping old to new column names, we are including the same column twice, thrice, etc. Thus, unique it.
