# CDP Data Processing and Analysis Pipeline

## Overview

This repository contains a comprehensive machine learning pipeline for processing, merging, and analyzing Carbon Disclosure Project (CDP) data spanning from 2010 to 2020. The pipeline integrates multiple advanced techniques including semantic similarity matching, multilingual translation, and climate-specific text classification to standardize and analyze corporate climate disclosures across time and organizational contexts.

## Pipeline Architecture

The pipeline consists of six main stages:

1. **Data Loading and Exploration**
2. **Data Processing and Standardization**
3. **Automated Column Merging**
4. **Multilingual Translation**
5. **Climate-Specific Text Classification**
6. **Metrics Generation and Analysis**

## System Requirements

* Python 3.8+
* CUDA-compatible GPU (recommended for optimal performance)
* Minimum 16GB RAM
* Storage: ~50GB for intermediate files and models

### Dependencies

```python
torch>=1.9.0
transformers>=4.20.0
polars>=0.18.0
sentence-transformers>=2.2.0
fast-langdetect>=0.3.0
scikit-learn>=1.1.0
networkx>=2.8.0
more-itertools>=8.14.0
```

## Stage 1: Data Loading and Exploration

### Input Data Structure

The pipeline expects Excel workbooks organized as:

```
data/cdp/output/raw/{label}/{year}.xlsx
```

Where `label` ∈ {investor, supply_chain} and `year` ∈ [2010, 2020].

### Configuration System

Each workbook requires a configuration JSON specifying:

* **join** : Primary key column for sheet consolidation
* **sheets** : List of sheet indices or ranges to process
* **merges** : Dictionary mapping section names to sheet ranges
* **redundant** : Regex patterns for identifying redundant columns
* **drop** : Regex patterns for columns to exclude
* **renames** : Column name standardization mappings

### Exploratory Data Analysis (EDA)

The exploration module (`exploration.py`) performs:

1. **Duplicate Detection** : Identifies columns with identical names within sheets
2. **Redundancy Analysis** : Finds columns appearing across multiple sheets (configurable threshold)
3. **Column Enumeration** : Catalogs unique column names per workbook
4. **Sheet Structure Analysis** : Documents sheet organization across workbooks

**Technical Implementation:**

* Uses `bin` parameter for proportional redundancy thresholds
* Applies `min` parameter for absolute occurrence thresholds
* Excludes `__UNNAMED__` columns from analysis
* Implements whitespace normalization for column matching

## Stage 2: Data Processing and Standardization

### Column Property System

The processor implements a structured column naming convention:

```
column={name}|sheet={source}|value={type}
```

**Property Components:**

* **column** : Original column identifier
* **sheet** : Source sheet name or `*` for metadata
* **value** : Data type (`response`, `meta`, `score`, `label`)

### Workbook Consolidation Process

1. **Sheet Filtering** : Extracts specified sheet ranges from configurations
2. **Horizontal Joining** : Consolidates sheets using the designated join column
3. **Field Cleaning** :

* Removes excessive whitespace using regex `r'\s+'`
* Eliminates Unicode line separators (`\u2028`, `\u2029`)
* Converts empty strings to null values

1. **Column Standardization** : Applies rename mappings and drops redundant columns
2. **List Aggregation** : Groups multiple responses per entity into list structures

**Data Quality Enhancements:**

* Maintains row order during aggregation (`maintain_order=True`)
* Preserves null values in list structures
* Implements strict regex-based column filtering
* Adds temporal metadata (year, label) to all records

### Output Schema

Processed dataframes use `pl.List(pl.String)` schema for all columns, enabling:

* Multiple responses per question
* Null value preservation
* Consistent data types across heterogeneous sources

## Stage 3: Automated Column Merging

### Semantic Similarity Framework

The merging system (`merging.py`) implements a sophisticated multi-metric similarity approach:

#### Fingerprint Generation

Each dataframe generates a "fingerprint" containing:

1. **Column Embeddings** : Sentence transformer encodings of column names
2. **Context Embeddings** : Windowed average of neighboring column embeddings
3. **Sample Embeddings** : Encodings of representative field values
4. **Metadata** : Column names and sheet groupings

**Technical Specifications:**

* **Model** : `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional embeddings)
* **Context Window** : Configurable (default: 1 neighboring column each side)
* **Sample Size** : Configurable number of representative values per column (default: 4)
* **Sampling Strategy** : Uniform with replacement, deterministic seeding

#### Multi-Metric Cost Matrix

The algorithm computes four similarity components:

1. **Levenshtein Distance (D_col)** : Character-level column name similarity
2. **Semantic Column Similarity (S_col)** : Cosine similarity of column name embeddings
3. **Contextual Similarity (S_context)** : Similarity of neighboring column patterns
4. **Field Content Similarity (S_field)** : Average pairwise similarity of sample values

**Cost Matrix Computation:**

```python
C = w_D_col * normalize(D_col) + 
    w_S_col * normalize(-S_col) + 
    w_S_context * normalize(-S_context) + 
    w_S_field * normalize(-S_field)
```

**Default Weights** (optimized via Optuna):

* `w_D_col`: 1.7
* `w_S_col`: 1.8
* `w_S_field`: 0.3
* `w_S_context`: 1.8

#### Optimal Assignment Algorithm

Uses NetworkX implementation of the Hungarian algorithm for minimum-weight bipartite matching.

#### Threshold Methods

Three thresholding approaches for filtering low-quality matches:

1. **Fixed** : User-defined threshold (default: 0.24)
2. **Cluster** : K-means clustering to separate good/poor matches
3. **Spread** : Statistical threshold based on mean + z-factor * std

### Hyperparameter Optimization

The pipeline includes Optuna-based optimization comparing automated mappings against manually curated ground truth, evaluating pairwise column correspondence accuracy.

## Stage 4: Multilingual Translation

### Translation Architecture (`translation.py`)

**Model Configuration:**

* **Base Model** : `alirezamsh/small100` (multilingual sequence-to-sequence)
* **Target Language** : English (ISO 639-1: 'en')
* **Optimization** : 8-bit quantization via BitsAndBytesConfig
* **Compilation** : PyTorch 2.0 compilation for inference acceleration

### Language Detection Pipeline

1. **Preprocessing** : Replaces newlines with spaces, truncates to 100 characters
2. **Detection** : Uses `fast-langdetect` with `low_memory=False` for accuracy
3. **Fallback** : Defaults to target language if detection fails

### Translation Process

**Tokenization Parameters:**

* **Padding** : Dynamic padding within batches
* **Truncation** : Enabled for sequences exceeding model limits
* **Target Language Specification** : Tokenizer configured with `tgt_lang` parameter

**Generation Parameters:**

* **Beam Search** : Single beam (`num_beams=1`) for efficiency
* **Batch Processing** : Configurable batch size (default: 64)
* **Memory Management** : GPU tensors with automatic cleanup

### Caching Strategy

Implements intelligent caching to avoid retranslating identical texts:

* Key: Original text string
* Value: Translated text
* Persistence: Session-level (cleared between pipeline runs)

### Output Enhancement

The translation module can optionally preserve:

* **Detected Languages** : Original language codes for linguistic analysis
* **Translation Status** : Boolean indicators for translated vs. original content

## Stage 5: Climate-Specific Text Classification

### Multi-Model Classification Framework (`classification.py`)

The pipeline employs eight specialized ClimateBERT models for comprehensive climate disclosure analysis:

#### Model Specifications

1. **Climate Specificity** (`climatebert/distilroberta-base-climate-specificity`)
   * **Purpose** : Distinguishes climate-specific from general content
   * **Labels** : `spec` (specific), `non` (non-specific)
2. **Environmental Claims** (`climatebert/environmental-claims`)
   * **Purpose** : Identifies environmental claims and statements
   * **Labels** : `yes`, `no`
3. **Transition Risk/Opportunity** (`climatebert/transition-physical`)
   * **Purpose** : Detects transition-related climate content
   * **Labels** : `LABEL_1` (transition-related), `LABEL_0` (not transition-related)
4. **Climate Commitment** (`climatebert/distilroberta-base-climate-commitment`)
   * **Purpose** : Identifies climate commitments and pledges
   * **Labels** : `yes`, `no`
5. **TCFD Framework Classification** (`climatebert/distilroberta-base-climate-tcfd`)
   * **Purpose** : Categorizes content by TCFD pillars
   * **Labels** : `strategy`, `governance`, `risk`, `metrics`, `none`
6. **Climate Sentiment** (`climatebert/distilroberta-base-climate-sentiment`)
   * **Purpose** : Analyzes sentiment toward climate issues
   * **Labels** : `opportunity`, `neutral`, `risk`
7. **Net-Zero/Reduction** (`climatebert/netzero-reduction`)
   * **Purpose** : Identifies net-zero and emission reduction content
   * **Labels** : `net-zero`, `reduction`, `none`
8. **Renewable Energy** (`climatebert/renewable`)
   * **Purpose** : Detects renewable energy discussions
   * **Labels** : `LABEL_1` (renewable-related), `LABEL_0` (not renewable-related)

### Classification Technical Implementation

**Model Loading:**

* **Precision** : Half precision (FP16) for GPU inference efficiency
* **Mode** : Evaluation mode with gradient computation disabled
* **Device** : Automatic CUDA detection with CPU fallback

**Tokenization Process:**

* **Max Length** : 512 tokens (BERT/RoBERTa standard)
* **Padding Strategy** : `max_length` for consistent batch processing
* **Truncation** : Enabled to handle long texts
* **Return Format** : PyTorch tensors

**Inference Pipeline:**

1. **Batch Processing** : Configurable batch size (default: 128)
2. **Forward Pass** : Model inference with attention masks
3. **Probability Extraction** : Softmax activation on logits
4. **Label Assignment** : Argmax for primary classification
5. **Score Preservation** : Full probability distributions retained

### Caching and Memory Management

**Two-Level Caching:**

* **Session Cache** : `(model_name, text) -> (label, scores)` mapping
* **Memory Optimization** : Explicit model unloading between models
* **GPU Management** : `torch.cuda.empty_cache()` and garbage collection

### Output Schema Enhancement

For each response column, the system generates:

* **Label Columns** : Primary classifications per model
* **Score Columns** : Probability distributions for all labels
* **Metadata** : Model identifiers and value types via column property system

### Chunked Output Support

Supports incremental processing with intermediate file output:

* **Format** : Parquet with ZSTD compression
* **Naming** : `classified_{model_name}.parquet`
* **Schema** : Consistent with main pipeline structure

## Stage 6: Metrics Generation and Analysis

### Quantitative Metrics Framework

The final stage generates comprehensive metrics for analysis:

#### Response-Level Metrics

For each response column:

1. **Average Climate Specificity** : Mean probability scores across responses
2. **Character Count** : Total character length of concatenated responses
3. **Response Count** : Number of non-null responses per question

#### Text Quality Indicators

* **Language Distribution** : Proportion of responses requiring translation
* **Response Completeness** : Coverage across question types
* **Content Density** : Character-to-response ratios

### Financial Analysis Components

#### Investment Analysis

**Currency Standardization:**

* Historical exchange rates (2010-2020) from IMF/World Bank
* Standardized currency mappings for CDP disclosure variations
* Inflation adjustment capabilities for temporal comparisons

**Key Metrics:**

* Investment amounts in climate initiatives
* Payback period distributions
* Return on investment calculations
* Temporal investment trends

#### Emissions Intensity Analysis

**Target Setting Trends:**

* Emission reduction ambition over time
* Target achievement rates
* Industry benchmark comparisons
* Regulatory compliance patterns

### Statistical Analysis Framework

The pipeline supports various analytical approaches:

1. **Time Series Analysis** : Temporal trend identification
2. **Cluster Analysis** : Company grouping by disclosure patterns
3. **Regression Analysis** : Factor influence on climate metrics
4. **Correlation Analysis** : Cross-metric relationships

### Visualization Support

Comprehensive plotting capabilities using:

* **Matplotlib/Seaborn** : Statistical visualizations
* **Altair** : Interactive time series plots
* **Custom Formatters** : Financial data presentation

## Performance Optimization

### Computational Efficiency

* **GPU Acceleration** : Full CUDA support for all ML operations
* **Batch Processing** : Optimized batch sizes for memory constraints
* **Model Compilation** : PyTorch 2.0 compilation where applicable
* **Memory Management** : Explicit cleanup between processing stages

### Storage Optimization

* **Parquet Format** : Columnar storage with compression
* **Schema Consistency** : Unified data types across pipeline stages
* **Incremental Processing** : Checkpoint capability for large datasets

### Scalability Considerations

* **Memory Mapping** : Large file handling without full memory loading
* **Distributed Processing** : Framework ready for multi-GPU scaling
* **Configuration Management** : Externalized parameters for easy adjustment

## Quality Assurance

### Data Validation

* **Schema Enforcement** : Strict type checking throughout pipeline
* **Null Handling** : Consistent null value treatment
* **Duplicate Detection** : Automatic identification of duplicate records

### Model Validation

* **Ground Truth Comparison** : Manual validation against expert annotations
* **Cross-Validation** : Model performance assessment
* **Confidence Scoring** : Uncertainty quantification for classifications

### Pipeline Monitoring

* **Progress Tracking** : Detailed progress bars for long-running operations
* **Error Handling** : Graceful failure recovery with detailed logging
* **Resource Monitoring** : Memory and GPU usage tracking

## Usage Instructions

### Basic Pipeline Execution

```python
# Initialize models
processor = Processor(merger=merger)
translator = Translator(target='en', model_name='alirezamsh/small100')
classifier = Classifier(models=CLIMATE_MODELS)

# Execute pipeline stages
workbooks = load_workbooks(years=[2010, 2020], labels=['investor', 'supply_chain'])
processed = processor.process_workbooks(workbooks)
merged, mapping = merger.merge(processed)
translated = translator.translate_df(merged, processor=processor)
classified = classifier.classify_df(translated, processor=processor)
metrics = generate_metrics(classified)
```

### Configuration Customization

The pipeline supports extensive customization through configuration files and model parameters. Key configuration areas include:

* **Similarity Weights** : Adjusting the importance of different similarity metrics
* **Threshold Methods** : Choosing optimal matching thresholds
* **Model Selection** : Swapping in domain-specific models
* **Batch Sizes** : Optimizing for available hardware resources

## Research Applications

This pipeline enables various research applications:

1. **Longitudinal Climate Disclosure Analysis** : Tracking corporate climate reporting evolution
2. **Cross-Sectoral Comparisons** : Analyzing disclosure patterns across industries
3. **Regulatory Impact Assessment** : Measuring policy effects on disclosure quality
4. **Investment Decision Analysis** : Connecting climate disclosures to financial outcomes
5. **Language and Cultural Analysis** : Examining regional differences in climate communication

## Citation

When using this pipeline in research publications, please cite both the software and the underlying models:

```bibtex
@software{cdp_pipeline_2024,
  title={CDP Data Processing and Analysis Pipeline},
  author={[Your Name]},
  year={2024},
  url={[Repository URL]}
}
```

Additionally, cite the ClimateBERT models and other dependencies as specified in their respective documentation.

## Contributing

Please refer to CONTRIBUTING.md for guidelines on code contributions, bug reports, and feature requests.

## License

[Specify your license here]

## Support

For technical support, please [contact information or issue tracker].
