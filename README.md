# cdp-comp-tru-bert-merge
2025_10_09_AE_cdp_comp_tru_bert_merge documents raw script in which CDP was manually merged along select column topics (initiative types, targets, audience, etc.). Later in the script, key ClimateBERT variables from the Python extract of mse-main (previously CH-cdp-bert-pipeline.zip) were re-incorporated to augment the core CDP manual merge with key BERT classification variables, and then merge the resulting CDP-BERT with Compustat and Trucost variables (mostly financial and informational, with additional emissions data from Trucost). The output (cdp_tru_comp_bert) was then uploaded to Redivis with 176 variables and 154,467 rows (383 MB). For public Redivis workflow see https://redivis.com/workflows/js40-7gggmx4ea. 

The Python script documents the extracts performed to collect key BERT variables from the scaled-up CDP translation, merge, and preprocessing. 

## Citation

If you use this code or methodology, please cite:

> Elfarsdottir, A. (2026). _Corporate Carbon Credibility: Detecting Signals of Credibility in Corporate Carbon Reports using Natural Language Processing_ (Doctoral dissertation). Stanford University. https://purl.stanford.edu/fr864tf9302

BibTeX:
```bibtex
@phdthesis{elfarsdottir2026,
  author = {Elfarsdottir, Aldis},
  title  = {[Corporate Carbon Credibility: Detecting Signals of Credibility in Corporate Carbon Reports using Natural Language Processing]},
  school = {[Stanford University]},
  year   = {[2026]},
  type   = {PhD dissertation},
  url    = {[https://purl.stanford.edu/fr864tf9302]}
}
```

Portions of the codebase (i.e., scaled-up CDP merge, translation, and ClimateBERT preprocessing) were developed by research assistant, Chase Hikida, and are used with permission; see `mse-main`, from `ChaseHikida/mse`.
