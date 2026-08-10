# cdp-comp-tru-bert-merge
2025_10_09_AE_cdp_comp_tru_bert_merge documents raw script in which CDP was manually merged along select column topics (initiative types, targets, audience, etc.). Later in the script, key ClimateBERT variables from the Python extract of CH-cdp-bert-pipeline.zip were re-incorporated to augment the core CDP manual merge with key BERT classification variables, and then merge the resulting CDP-BERT with Compustat and Trucost variables (mostly financial and informational, with additional emissions data from Trucost). The output of cdp_comp_tru_bert was then uploaded to Redivis. For public Redivis workflow see https://redivis.com/workflows/js40-7gggmx4ea. 

The Python script documents the extracts performed to collect key BERT variables from the scaled-up CDP translation, merge, and preprocessing performed by Chase Hikida. 

## Citation

If you use this code or methodology, please cite:

> Elfarsdottir, A. ([2026]). _Corporate Carbon Credibility: Detecting Signals of Credibility in Corporate Carbon Reports using Natural Language Processing_ (Doctoral dissertation). Stanford University. [URL coming soon]

BibTeX:
```bibtex
@phdthesis{elfarsdottir[year],
  author = {Elfarsdottir, Aldis},
  title  = {[Corporate Carbon Credibility: Detecting Signals of Credibility in Corporate Carbon Reports using Natural Language Processing]},
  school = {[Stanford University]},
  year   = {[2026]},
  type   = {PhD dissertation},
  url    = {[URL available soon]}
}
```

Portions of the CDP merge, translation, and ClimateBERT preprocessing pipeline were developed by Chase Hikida and are used with permission; see `CH-cdp-bert-pipeline.zip` for that codebase, from `ChaseHikida/mse`.
