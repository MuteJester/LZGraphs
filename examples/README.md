# LZGraphs examples

This directory holds the user-facing tutorial notebooks. The notebooks are
organized into two parallel tracks, one for each graph family that ships
with the package. Pick the track that matches what you want to learn, and
work through the lessons in order.

```
examples/
├── data/                 shared example data files (TCR CDR3 repertoires)
├── lzgraph/              LZ76-based LZGraph class (the published, citable one)
│   ├── 01_Getting_Started.ipynb
│   ├── 02_Analytics_and_Diversity.ipynb
│   └── 03_Advanced_Usage.ipynb
└── flashback/            FlashBackGraph class (Markovian DAG variant)
    ├── 01_Getting_Started.ipynb
    ├── 02_Analytics_and_Diversity.ipynb
    └── 03_Advanced_Usage.ipynb
```

## Which track should I start with?

If you have not used either class before, start with **`lzgraph/01_Getting_Started.ipynb`**. The LZGraph notebooks introduce the
core concepts (LZ76 tokenization, `pgen`, simulate, save/load) that the
FlashBack notebooks build on. After Lesson 1, you can jump to the FlashBack
track if you want.

Quick guide to picking a class for your work:

| You want to ... | Use |
|---|---|
| Score / simulate / cluster TCR or BCR CDR3 repertoires with V/J gene constraints | `LZGraph` |
| Build a fast Markovian model of an arbitrary symbol-sequence corpus | `FlashBackGraph` |
| Build and validate from a very large file (millions of sequences) with bounded memory | `FlashBackGraph` + `FlashBackStream` |
| Use the LZGraph methodology cited in Konstantinovsky and Yaari, 2023 | `LZGraph` |

See the project [README.md](../README.md) for a side-by-side feature table.

## Lesson outline

Both tracks share the same three-lesson arc:

1. **Getting Started.** Build a graph, score a sequence, simulate, save/load.
2. **Analytics and Diversity.** Hill numbers, log-pgen distribution, occupancy
   predictions, repertoire comparison.
3. **Advanced Usage.** Gene access (LZGraph), set algebra, Bayesian
   posterior personalization, ML feature extraction.

## Data files

The `data/` directory contains three small example repertoires:

| File | Format | Notes |
|---|---|---|
| `ExampleData1.csv` | `cdr3_rearrangement` (nucleotide) | 5,000 sequences, no gene calls |
| `ExampleData2.csv` | `cdr3_rearrangement`, `V`, `J` | 5,000 NT sequences with V/J |
| `ExampleData3.csv` | `cdr3_amino_acid`, `V`, `J` | 5,000 AA sequences with V/J (used in most notebooks) |

## Running the notebooks

From this directory:

```bash
cd lzgraph             # or: cd flashback
jupyter notebook       # then open the lesson you want
```

Each notebook references data via `../data/<filename>`, so the working
directory must be the lesson's own folder (which is what `jupyter
notebook` defaults to when launched from inside that folder).
