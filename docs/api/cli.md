---
tags:
  - CLI
---

# CLI Reference

The `lzg` command-line tool provides fast, scriptable access to every major
LZGraphs operation: building graphs, scoring sequences, simulating repertoires,
measuring diversity, and more. Every command reads from files (or stdin) and
writes tab-separated text to stdout, so it slots naturally into Unix pipelines.

---

## Installation and verification

After installing LZGraphs (`pip install LZGraphs`), the `lzg` entry point is
available system-wide.

```bash
# Confirm it is installed
lzg --version
```

```text
lzg 3.0.1
```

```bash
# See all available commands
lzg --help
```

```text
usage: lzg [-h] [--version] [-q] {build,info,score,simulate,diversity,compare,decompose,saturation,predict,posterior} ...

LZGraphs — LZ76 compression graphs for immune repertoire analysis

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -q, --quiet           suppress progress

commands:
  {build,info,score,simulate,diversity,compare,decompose,saturation,predict,posterior}
    build               Build a graph from sequences
    info                Inspect a saved graph
    score               Compute LZPGEN for sequences
    simulate            Generate sequences from a graph
    diversity           Diversity metrics
    compare             Compare two repertoires
    decompose           LZ76-decompose sequences
    saturation          Node/edge saturation curve
    predict             Occupancy predictions
    posterior           Bayesian posterior update
```

!!! tip "Global flags"
    Every subcommand accepts the global `-q / --quiet` flag, which suppresses
    the informational messages printed to stderr. Useful when piping stdout
    into another program.

---

## Input file formats

`lzg` auto-detects the file format from the first line.  Three formats are
supported:

### Plain text -- one sequence per line

```text title="sequences.txt"
CASSLAPGATNEKLFF
CASSLVGGPYEQYF
CASSQEAGGTDTQYF
```

### Plain text with abundances

Tab-separate each sequence from its integer count:

```text title="sequences_abd.txt"
CASSLAPGATNEKLFF	5
CASSLVGGPYEQYF	12
CASSQEAGGTDTQYF	1
```

### TSV / CSV with header

Any `.tsv`, `.csv`, or `.gz`-compressed tabular file with a header row.
Column names are auto-detected per variant:

| Variant | Auto-detected sequence columns (first match wins) |
|---------|---------------------------------------------------|
| `aap`   | `junction_aa`, `cdr3_amino_acid`, `cdr3_aa`, `aminoacid` |
| `ndp`   | `junction`, `cdr3_rearrangement`, `cdr3_nt`, `nucleotide` |
| `naive` | `junction_aa`, `cdr3_amino_acid`, `junction`, `cdr3_rearrangement` |

Fallback columns tried for all variants: `sequence`, `cdr3`, `seq`.

Gene columns default to `v_call` and `j_call` (AIRR standard).  Abundance
defaults to `duplicate_count`.

```text title="repertoire.tsv"
junction_aa	v_call	j_call	duplicate_count
CASSLAPGATNEKLFF	TRBV5-1*01	TRBJ1-4*01	5
CASSLVGGPYEQYF	TRBV28*01	TRBJ2-7*01	12
CASSQEAGGTDTQYF	TRBV4-2*01	TRBJ2-3*01	1
```

!!! note "Reading from stdin"
    Commands that accept an `input` positional argument default to `-` (stdin)
    when omitted, so you can pipe data directly:

    ```bash
    cat sequences.txt | lzg decompose
    ```

!!! note "Gzip support"
    Any input file ending in `.gz` is transparently decompressed.

---

## Commands

### `build` -- Build a graph from sequences

Read sequences (and optionally V/J genes and abundances) from a tabular or
plain-text file, construct an LZ76 compression graph, and save it to a
compact `.lzg` binary.

**Usage**

```text
lzg build INPUT -o OUTPUT [options]
```

**Arguments and options**

| Flag | Description |
|------|-------------|
| `INPUT` | Input file (`.txt`, `.tsv`, `.csv`, `.gz`, or `-` for stdin) |
| `-o, --output` | **(required)** Output `.lzg` file path |
| `-V, --variant` | Graph variant: `aap`, `ndp`, or `naive` (default: `aap`) |
| `-s, --seq-column` | Sequence column name (default: auto-detect) |
| `--v-column` | V gene column name (default: `v_call`) |
| `--j-column` | J gene column name (default: `j_call`) |
| `-a, --abundance-column` | Abundance / count column name |
| `--no-genes` | Ignore gene columns even if present |
| `--smoothing` | Laplace smoothing constant (default: `0.0`) |

**Example**

```bash
lzg build repertoire.tsv -o repertoire.lzg -V aap
```

```text
[build] 48312 sequences read (14 V genes, 13 J genes) (0.31s)
[build] 9842 nodes, 27531 edges (1.47s)
[build] saved repertoire.lzg (2104.3 KB)
```

!!! tip "Choosing a variant"
    Use `aap` for amino-acid CDR3 sequences (the most common case), `ndp` for
    nucleotide sequences with positional encoding, and `naive` for raw
    character-level decomposition of any string.

---

### `info` -- Inspect a saved graph

Print a structured summary of a `.lzg` file: graph size, diversity profile,
generation probability statistics, and optionally V/J gene marginals.

**Usage**

```text
lzg info GRAPH [options]
```

**Arguments and options**

| Flag | Description |
|------|-------------|
| `GRAPH` | Path to a `.lzg` file |
| `--genes` | Include V/J gene marginal probabilities |
| `--all` | Print everything (genes + full Hill curve) |
| `--json` | Output as JSON instead of tagged text |

**Example**

```bash
lzg info repertoire.lzg
```

```text
# lzg info v3.0.1 — repertoire.lzg
GR	variant	aap
GR	nodes	9842
GR	edges	27531
GR	initial_states	387
GR	terminal_states	214
GR	is_dag	yes
GR	has_gene_data	yes
GR	path_count	3.18204e+08
DV	effective_diversity	4021.3312
DV	entropy_nats	8.2993
DV	entropy_bits	11.9729
DV	uniformity	0.9032
PR	pgen_mean	-18.4210
PR	pgen_std	3.2714
PR	dynamic_range_decades	12.6831
PR	is_proper	yes
```

```bash
# Gene marginals included
lzg info repertoire.lzg --genes
```

!!! tip "Machine-readable output"
    The tagged text format (`PREFIX<tab>KEY<tab>VALUE`) is easy to parse with
    `awk` or `cut`. For structured consumption, use `--json`:

    ```bash
    lzg info repertoire.lzg --json | python3 -m json.tool
    ```

---

### `score` -- Compute LZPGEN for sequences

Score one or more sequences against a graph, producing the log-probability
(LZPGEN) of each sequence under the graph's generative model.

**Usage**

```text
lzg score GRAPH [INPUT] [options]
```

**Arguments and options**

| Flag | Description |
|------|-------------|
| `GRAPH` | Path to a `.lzg` file |
| `INPUT` | Sequence file (default: stdin) |
| `-s, --seq-column` | Sequence column name (default: auto-detect) |
| `-o, --output` | Output file (default: stdout) |
| `--prob` | Output raw probability instead of log-probability |
| `--append` | Pass through input columns (not just sequence) |
| `--json` | JSON output |

**Example**

```bash
lzg score repertoire.lzg query_sequences.txt
```

```text
sequence	lzpgen
CASSLAPGATNEKLFF	-14.831204
CASSLVGGPYEQYF	-16.229417
CASSQEAGGTDTQYF	-19.003851
[score] scored 3 sequences
```

```bash
# Pipe from stdin, output probabilities
echo "CASSLAPGATNEKLFF" | lzg score repertoire.lzg --prob
```

```text
sequence	pgen
CASSLAPGATNEKLFF	0.000000
[score] scored 1 sequences
```

!!! note
    The `--prob` flag exponentiates the log-probability. For very rare
    sequences the probability will be indistinguishable from zero in
    fixed-precision output -- prefer log-probabilities for downstream analysis.

---

### `simulate` -- Generate sequences from a graph

Sample new sequences from the graph's learned transition model using a fast C
extension (or optimized Python fallback). Optionally constrain by V/J gene.

**Usage**

```text
lzg simulate GRAPH -n COUNT [options]
```

**Arguments and options**

| Flag | Description |
|------|-------------|
| `GRAPH` | Path to a `.lzg` file |
| `-n, --count` | **(required)** Number of sequences to generate |
| `-o, --output` | Output file (default: stdout) |
| `--seed` | RNG seed for reproducibility |
| `--v-gene` | Constrain to a specific V gene |
| `--j-gene` | Constrain to a specific J gene |
| `--sample-genes` | Sample V/J genes from the joint distribution |
| `--with-details` | Include `lzpgen` and `n_tokens` columns |
| `--json` | JSON output |

**Example**

```bash
lzg simulate repertoire.lzg -n 5 --seed 42
```

```text
CASSLGQAYEQYF
CASSPAGGTEAFF
CASSQDRANYGYTF
CASSFRGGNTIYF
CASSLEETQYF
[simulate] generated 5 sequences
```

```bash
# With generation probability and token count
lzg simulate repertoire.lzg -n 3 --seed 42 --with-details
```

```text
sequence	lzpgen	n_tokens
CASSLGQAYEQYF	-15.203419	5
CASSPAGGTEAFF	-17.841002	6
CASSQDRANYGYTF	-16.558134	6
[simulate] generated 3 sequences
```

!!! tip "Gene-constrained generation"
    If the graph was built with gene data, you can condition on specific genes:

    ```bash
    lzg simulate repertoire.lzg -n 1000 --v-gene "TRBV5-1*01" --seed 7
    ```

    Or sample V/J pairs from the joint distribution with `--sample-genes`,
    which attaches gene labels to each simulated sequence.

---

### `diversity` -- Diversity and structural statistics

Compute Hill diversity numbers, effective diversity, Shannon entropy,
uniformity, and generation-probability moments for a graph.

**Usage**

```text
lzg diversity GRAPH [options]
```

**Arguments and options**

| Flag | Description |
|------|-------------|
| `GRAPH` | Path to a `.lzg` file |
| `--hill` | Comma-separated Hill orders (default: `0,1,2,5,inf`) |
| `--json` | JSON output |

**Example**

```bash
lzg diversity repertoire.lzg
```

```text
# lzg diversity v3.0.1
HL	0	318204000.0000
HL	1	4021.3312
HL	2	1847.5590
HL	5	623.4102
HL	inf	42.1837
DV	effective_diversity	4021.3312
DV	entropy_nats	8.2993
DV	entropy_bits	11.9729
DV	uniformity	0.9032
DR	dynamic_range_decades	12.6831
DR	pgen_mean	-18.4210
DR	pgen_std	3.2714
```

!!! note "Hill number interpretation"
    - **Order 0**: Total richness (number of distinct achievable sequences).
    - **Order 1**: Exponential of Shannon entropy -- the "effective" number of
      equally-likely sequences.
    - **Order 2**: Inverse Simpson concentration.
    - **Order inf**: Inverse of the maximum probability -- dominated by the
      single most likely sequence.

---

### `compare` -- Jensen-Shannon divergence between two graphs

Measure the distributional distance between two repertoire graphs using
Jensen-Shannon divergence, along with structural overlap statistics
(shared nodes/edges, Jaccard indices).

**Usage**

```text
lzg compare GRAPH_A GRAPH_B [options]
```

**Arguments and options**

| Flag | Description |
|------|-------------|
| `GRAPH_A` | First `.lzg` file |
| `GRAPH_B` | Second `.lzg` file |
| `--json` | JSON output |

**Example**

```bash
lzg compare healthy.lzg disease.lzg
```

```text
# lzg compare v3.0.1 — healthy.lzg vs disease.lzg
CP	jsd	0.142837
CP	nodes_a	9842
CP	nodes_b	11204
CP	nodes_shared	6318
CP	edges_a	27531
CP	edges_b	30819
CP	edges_shared	14207
CP	jaccard_nodes	0.4295
CP	jaccard_edges	0.3217
```

!!! note
    JSD is symmetric and bounded in [0, 1]. A value near 0 means the
    two graphs encode nearly identical generation-probability distributions.

---

### `decompose` -- LZ76-decompose sequences

Print the Lempel-Ziv 76 decomposition of each input sequence. Useful for
understanding how the algorithm tokenizes a sequence before graph construction.

**Usage**

```text
lzg decompose [INPUT] [options]
```

**Arguments and options**

| Flag | Description |
|------|-------------|
| `INPUT` | Sequence file (default: stdin) |
| `-s, --seq-column` | Sequence column name (default: auto-detect) |
| `-o, --output` | Output file (default: stdout) |
| `-d, --delimiter` | Token delimiter in output (default: `\|`) |
| `--json` | JSON output |

**Example**

```bash
echo "CASSLAPGATNEKLFF" | lzg decompose
```

```text
sequence	tokens	n_tokens
CASSLAPGATNEKLFF	C|A|S|SL|AP|G|AT|NE|KL|FF	10
```

```bash
# Use a different delimiter
echo "CASSLAPGATNEKLFF" | lzg decompose -d " "
```

```text
sequence	tokens	n_tokens
CASSLAPGATNEKLFF	C A S SL AP G AT NE KL FF	10
```

!!! tip
    Pipe a file of sequences through `decompose` to get a quick sense of
    complexity. Sequences with fewer tokens (relative to their length) are
    more repetitive and will contribute fewer unique nodes to the graph.

---

### `saturation` -- Node/edge saturation curve

Track how the number of unique nodes and edges grows as sequences are
incrementally added. This helps assess whether a repertoire has been
sequenced deeply enough to capture its structural diversity.

**Usage**

```text
lzg saturation INPUT [options]
```

**Arguments and options**

| Flag | Description |
|------|-------------|
| `INPUT` | Sequence file |
| `-V, --variant` | Graph variant: `aap`, `ndp`, or `naive` (default: `aap`) |
| `-s, --seq-column` | Sequence column name (default: auto-detect) |
| `-o, --output` | Output file (default: stdout) |
| `--log-every` | Record a data point every N sequences (default: `100`) |
| `--json` | JSON output |

**Example**

```bash
lzg saturation repertoire.tsv -V aap --log-every 500 -o saturation.tsv
```

```text
[saturation] 48312 sequences, variant=aap
```

```bash
head -5 saturation.tsv
```

```text
n_sequences	n_nodes	n_edges
500	1847	3214
1000	3102	5891
1500	4018	8012
2000	4729	9847
```

!!! tip
    Plot the output to visually check for plateau. If the curve is still
    climbing steeply at the end, you likely need deeper sequencing.

---

### `predict` -- Occupancy predictions

Predict ecological properties of the repertoire at arbitrary sequencing
depths. Three subcommands are available: `richness`, `overlap`, and `sharing`.

#### `predict richness`

Estimate the number of distinct sequences (species richness) that would be
observed at given sampling depths.

**Usage**

```text
lzg predict richness GRAPH --depths DEPTHS [options]
```

| Flag | Description |
|------|-------------|
| `GRAPH` | `.lzg` file |
| `--depths` | **(required)** Comma-separated depths or `START:END:N` for log-spaced |
| `-o, --output` | Output file (default: stdout) |
| `--json` | JSON output |

**Example**

```bash
lzg predict richness repertoire.lzg --depths 1000,10000,100000,1000000
```

```text
depth	predicted_richness
1000	987.4210
10000	8241.3019
100000	42819.5531
1000000	118402.7743
```

```bash
# Log-spaced depths: 10 points from 1000 to 1000000
lzg predict richness repertoire.lzg --depths 1000:1000000:10
```

#### `predict overlap`

Predict the expected number of sequences shared between two independent
samples of sizes d_i and d_j drawn from the same repertoire.

**Usage**

```text
lzg predict overlap GRAPH --di D_I --dj D_J [options]
```

| Flag | Description |
|------|-------------|
| `GRAPH` | `.lzg` file |
| `--di` | **(required)** Depth of sample i |
| `--dj` | **(required)** Depth of sample j |
| `-o, --output` | Output file (default: stdout) |
| `--json` | JSON output |

**Example**

```bash
lzg predict overlap repertoire.lzg --di 50000 --dj 50000
```

```text
PO	d_i	50000
PO	d_j	50000
PO	predicted_overlap	12847.3019
```

#### `predict sharing`

Predict the sharing spectrum: how many sequences are expected to appear in
exactly k out of N donors, given draw sizes for each donor.

**Usage**

```text
lzg predict sharing GRAPH --draws D1,D2,... [options]
```

| Flag | Description |
|------|-------------|
| `GRAPH` | `.lzg` file |
| `--draws` | **(required)** Comma-separated draw sizes, one per donor |
| `--max-k` | Maximum sharing degree (default: number of donors) |
| `-o, --output` | Output file (default: stdout) |
| `--json` | JSON output |

**Example**

```bash
lzg predict sharing repertoire.lzg --draws 10000,10000,10000
```

```text
k	expected_count
1	18421.201832
2	3814.490217
3	412.083104
```

---

### `posterior` -- Bayesian posterior update

Update a prior graph with new observations, producing a posterior graph.
This is the Bayesian mechanism for incorporating new sequencing data into
an existing model without rebuilding from scratch.

**Usage**

```text
lzg posterior PRIOR NEW_DATA -o OUTPUT [options]
```

**Arguments and options**

| Flag | Description |
|------|-------------|
| `PRIOR` | Prior `.lzg` graph file |
| `NEW_DATA` | File with new observations |
| `-o, --output` | **(required)** Output `.lzg` file for the posterior graph |
| `-s, --seq-column` | Sequence column name (default: auto-detect) |
| `-a, --abundance-column` | Abundance column name |
| `--kappa` | Prior strength / concentration parameter (default: `1.0`) |

**Example**

```bash
lzg posterior day0.lzg day30_repertoire.tsv -o day30_posterior.lzg --kappa 0.5
```

```text
[posterior] 21847 new sequences, kappa=0.5
[posterior] saved day30_posterior.lzg
```

!!! tip "Choosing kappa"
    `kappa` controls how much weight the prior receives relative to the new
    data. A value of `1.0` treats prior and data equally. Values below 1
    let the new data dominate; values above 1 make the posterior more
    conservative.

---

## Common workflows

These examples show how `lzg` commands chain together for typical analysis
tasks.

### Build and inspect

```bash
# Build the graph
lzg build repertoire.tsv -o rep.lzg

# Quick summary
lzg info rep.lzg

# Full details as JSON
lzg info rep.lzg --all --json > rep_info.json
```

### Build, simulate, and score

Generate synthetic sequences from a repertoire and then score them to verify
the generation probability distribution.

```bash
# Build
lzg build repertoire.tsv -o rep.lzg

# Simulate 10k sequences with details
lzg simulate rep.lzg -n 10000 --seed 42 --with-details -o synthetic.tsv

# Score an independent set against the same graph
lzg score rep.lzg test_sequences.txt -o scored.tsv
```

### Compare two repertoires

```bash
lzg build healthy.tsv -o healthy.lzg
lzg build disease.tsv -o disease.lzg

# Distributional distance + structural overlap
lzg compare healthy.lzg disease.lzg

# Side-by-side diversity profiles (JSON for scripting)
lzg diversity healthy.lzg --json > healthy_div.json
lzg diversity disease.lzg --json > disease_div.json
```

### Longitudinal tracking with posterior updates

```bash
# Day 0 baseline
lzg build day0.tsv -o day0.lzg

# Day 30: update the prior with new data
lzg posterior day0.lzg day30.tsv -o day30.lzg --kappa 1.0

# Day 60: chain another update
lzg posterior day30.lzg day60.tsv -o day60.lzg --kappa 1.0

# Compare baseline to final
lzg compare day0.lzg day60.lzg
```

### Saturation check before building

```bash
# Is 50k sequences enough?
lzg saturation repertoire.tsv --log-every 1000 -o sat.tsv

# If saturated, build the graph
lzg build repertoire.tsv -o rep.lzg
```

### Richness extrapolation

```bash
lzg build repertoire.tsv -o rep.lzg

# Predict how many unique sequences you'd see at 1M depth
lzg predict richness rep.lzg --depths 1000:1000000:20 -o richness.tsv
```

### Unix pipeline: decompose and count tokens

```bash
# Average number of LZ76 tokens per sequence
lzg decompose repertoire.tsv | tail -n +2 | awk -F'\t' '{sum+=$3; n++} END {print sum/n}'
```

```text
8.417
```
