#!/usr/bin/env bash
# Manual design review kit for the redesigned lzg CLI.
#
#   bash try_the_cli.sh
#
# Builds mock BCR repertoire data in a temp dir, then walks the CLI so you can
# judge the visual design. Everything is cleaned up on exit.
#
# Run this in a REAL terminal, not through a pipe: the live panel only appears
# when stderr is a TTY. Resize the window between sections to see it reflow.
set -u

cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src"
LZG="python3 -m LZGraphs.cli"
D=$(mktemp -d); trap 'rm -rf "$D"' EXIT

pause(){ printf '\n\033[2m   [enter to continue]\033[0m'; read -r _ </dev/tty; printf '\n'; }
hdr(){ printf '\n\033[1m%s\033[0m\n\033[2m   %s\033[0m\n\n' "$1" "$2"; }

printf '\n\033[1mlzg design review\033[0m  \033[2m(version %s)\033[0m\n' "$($LZG --version | awk '{print $2}')"
printf '\033[2mmock BCR data in %s\033[0m\n' "$D"

# ---------------------------------------------------------------- mock data
python3 - "$D" <<'PY'
import gzip, os, random, sys
d = sys.argv[1]
random.seed(11)
AA = "ACDEFGHIKLMNPQRSTVWY"
def cdr3():
    return "C" + "".join(random.choice(AA) for _ in range(random.randint(8, 22))) + "W"

# small AIRR TSV, for reading the panel at rest
small = [cdr3() for _ in range(800)]
with open(os.path.join(d, "small.tsv"), "w") as f:
    f.write("sequence_id\tv_call\tj_call\tjunction_aa\tduplicate_count\tproductive\n")
    for i, s in enumerate(small):
        f.write(f"seq{i}\tIGHV1-2*02\tIGHJ4*02\t{s}\t{random.randint(1,50)}\t"
                f"{'T' if i % 37 else 'F'}\n")

# large PLAIN UNCOMPRESSED file: this is the one that shows a live progress bar
with open(os.path.join(d, "big.txt"), "w") as f:
    for _ in range(400000):
        f.write(cdr3() + "\n")

# FASTA, the format that used to ingest '>' headers as sequences
with open(os.path.join(d, "repertoire.fa"), "w") as f:
    for i, s in enumerate(small):
        f.write(f">seq{i}\n{s}\n")

# gzipped AIRR TSV
with gzip.open(os.path.join(d, "small.tsv.gz"), "wt") as f:
    f.write(open(os.path.join(d, "small.tsv")).read())

# a file with problems, to see the warning and error UX
with open(os.path.join(d, "messy.tsv"), "w") as f:
    f.write("junction_aa\tduplicate_count\tproductive\n")
    for i, s in enumerate(small[:200]):
        seq = s if i % 25 else "CASS!!BROKEN"
        f.write(f"{seq}\t{random.randint(1,9)}\t{'T' if i % 11 else 'F'}\n")

# nucleotide data fed to an amino-acid engine, to see the alphabet warning
with open(os.path.join(d, "nucleotide.txt"), "w") as f:
    for _ in range(300):
        f.write("".join(random.choice("ACGT") for _ in range(60)) + "\n")
PY

hdr "1. inspect before building" "lzg validate-input small.tsv"
$LZG validate-input "$D/small.tsv"; pause

hdr "2. the live panel, with a real progress bar" \
    "lzg build big.txt   (400k sequences, plain uncompressed: this is what animates)"
$LZG build "$D/big.txt" -o "$D/big.lzg"; pause

hdr "3. an AIRR TSV, with non-productive rows dropped" "lzg build small.tsv"
$LZG build "$D/small.tsv" -o "$D/small.lzg"; pause

hdr "4. FASTA (used to ingest '>' headers) and gzip, detected by content" \
    "lzg build repertoire.fa   /   lzg build small.tsv.gz"
$LZG build "$D/repertoire.fa" -o "$D/fa.lzg"
$LZG build "$D/small.tsv.gz" -o "$D/gz.lzg"; pause

hdr "5. warnings: malformed and non-productive records" "lzg build messy.tsv"
$LZG build "$D/messy.tsv" -o "$D/messy.lzg"; pause

hdr "6. the alphabet warning" "lzg build nucleotide.txt -V aap"
$LZG build "$D/nucleotide.txt" -o "$D/nt.lzg" -V aap; pause

hdr "7. an error" "lzg build small.tsv --seq-column NOPE"
$LZG build "$D/small.tsv" -o "$D/x.lzg" --seq-column NOPE; pause

hdr "8. the same build in CI / pipe mode" "lzg --ui plain build small.tsv"
$LZG --ui plain build "$D/small.tsv" -o "$D/p.lzg"; pause

hdr "9. simulate, and proof that stdout stays clean" \
    "lzg simulate small.lzg -n 5   then   | wc -l  (no escape codes in a pipe)"
$LZG simulate "$D/small.lzg" -n 5 --seed 1
printf '\n\033[2m   piped through wc -l:\033[0m '
$LZG simulate "$D/small.lzg" -n 1000 --seed 1 2>/dev/null | wc -l

printf '\n\033[1mthings worth trying yourself\033[0m\n'
cat <<EOF

   resize the terminal narrow, then rerun section 2
   lzg --ui quiet build $D/small.tsv -o /tmp/q.lzg      (silent unless it fails)
   NO_COLOR=1 lzg build $D/small.tsv -o /tmp/n.lzg      (no colour, still a panel)
   lzg build $D/small.tsv -o /tmp/r.lzg 2>&1 | cat      (auto-degrades to plain)
   lzg build $D/big.txt -o /tmp/i.lzg                   (then hit ctrl-c: cursor must come back)

EOF
