#!/bin/sh
# Non-interactive release smoke test for the `lzg` CLI.
#
# Generates a small mock AIRR repertoire, then exercises validate-input,
# build, info, and simulate against it end to end. Used from two places in
# .github/workflows/release.yml: the wheel-smoke job (a built wheel
# installed into a clean venv) and the GHCR image smoke step (this same
# script run inside the container). Both need a plain, non-interactive
# script rather than the design-review kit at try_the_cli.sh, which pauses
# for a real terminal between sections (see its own header comment) and is
# meant for a human to watch, not for CI.
#
# Usage: LZG=lzg sh scripts/release_smoke.sh /path/to/scratch/dir
# LZG defaults to "lzg" (whatever resolves first on PATH); the release
# workflow points it at a specific venv's or container's binary instead.
set -eu

OUT="${1:?usage: release_smoke.sh <output-dir>}"
LZG="${LZG:-lzg}"

mkdir -p "$OUT"

echo "== lzg --version =="
$LZG --version

echo "== generating mock AIRR repertoire =="
python3 - "$OUT" <<'PY'
import random
import os
import sys

d = sys.argv[1]
random.seed(11)
AA = "ACDEFGHIKLMNPQRSTVWY"


def cdr3():
    body = "".join(random.choice(AA) for _ in range(random.randint(8, 22)))
    return "C" + body + "W"


rows = [cdr3() for _ in range(500)]
path = os.path.join(d, "smoke.tsv")
with open(path, "w") as f:
    f.write("sequence_id\tv_call\tj_call\tjunction_aa\tduplicate_count\tproductive\n")
    for i, s in enumerate(rows):
        f.write(
            f"seq{i}\tIGHV1-2*02\tIGHJ4*02\t{s}\t{random.randint(1, 50)}\tT\n"
        )
print(f"wrote {len(rows)} records to {path}")
PY

echo "== lzg validate-input =="
$LZG validate-input "$OUT/smoke.tsv"

echo "== lzg build =="
$LZG --ui plain build "$OUT/smoke.tsv" -o "$OUT/smoke.lzg"

echo "== lzg info =="
$LZG info "$OUT/smoke.lzg"

echo "== lzg simulate =="
$LZG simulate "$OUT/smoke.lzg" -n 5 --seed 1

test -s "$OUT/smoke.lzg"
echo "release smoke OK"
