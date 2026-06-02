CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -std=c11 -Iinclude
LDFLAGS = -lm

# ── Release asset config (see `publish-foundation` target at bottom) ──
VERSION          := $(shell grep -oP '__version__\s*=\s*"\K[^"]+' src/LZGraphs/__init__.py)
TAG              ?= v$(VERSION)
FOUNDATION_GRAPH ?= .private/foundation_assets_2026-04-19/flashback_foundation.lzg
FOUNDATION_ASSET ?= flashback_foundation_$(TAG).lzg
GH_REPO_SLUG     ?= MuteJester/LZGraphs

# Collect all .c files under lib/
SRCS = $(shell find lib -name '*.c')
OBJS = $(patsubst lib/%.c, build/%.o, $(SRCS))

.PHONY: all clean test bench_ops publish-foundation

all: build/liblzgraph.a

# Static library
build/liblzgraph.a: $(OBJS)
	ar rcs $@ $^

# Compile each .c → .o, mirroring the lib/ directory structure in build/
build/%.o: lib/%.c | build_dirs
	$(CC) $(CFLAGS) -c $< -o $@

# Create build subdirectories matching lib/
build_dirs:
	@mkdir -p $(sort $(dir $(OBJS)))

# Test runners
test: test_core test_graph test_forward test_simulate test_analytics test_occupancy test_io test_posterior test_pgen_dist test_ndp test_gene_data test_sharing test_graph_ops test_diversity test_features test_genomic_simulate test_walk_dict

test_core: build/test_core
	./build/test_core

test_graph: build/test_graph
	./build/test_graph

build/test_core: tests/c_unit/test_core.c build/liblzgraph.a | build_dirs
	$(CC) $(CFLAGS) $< -Lbuild -llzgraph $(LDFLAGS) -o $@

build/test_graph: tests/c_unit/test_graph.c build/liblzgraph.a | build_dirs
	$(CC) $(CFLAGS) $< -Lbuild -llzgraph $(LDFLAGS) -o $@

test_forward: build/test_forward
	./build/test_forward

build/test_forward: tests/c_unit/test_forward.c build/liblzgraph.a | build_dirs
	$(CC) $(CFLAGS) $< -Lbuild -llzgraph $(LDFLAGS) -o $@

test_simulate: build/test_simulate
	./build/test_simulate

build/test_simulate: tests/c_unit/test_simulate.c build/liblzgraph.a | build_dirs
	$(CC) $(CFLAGS) $< -Lbuild -llzgraph $(LDFLAGS) -o $@

test_analytics: build/test_analytics
	./build/test_analytics

build/test_analytics: tests/c_unit/test_analytics.c build/liblzgraph.a | build_dirs
	$(CC) $(CFLAGS) $< -Lbuild -llzgraph $(LDFLAGS) -o $@

test_occupancy: build/test_occupancy
	./build/test_occupancy

build/test_occupancy: tests/c_unit/test_occupancy.c build/liblzgraph.a | build_dirs
	$(CC) $(CFLAGS) $< -Lbuild -llzgraph $(LDFLAGS) -o $@

test_io: build/test_io
	./build/test_io

build/test_io: tests/c_unit/test_io.c build/liblzgraph.a | build_dirs
	$(CC) $(CFLAGS) $< -Lbuild -llzgraph $(LDFLAGS) -o $@

test_posterior: build/test_posterior
	./build/test_posterior

build/test_posterior: tests/c_unit/test_posterior.c build/liblzgraph.a | build_dirs
	$(CC) $(CFLAGS) $< -Lbuild -llzgraph $(LDFLAGS) -o $@

test_pgen_dist: build/test_pgen_dist
	./build/test_pgen_dist

build/test_pgen_dist: tests/c_unit/test_pgen_dist.c build/liblzgraph.a | build_dirs
	$(CC) $(CFLAGS) $< -Lbuild -llzgraph $(LDFLAGS) -o $@

test_ndp: build/test_ndp
	./build/test_ndp

build/test_ndp: tests/c_unit/test_ndp.c build/liblzgraph.a | build_dirs
	$(CC) $(CFLAGS) $< -Lbuild -llzgraph $(LDFLAGS) -o $@

test_gene_data: build/test_gene_data
	./build/test_gene_data

build/test_gene_data: tests/c_unit/test_gene_data.c build/liblzgraph.a | build_dirs
	$(CC) $(CFLAGS) $< -Lbuild -llzgraph $(LDFLAGS) -o $@

test_sharing: build/test_sharing
	./build/test_sharing

build/test_sharing: tests/c_unit/test_sharing.c build/liblzgraph.a | build_dirs
	$(CC) $(CFLAGS) $< -Lbuild -llzgraph $(LDFLAGS) -o $@

test_graph_ops: build/test_graph_ops
	./build/test_graph_ops

build/test_graph_ops: tests/c_unit/test_graph_ops.c build/liblzgraph.a | build_dirs
	$(CC) $(CFLAGS) $< -Lbuild -llzgraph $(LDFLAGS) -o $@

test_diversity: build/test_diversity
	./build/test_diversity

build/test_diversity: tests/c_unit/test_diversity.c build/liblzgraph.a | build_dirs
	$(CC) $(CFLAGS) $< -Lbuild -llzgraph $(LDFLAGS) -o $@

test_features: build/test_features
	./build/test_features

build/test_features: tests/c_unit/test_features.c build/liblzgraph.a | build_dirs
	$(CC) $(CFLAGS) $< -Lbuild -llzgraph $(LDFLAGS) -o $@

test_genomic_simulate: build/test_genomic_simulate
	./build/test_genomic_simulate

build/test_genomic_simulate: tests/c_unit/test_genomic_simulate.c build/liblzgraph.a | build_dirs
	$(CC) $(CFLAGS) $< -Lbuild -llzgraph $(LDFLAGS) -o $@

test_walk_dict: build/test_walk_dict
	./build/test_walk_dict

build/test_walk_dict: tests/c_unit/test_walk_dict.c build/liblzgraph.a | build_dirs
	$(CC) $(CFLAGS) $< -Lbuild -llzgraph $(LDFLAGS) -o $@

clean:
	rm -rf build

bench_ops: build/bench_ops

build/bench_ops: tools/bench_ops.c build/liblzgraph.a | build_dirs
	$(CC) $(CFLAGS) $< -Lbuild -llzgraph $(LDFLAGS) -o $@

# ═══════════════════════════════════════════════════════════
# Release asset: manuscript foundation FlashBack graph
# ═══════════════════════════════════════════════════════════
# CI (build-wheels.yml) publishes to PyPI and creates the GitHub release on a
# `v*` tag push. The foundation graph is large (~328 MB) and lives in the
# gitignored .private/ tree, so CI never sees it; this target attaches it to
# the already-created release as a downloadable asset, with a SHA-256 sidecar.
# Run locally AFTER the tag's release exists.
#
#   make publish-foundation                            # TAG defaults to v$(VERSION)
#   make publish-foundation TAG=v3.1.0                 # explicit tag
#   make publish-foundation FOUNDATION_GRAPH=/path.lzg # different source graph
publish-foundation:
	@test -f "$(FOUNDATION_GRAPH)" || { echo "ERROR: foundation graph not found: $(FOUNDATION_GRAPH)"; exit 1; }
	@gh release view "$(TAG)" --repo "$(GH_REPO_SLUG)" >/dev/null 2>&1 || { echo "ERROR: release $(TAG) does not exist yet. Push the tag and let CI create the release first."; exit 1; }
	@echo "Computing SHA-256 of $(FOUNDATION_GRAPH) ($$(du -h "$(FOUNDATION_GRAPH)" | cut -f1))..."
	@sha256sum "$(FOUNDATION_GRAPH)" | awk -v n="$(FOUNDATION_ASSET)" '{print $$1"  "n}' > "/tmp/$(FOUNDATION_ASSET).sha256"
	@echo "Uploading to release $(TAG) as $(FOUNDATION_ASSET) ..."
	@gh release upload "$(TAG)" --repo "$(GH_REPO_SLUG)" --clobber \
		"$(FOUNDATION_GRAPH)#$(FOUNDATION_ASSET)" \
		"/tmp/$(FOUNDATION_ASSET).sha256#$(FOUNDATION_ASSET).sha256"
	@echo "Done -> https://github.com/$(GH_REPO_SLUG)/releases/download/$(TAG)/$(FOUNDATION_ASSET)"
