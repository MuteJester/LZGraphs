CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -std=c11 -Iinclude
LDFLAGS = -lm

# Collect all .c files under lib/
SRCS = $(shell find lib -name '*.c')
OBJS = $(patsubst lib/%.c, build/%.o, $(SRCS))

.PHONY: all clean test

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
test: test_core test_graph test_forward test_simulate test_analytics test_occupancy test_io_posterior test_pgen_dist test_variants test_gene_data test_sharing test_graph_ops test_diversity test_features test_genomic_simulate test_walk_dict

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

test_io_posterior: build/test_io_posterior
	./build/test_io_posterior

build/test_io_posterior: tests/c_unit/test_io_posterior.c build/liblzgraph.a | build_dirs
	$(CC) $(CFLAGS) $< -Lbuild -llzgraph $(LDFLAGS) -o $@

test_pgen_dist: build/test_pgen_dist
	./build/test_pgen_dist

build/test_pgen_dist: tests/c_unit/test_pgen_dist.c build/liblzgraph.a | build_dirs
	$(CC) $(CFLAGS) $< -Lbuild -llzgraph $(LDFLAGS) -o $@

test_variants: build/test_variants
	./build/test_variants

build/test_variants: tests/c_unit/test_variants.c build/liblzgraph.a | build_dirs
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
