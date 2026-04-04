#include "io_reader.h"
#include "io_writer.h"

LZGError lzg_graph_save(const LZGGraph *g, const char *path) {
    return lzg_graph_save_impl(g, path);
}

LZGError lzg_graph_load(const char *path, LZGGraph **out) {
    return lzg_graph_load_impl(path, out);
}
