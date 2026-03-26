/**
 * @file common.c
 * @brief Shared utility implementations: amino acid mapping, error context.
 */
#include "lzgraph/common.h"
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

/* ── Thread-local error context ────────────────────────────── */

#define LZG_ERR_MSG_SIZE 512

/* Thread-local storage: C11 _Thread_local, or MSVC __declspec(thread) */
#ifdef _MSC_VER
  #define TLS __declspec(thread)
#else
  #define TLS _Thread_local
#endif

static TLS LZGError  tl_error_code = LZG_OK;
static TLS char      tl_error_msg[LZG_ERR_MSG_SIZE] = "";

const char *lzg_error_message(void) {
    return tl_error_msg;
}

LZGError lzg_last_error(void) {
    return tl_error_code;
}

void lzg_set_error(LZGError code, const char *fmt, ...) {
    tl_error_code = code;
    if (fmt) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(tl_error_msg, LZG_ERR_MSG_SIZE, fmt, ap);
        va_end(ap);
    } else {
        tl_error_msg[0] = '\0';
    }
}

void lzg_clear_error(void) {
    tl_error_code = LZG_OK;
    tl_error_msg[0] = '\0';
}

/* ── Logging ────────────────────────────────────────────────── */

static LZGLogLevel  g_log_level = LZG_LOG_NONE;
static lzg_log_fn   g_log_cb    = NULL;
static void         *g_log_data  = NULL;

void lzg_log_set(LZGLogLevel max_level, lzg_log_fn callback, void *data) {
    g_log_level = callback ? max_level : LZG_LOG_NONE;
    g_log_cb    = callback;
    g_log_data  = data;
}

LZGLogLevel lzg_log_level(void) {
    return g_log_level;
}

void lzg_log_emit(LZGLogLevel level, const char *fmt, ...) {
    if (!g_log_cb || level > g_log_level) return;
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    g_log_cb(level, buf, g_log_data);
}

/* ── Amino acid mapping ────────────────────────────────────── */


uint8_t lzg_aa_to_bit(char c) {
    switch (c) {
        case 'A': return  0; case 'C': return  1; case 'D': return  2;
        case 'E': return  3; case 'F': return  4; case 'G': return  5;
        case 'H': return  6; case 'I': return  7; case 'K': return  8;
        case 'L': return  9; case 'M': return 10; case 'N': return 11;
        case 'P': return 12; case 'Q': return 13; case 'R': return 14;
        case 'S': return 15; case 'T': return 16; case 'V': return 17;
        case 'W': return 18; case 'Y': return 19;
        case '@': return 20; /* start sentinel */
        case '$': return 21; /* end sentinel */
        default:  return 31; /* non-standard → high bit, ignored in bitmask */
    }
}
