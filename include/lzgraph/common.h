/**
 * @file common.h
 * @brief Shared types, macros, and platform abstractions for C-LZGraph.
 */
#ifndef LZGRAPH_COMMON_H
#define LZGRAPH_COMMON_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <math.h>

/* ── Result / error codes ───────────────────────────────────── */

typedef enum {
    LZG_OK = 0,

    /* Memory */
    LZG_ERR_ALLOC,                /* malloc/calloc/realloc failed   */

    /* Input validation */
    LZG_ERR_NULL_ARG,             /* required pointer is NULL       */
    LZG_ERR_EMPTY_INPUT,          /* empty sequence list or array   */
    LZG_ERR_INVALID_SEQUENCE,     /* bad characters or length       */
    LZG_ERR_INVALID_VARIANT,      /* unknown LZGVariant value       */
    LZG_ERR_LENGTH_MISMATCH,      /* array lengths don't match      */
    LZG_ERR_PARAM_OUT_OF_RANGE,   /* parameter outside valid range  */

    /* Graph state */
    LZG_ERR_NOT_BUILT,            /* graph not yet built/finalized  */
    LZG_ERR_NO_LIVE_PATHS,        /* no live initial states exist   */
    LZG_ERR_HAS_CYCLES,           /* graph is not a DAG             */

    /* Gene data */
    LZG_ERR_NO_GENE_DATA,         /* gene op on graph without genes */
    LZG_ERR_GENE_NOT_FOUND,       /* gene name not in string pool   */

    /* Graph operations */
    LZG_ERR_VARIANT_MISMATCH,     /* set op on different variants   */
    LZG_ERR_MISSING_EDGE,         /* walk requires a missing edge   */

    /* IO */
    LZG_ERR_IO_OPEN,              /* cannot open file               */
    LZG_ERR_IO_READ,              /* read error or unexpected EOF   */
    LZG_ERR_IO_WRITE,             /* write error                    */
    LZG_ERR_IO_CORRUPT,           /* CRC mismatch or bad structure  */
    LZG_ERR_IO_VERSION,           /* unsupported format version     */

    /* Numerical */
    LZG_ERR_CONVERGENCE,          /* iterative method did not converge */

    /* Internal (library bugs — should never reach the user) */
    LZG_ERR_OVERFLOW,             /* fixed-size buffer exceeded     */
    LZG_ERR_INTERNAL,             /* invariant violation (report!)  */

    /* Backward compatibility aliases (map old codes to new) */
    LZG_ERR_INVALID_ARG   = LZG_ERR_NULL_ARG,
    LZG_ERR_NOT_FINALIZED = LZG_ERR_NOT_BUILT,
    LZG_ERR_EMPTY_DATA    = LZG_ERR_EMPTY_INPUT,
    LZG_ERR_IO            = LZG_ERR_IO_OPEN,
} LZGError;

/* ── Thread-local error context ────────────────────────────── */

/**
 * Get the human-readable error message for the last error.
 * Returns "" if no error has been set. Thread-safe (thread-local storage).
 */
const char *lzg_error_message(void);

/**
 * Get the error code of the last error.
 */
LZGError lzg_last_error(void);

/**
 * Set the error context. Called internally before returning an error code.
 * Supports printf-style formatting.
 *
 * Usage:
 *   lzg_set_error(LZG_ERR_NULL_ARG, "sequences must not be NULL");
 *   return LZG_ERR_NULL_ARG;
 *
 * Or use the convenience macro:
 *   return LZG_FAIL(LZG_ERR_NULL_ARG, "sequences must not be NULL");
 */
void lzg_set_error(LZGError code, const char *fmt, ...);

/**
 * Clear the error state.
 */
void lzg_clear_error(void);

/**
 * Convenience macro: set error and return the code in one line.
 */
#define LZG_FAIL(code, ...) (lzg_set_error((code), __VA_ARGS__), (code))

/* ── Graph variant ──────────────────────────────────────────── */

typedef enum {
    LZG_VARIANT_AAP   = 0,  /* Amino Acid Positional           */
    LZG_VARIANT_NDP   = 1,  /* Nucleotide Double Positional    */
    LZG_VARIANT_NAIVE = 2,  /* Naive (no position encoding)    */
} LZGVariant;

/* ── Compiler hints ─────────────────────────────────────────── */

#if defined(__GNUC__) || defined(__clang__)
  #define LZG_LIKELY(x)   __builtin_expect(!!(x), 1)
  #define LZG_UNLIKELY(x) __builtin_expect(!!(x), 0)
  #define LZG_INLINE       static inline __attribute__((always_inline))
#else
  #define LZG_LIKELY(x)   (x)
  #define LZG_UNLIKELY(x) (x)
  #define LZG_INLINE       static inline
#endif

/* ── Logging ────────────────────────────────────────────────── */

/**
 * Log severity levels (libgit2-style severity ladder).
 *
 * Set the max level via lzg_log_set(). Messages above the threshold
 * are silently dropped (zero overhead if level < threshold).
 */
typedef enum {
    LZG_LOG_NONE  = 0,   /* logging disabled                          */
    LZG_LOG_ERROR = 1,   /* non-fatal errors worth reporting          */
    LZG_LOG_WARN  = 2,   /* recoverable issues (skipped seqs, slow)   */
    LZG_LOG_INFO  = 3,   /* progress and timing                       */
    LZG_LOG_DEBUG = 4,   /* algorithm decisions, internal metrics      */
    LZG_LOG_TRACE = 5,   /* per-item detail (very verbose)            */
} LZGLogLevel;

/**
 * Log callback signature.
 *
 * @param level   Severity of the message.
 * @param msg     Null-terminated message string.
 * @param data    User-provided context pointer (from lzg_log_set).
 */
typedef void (*lzg_log_fn)(LZGLogLevel level, const char *msg, void *data);

/**
 * Set the global log callback and maximum verbosity level.
 *
 * Only messages with level <= max_level are delivered to the callback.
 * Pass callback=NULL or max_level=LZG_LOG_NONE to disable logging.
 *
 * @param max_level  Maximum level to emit (LZG_LOG_NONE disables all).
 * @param callback   Function to receive messages. NULL = disable.
 * @param data       User context passed to every callback invocation.
 *
 * Example:
 *   void my_logger(LZGLogLevel lvl, const char *msg, void *data) {
 *       fprintf(stderr, "[LZG %d] %s\n", lvl, msg);
 *   }
 *   lzg_log_set(LZG_LOG_INFO, my_logger, NULL);
 */
void lzg_log_set(LZGLogLevel max_level, lzg_log_fn callback, void *data);

/**
 * Get the current log level (for fast short-circuit checks).
 */
LZGLogLevel lzg_log_level(void);

/* Internal: emit a log message (do not call directly — use macros). */
void lzg_log_emit(LZGLogLevel level, const char *fmt, ...);

/**
 * Log macros — the primary interface for library code.
 *
 * These check the level BEFORE formatting, so disabled levels have
 * zero cost (no vsnprintf call). Define LZG_NO_LOG at compile time
 * to eliminate all logging code entirely.
 *
 * Usage:
 *   LZG_INFO("graph built: %u nodes, %u edges (%.1f ms)", nn, ne, ms);
 *   LZG_WARN("sequence '%s' skipped: length 0", seq);
 *   LZG_DEBUG("string pool: %u slots, load=%.2f", cap, load);
 */
#ifdef LZG_NO_LOG
  #define LZG_LOG(lvl, ...) ((void)0)
#else
  #define LZG_LOG(lvl, ...) \
      do { if ((lvl) <= lzg_log_level()) lzg_log_emit((lvl), __VA_ARGS__); } while(0)
#endif

#define LZG_ERROR(...) LZG_LOG(LZG_LOG_ERROR, __VA_ARGS__)
#define LZG_WARN(...)  LZG_LOG(LZG_LOG_WARN,  __VA_ARGS__)
#define LZG_INFO(...)  LZG_LOG(LZG_LOG_INFO,  __VA_ARGS__)
#define LZG_DEBUG(...) LZG_LOG(LZG_LOG_DEBUG, __VA_ARGS__)
#define LZG_TRACE(...) LZG_LOG(LZG_LOG_TRACE, __VA_ARGS__)

/* ── Numerical constants ────────────────────────────────────── */

#define LZG_EPS       1e-300
#define LZG_LOG_EPS  (-690.7755278982137)  /* log(1e-300) */
#define LZG_NAN       ((double)NAN)

/* ── Alphabet ──────────────────────────────────────────────── */

#define LZG_AA_COUNT  20

/** Sentinel characters for sequence boundaries. */
#define LZG_START_SENTINEL '@'
#define LZG_END_SENTINEL   '$'

/**
 * Map a character to a bit index for the walk dictionary bitmask.
 *
 * Bits 0-19:  Standard amino acids (A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y)
 * Bit 20:     Start sentinel '@'
 * Bit 21:     End sentinel '$'
 * Bit 31:     Non-standard (ignored in bitmask checks)
 */
uint8_t lzg_aa_to_bit(char c);  /* implemented in lib/core/common.c */

/**
 * Compute the character bitmask for a string of length `len`.
 * Each amino acid sets its corresponding bit in the uint32.
 */
LZG_INLINE uint32_t lzg_aa_bitmask(const char *s, uint32_t len) {
    uint32_t mask = 0;
    for (uint32_t i = 0; i < len; i++)
        mask |= (1u << lzg_aa_to_bit(s[i]));
    return mask;
}

#endif /* LZGRAPH_COMMON_H */
