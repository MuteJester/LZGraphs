/**
 * @file test_utils.h
 * @brief Shared test-runner macros for tests/c_unit.
 *
 * Each test file is expected to declare its own file-static counters:
 *   static int pass_count = 0, fail_count = 0;
 *
 * RUN_TEST prints the test name (left-padded), invokes it, and lets the
 * test body call PASS() or ASSERT_MSG() to update the counters.
 */
#ifndef LZGRAPH_TEST_UTILS_H
#define LZGRAPH_TEST_UTILS_H

#include <stdio.h>

#define RUN_TEST(fn) do { printf("  %-55s ", #fn); fn(); } while (0)

#define ASSERT_MSG(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); fail_count++; return; } \
} while (0)

#define PASS() do { printf("PASS\n"); pass_count++; } while (0)

#endif /* LZGRAPH_TEST_UTILS_H */
