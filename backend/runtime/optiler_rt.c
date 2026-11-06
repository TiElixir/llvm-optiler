/* Optiler runtime, linked into every model.so.
 *
 *  - memref alloc/free helpers (MLIR's generic-function lowering calls these), counting
 *    allocations and live/peak bytes so alloc_count and peak_bytes are *measured*;
 *  - cycle-counter probes used by the instrumented build (attribution = "probe").
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
static inline uint64_t optiler_now(void) { return __rdtsc(); }
#else
#include <time.h>
static inline uint64_t optiler_now(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
#endif

#define OPTILER_HDR 16 /* keeps the user pointer 16-byte aligned */
#define OPTILER_MAX_PROBES 4096

static uint64_t g_alloc_count, g_free_count, g_live_bytes, g_peak_bytes, g_total_bytes;
static uint64_t g_probe_cycles[OPTILER_MAX_PROBES];
static uint64_t g_probe_calls[OPTILER_MAX_PROBES];
static uint64_t g_probe_start[OPTILER_MAX_PROBES];

void *_mlir_memref_to_llvm_alloc(size_t size) {
  unsigned char *raw = (unsigned char *)malloc(size + OPTILER_HDR);
  if (!raw) return NULL;
  memcpy(raw, &size, sizeof(size));
  g_alloc_count++;
  g_total_bytes += size;
  g_live_bytes += size;
  if (g_live_bytes > g_peak_bytes) g_peak_bytes = g_live_bytes;
  return raw + OPTILER_HDR;
}

void *_mlir_memref_to_llvm_aligned_alloc(size_t alignment, size_t size) {
  (void)alignment; /* MLIR over-allocates and aligns manually when needed */
  return _mlir_memref_to_llvm_alloc(size);
}

void _mlir_memref_to_llvm_free(void *ptr) {
  if (!ptr) return;
  unsigned char *raw = (unsigned char *)ptr - OPTILER_HDR;
  size_t size;
  memcpy(&size, raw, sizeof(size));
  g_free_count++;
  g_live_bytes -= size;
  free(raw);
}

void optiler_stats_reset(void) {
  g_alloc_count = g_free_count = g_live_bytes = g_peak_bytes = g_total_bytes = 0;
}
/* out = {alloc_count, free_count, live_bytes, peak_bytes, total_bytes} */
void optiler_stats_get(uint64_t *out) {
  out[0] = g_alloc_count; out[1] = g_free_count; out[2] = g_live_bytes;
  out[3] = g_peak_bytes;  out[4] = g_total_bytes;
}

/* ------------------------------------------------------------- probes */
void __optiler_probe_enter(int32_t id) {
  if ((uint32_t)id < OPTILER_MAX_PROBES) g_probe_start[id] = optiler_now();
}
void __optiler_probe_exit(int32_t id) {
  if ((uint32_t)id < OPTILER_MAX_PROBES) {
    g_probe_cycles[id] += optiler_now() - g_probe_start[id];
    g_probe_calls[id]++;
  }
}
void optiler_probe_reset(void) {
  memset(g_probe_cycles, 0, sizeof(g_probe_cycles));
  memset(g_probe_calls, 0, sizeof(g_probe_calls));
}
uint64_t optiler_probe_cycles(int32_t id) { return g_probe_cycles[id]; }
uint64_t optiler_probe_calls(int32_t id) { return g_probe_calls[id]; }
uint64_t optiler_now_cycles(void) { return optiler_now(); }
