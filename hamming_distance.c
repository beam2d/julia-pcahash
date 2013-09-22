#include <smmintrin.h>
#include <stdint.h>

int hamming_distance(uint64_t* x, uint64_t* y, int64_t k) {
  int64_t niter = (k + 63) / 64;
  int accum = 0;
  for (int64_t i = 0; i < niter; ++i) {
    accum += _mm_popcnt_u64(x[i] ^ y[i]);
  }
  return accum;
}
