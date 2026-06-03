#include "fast_rand.h"

#if defined(_MSC_VER)
#define FAST_TMAE_THREAD_LOCAL __declspec(thread)
#elif defined(__GNUC__) || defined(__clang__)
#define FAST_TMAE_THREAD_LOCAL __thread
#else
#define FAST_TMAE_THREAD_LOCAL _Thread_local
#endif

// Seed/state for the RNG.
static FAST_TMAE_THREAD_LOCAL uint64_t xorshift_state[2] = {0xcafef00dbadc0ffeULL, 0xdeadbeef12345678ULL};


// Seeding function.
void xorshift128p_seed(uint64_t seed) {
    xorshift_state[0] = seed;
    xorshift_state[1] = ~seed;  // Just for some variety. You can use any other method you like.
}

// Fast XOR-Shift128+ RNG.
uint32_t xorshift128p_fast() {
    uint64_t s1 = xorshift_state[0];
    const uint64_t s0 = xorshift_state[1];
    xorshift_state[0] = s0;
    s1 ^= s1 << 23;
    xorshift_state[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);

    return (uint32_t)(xorshift_state[1] + s0);
}


