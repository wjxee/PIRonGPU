#pragma once
#include "context_cpu.hpp"

namespace heoncpu{
    void sk_multiplication_cpu(
        Data64* ct1, Data64* sk, Data64* output,
        Modulus64* modulus, int n_power,
        int decomp_mod_count,
        int grid_x, int grid_y, int grid_z, int block_size);

    void decryption_kernel_cpu(
        Data64* ct0, Data64* ct1, Data64* plain,
        Modulus64* modulus, Modulus64 plain_mod,
        Modulus64 gamma, Data64* Qi_t,
        Data64* Qi_gamma, Data64* Qi_inverse,
        Data64 mulq_inv_t, Data64 mulq_inv_gamma,
        Data64 inv_gamma, int n_power,
        int decomp_mod_count,
        int grid_x, int grid_y, int grid_z, int block_size);    

}