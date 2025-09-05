#pragma once
// #include <curand_kernel.h>
#include "context_cpu.hpp"

namespace heoncpu
{
    // __global__ void modular_uniform_random_number_generation_kernel(
    //     Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
    //     int seed, int offset);
    void modular_uniform_random_number_generation_cpu(
        Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
        int seed, int offset);
    // __global__ void modular_uniform_random_number_generation_kernel(
    //     Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
    //     int seed, int offset, int* mod_index);

    void modular_gaussian_random_number_generation_cpu(
        Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
        int seed, int offset);

    // __global__ void modular_gaussian_random_number_generation_kernel(
    //     Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
    //     int seed, int offset, int* mod_index);

    // __global__ void modular_ternary_random_number_generation_kernel(
    //     Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
    //     int seed, int offset);

    // __global__ void modular_ternary_random_number_generation_kernel(
    //     Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
    //     int seed, int offset, int* mod_index);

} // namespace heongpu 