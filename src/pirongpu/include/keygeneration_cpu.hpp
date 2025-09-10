#pragma once
// #include "common.cuh"
// #include "cuda_runtime.h"
#include "ntt_cpu.hpp"
#include "context_cpu.hpp"

namespace heoncpu
{
    // Secret Key Generation

    // __global__ void secretkey_gen_kernel(int* secret_key, int hamming_weight,
    //                                      int n_power, int seed);
    void secretkey_gen_cpu(int* secret_key, int hamming_weight, int n_power, int seed,
                       int grid_x, int grid_y, int grid_z, int block_size);

    // __global__ void secretkey_rns_kernel(int* input, Data64* output,
    //                                      Modulus64* modulus, int n_power,
    //                                      int rns_mod_count);
    void secretkey_rns_cpu(int* input, Data64* output, Modulus64* modulus, 
                       int n_power, int rns_mod_count,
                       int grid_x, int grid_y, int grid_z, int block_size);

    // Public Key Generation

    // __global__ void publickey_gen_kernel(Data64* public_key, Data64* secret_key,
    //                                      Data64* error_poly, Data64* a_poly,
    //                                      Modulus64* modulus, int n_power,
    //                                      int rns_mod_count);
    void publickey_gen_cpu(
        Data64* public_key, Data64* secret_key, Data64* error_poly, Data64* a_poly,
        Modulus64* modulus, int n_power, int rns_mod_count);

    // __global__ void threshold_pk_addition(Data64* pk1, Data64* pk2,
    //                                       Data64* pkout, Modulus64* modulus,
    //                                       int n_power, bool first);

    // // Relinearization Key Generation

    // __global__ void relinkey_gen_kernel(Data64* relin_key, Data64* secret_key,
    //                                     Data64* error_poly, Data64* a_poly,
    //                                     Modulus64* modulus, Data64* factor,
    //                                     int n_power, int rns_mod_count);

    // __global__ void multi_party_relinkey_piece_method_I_stage_I_kernel(
    //     Data64* rk, Data64* sk, Data64* a, Data64* u, Data64* e0, Data64* e1,
    //     Modulus64* modulus, Data64* factor, int n_power, int rns_mod_count);

    // __global__ void multi_party_relinkey_piece_method_I_II_stage_II_kernel(
    //     Data64* rk_1, Data64* rk_2, Data64* sk, Data64* u, Data64* e0,
    //     Data64* e1, Modulus64* modulus, int n_power, int rns_mod_count,
    //     int decomp_mod_count);

    // __global__ void multi_party_relinkey_piece_method_II_stage_I_kernel(
    //     Data64* rk, Data64* sk, Data64* a, Data64* u, Data64* e0, Data64* e1,
    //     Modulus64* modulus, Data64* factor, int* Sk_pair, int n_power,
    //     int l_tilda, int d, int Q_size, int P_size);

    // __global__ void multi_party_relinkey_method_I_stage_I_kernel(
    //     Data64* rk_in, Data64* rk_out, Modulus64* modulus, int n_power,
    //     int Q_prime_size, int l, bool first);

    // __global__ void multi_party_relinkey_method_I_stage_II_kernel(
    //     Data64* rk1, Data64* rk2, Data64* rk_out, Modulus64* modulus,
    //     int n_power, int Q_prime_size, int l);

    // __global__ void multi_party_relinkey_method_I_stage_II_kernel(
    //     Data64* rk_in, Data64* rk_out, Modulus64* modulus, int n_power,
    //     int Q_prime_size, int l);

    // __global__ void relinkey_gen_II_kernel(
    //     Data64* relin_key_temp, Data64* secret_key, Data64* error_poly,
    //     Data64* a_poly, Modulus64* modulus, Data64* factor, int* Sk_pair,
    //     int n_power, int l_tilda, int d, int Q_size, int P_size);

    // __global__ void relinkey_gen_II_leveled_kernel(
    //     Data64* relin_key_temp, Data64* secret_key, Data64* error_poly,
    //     Data64* a_poly, Modulus64* modulus, Data64* factor, int* Sk_pair,
    //     int n_power, int l_tilda, int d, int Q_size, int P_size,
    //     int* mod_index);

    // __global__ void relinkey_DtoB_kernel(
    //     Data64* relin_key_temp, Data64* relin_key, Modulus64* modulus,
    //     Modulus64* B_base, Data64* base_change_matrix_D_to_B,
    //     Data64* Mi_inv_D_to_B, Data64* prod_D_to_B, int* I_j_, int* I_location_,
    //     int n_power, int l_tilda, int d_tilda, int d, int r_prime);

    // __global__ void relinkey_DtoB_leveled_kernel(
    //     Data64* relin_key_temp, Data64* relin_key, Modulus64* modulus,
    //     Modulus64* B_base, Data64* base_change_matrix_D_to_B,
    //     Data64* Mi_inv_D_to_B, Data64* prod_D_to_B, int* I_j_, int* I_location_,
    //     int n_power, int l_tilda, int d_tilda, int d, int r_prime,
    //     int* mod_index);

    // // Galois Key Generation

    int steps_to_galois_elt(int steps, int coeff_count, int group_order);

    // __device__ int bitreverse_gpu(int index, int n_power);

    int permutation(int index, int galois_elt, int coeff_count,
                               int n_power);

    void galoiskey_gen_cpu(
        Data64* galois_key, Data64* secret_key, Data64* error_poly, Data64* a_poly,
        Modulus64* modulus, Data64* factor, int galois_elt, int n_power, int rns_mod_count,
        int grid_x, int grid_y, int grid_z, int block_size);

    // __global__ void galoiskey_gen_kernel(Data64* galois_key, Data64* secret_key,
    //                                      Data64* error_poly, Data64* a_poly,
    //                                      Modulus64* modulus, Data64* factor,
    //                                      int galois_elt, int n_power,
    //                                      int rns_mod_count);

    // __global__ void galoiskey_gen_II_kernel(
    //     Data64* galois_key_temp, Data64* secret_key, Data64* error_poly,
    //     Data64* a_poly, Modulus64* modulus, Data64* factor, int galois_elt,
    //     int* Sk_pair, int n_power, int l_tilda, int d, int Q_size, int P_size);
    void galoiskey_gen_II_cpu(
            Data64* galois_key_temp, Data64* secret_key, Data64* error_poly, Data64* a_poly,
            Modulus64* modulus, Data64* factor, int galois_elt, int* Sk_pair, int n_power,
            int l_tilda, int d, int Q_size, int P_size,
            int grid_x, int grid_y, int grid_z, int block_size);
    // __global__ void multi_party_galoiskey_gen_method_I_II_kernel(
    //     Data64* gk_1, Data64* gk_2, Modulus64* modulus, int n_power,
    //     int rns_mod_count, int decomp_mod_count, bool first);

    // // Switch Key Generation

    // __global__ void switchkey_gen_kernel(Data64* switch_key,
    //                                      Data64* new_secret_key,
    //                                      Data64* old_secret_key,
    //                                      Data64* error_poly, Data64* a_poly,
    //                                      Modulus64* modulus, Data64* factor,
    //                                      int n_power, int rns_mod_count);

    // __global__ void switchkey_gen_II_kernel(
    //     Data64* switch_key, Data64* new_secret_key, Data64* old_secret_key,
    //     Data64* error_poly, Data64* a_poly, Modulus64* modulus, Data64* factor,
    //     int* Sk_pair, int n_power, int l_tilda, int d, int Q_size, int P_size);

    // __global__ void switchkey_kernel(Data64* switch_key, Data64* new_secret_key,
    //                                  Data64* old_secret_key, Data64* e_a,
    //                                  Modulus64* modulus, Data64* factor,
    //                                  int n_power, int rns_mod_count);

} // namespace heongpu 