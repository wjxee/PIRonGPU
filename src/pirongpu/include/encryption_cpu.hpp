#pragma once
// #include "common.cuh"
// #include "cuda_runtime.h"
#include "context_cpu.hpp"

namespace heoncpu
{
    // __global__ void pk_u_kernel(Data64* pk, Data64* u, Data64* pk_u,
    //                             Modulus64* modulus, int n_power,
    //                             int rns_mod_count);
    void pk_u_kernel_cpu(
        Data64* pk,          // 输入：公钥数据
        Data64* u,           // 输入：u多项式
        Data64* pk_u,        // 输出：pk*u结果
        Modulus64* modulus,  // 模数数组
        int n_power,         // 位移参数（2^n_power）
        int rns_mod_count,   // RNS模数数量（Q_prime_size_）
        int grid_x,          // 网格X维度（块数量）
        int grid_y,          // 网格Y维度（块数量）
        int grid_z,          // 网格Z维度（块数量）
        int block_size       // 线程块大小（线程数）
    );
    // __global__ void
    // enc_div_lastq_bfv_kernel(Data64* pk, Data64* e, Data64* plain, Data64* ct,
    //                          Modulus64* modulus, Data64* half, Data64* half_mod,
    //                          Data64* last_q_modinv, Modulus64 plain_mod,
    //                          Data64 Q_mod_t, Data64 upper_threshold,
    //                          Data64* coeffdiv_plain, int n_power,
    //                          int Q_prime_size, int Q_size, int P_size);
    void enc_div_lastq_bfv_cpu(Data64* pk, Data64* e, Data64* plain, Data64* ct,
                                Modulus64* modulus, Data64* half, Data64* half_mod,
                                Data64* last_q_modinv, Modulus64 plain_mod,
                                Data64 Q_mod_t, Data64 upper_threshold,
                                Data64* coeffdiv_plain, int n_power,
                                int Q_prime_size, int Q_size, int P_size,
                                int grid_x, int grid_y, int grid_z, int block_size);
    // __global__ void enc_div_lastq_ckks_kernel(Data64* pk, Data64* e, Data64* ct,
    //                                           Modulus64* modulus, Data64* half,
    //                                           Data64* half_mod,
    //                                           Data64* last_q_modinv,
    //                                           int n_power, int Q_prime_size,
    //                                           int Q_size, int P_size);

    // __global__ void cipher_message_add_kernel(Data64* ciphertext,
    //                                           Data64* plaintext,
    //                                           Modulus64* modulus, int n_power);

} // namespace heongpu 