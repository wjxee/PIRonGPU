 
#include "encryption_cpu.hpp"

namespace heoncpu
{
    // __global__ void pk_u_kernel(Data64* pk, Data64* u, Data64* pk_u,
    //                             Modulus64* modulus, int n_power,
    //                             int rns_mod_count)
    // {
    //     int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
    //     int block_y = blockIdx.y; // rns_mod_count
    //     int block_z = blockIdx.z; // 2

    //     Data64 pk_ = pk[idx + (block_y << n_power) +
    //                     ((rns_mod_count << n_power) * block_z)];
    //     Data64 u_ = u[idx + (block_y << n_power)];

    //     Data64 pk_u_ = OPERATOR_GPU_64::mult(pk_, u_, modulus[block_y]);

    //     pk_u[idx + (block_y << n_power) +
    //          ((rns_mod_count << n_power) * block_z)] = pk_u_;
    // }
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
    ) {
        // 模拟网格和线程块执行
        for (int block_z = 0; block_z < grid_z; ++block_z) {       // Z维度循环
            for (int block_y = 0; block_y < grid_y; ++block_y) {    // Y维度循环
                for (int block_x = 0; block_x < grid_x; ++block_x) { // X维度循环
                    for (int thread_idx = 0; thread_idx < block_size; ++thread_idx) { // 线程循环
                        // 计算全局索引（等效idx = blockIdx.x * blockDim.x + threadIdx.x）
                        int idx = block_x * block_size + thread_idx;
                        
                        // 计算数据偏移（等效CUDA索引）
                        int pk_offset = idx + (block_y << n_power) + ((rns_mod_count << n_power) * block_z);
                        int u_offset = idx + (block_y << n_power); // 注意：u无block_z项
                        int pk_u_offset = pk_offset; // 输出偏移同pk
                        
                        // 执行模乘：pk_u = pk * u mod modulus
                        Data64 pk_val = pk[pk_offset];
                        Data64 u_val = u[u_offset];
                        Data64 pk_u_val = OPERATOR64::mult(pk_val, u_val, modulus[block_y]);
                        
                        // 写入结果
                        pk_u[pk_u_offset] = pk_u_val;
                    }
                }
            }
        }
    }
    // __global__ void enc_div_lastq_kernel(
    //     Data64* pk, Data64* e, Data64* plain, Data64* ct, Modulus64* modulus,
    //     Data64 half, Data64* half_mod, Data64* last_q_modinv,
    //     Modulus64 plain_mod, Data64 Q_mod_t, Data64 upper_threshold,
    //     Data64* coeffdiv_plain, int n_power, int decomp_mod_count)
    // {
    //     int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
    //     int block_y = blockIdx.y; // Decomposition Modulus Count
    //     int block_z = blockIdx.z; // Cipher Size (2)

    //     Data64 last_pk = pk[idx + (decomp_mod_count << n_power) +
    //                         (((decomp_mod_count + 1) << n_power) * block_z)];
    //     Data64 last_e = e[idx + (decomp_mod_count << n_power) +
    //                       (((decomp_mod_count + 1) << n_power) * block_z)];

    //     last_pk =
    //         OPERATOR_GPU_64::add(last_pk, last_e, modulus[decomp_mod_count]);

    //     last_pk =
    //         OPERATOR_GPU_64::add(last_pk, half, modulus[decomp_mod_count]);

    //     Data64 zero_ = 0;
    //     last_pk = OPERATOR_GPU_64::add(last_pk, zero_, modulus[block_y]);

    //     last_pk =
    //         OPERATOR_GPU_64::sub(last_pk, half_mod[block_y], modulus[block_y]);

    //     Data64 input_ = pk[idx + (block_y << n_power) +
    //                        (((decomp_mod_count + 1) << n_power) * block_z)];

    //     //
    //     Data64 e_ = e[idx + (block_y << n_power) +
    //                   (((decomp_mod_count + 1) << n_power) * block_z)];
    //     input_ = OPERATOR_GPU_64::add(input_, e_, modulus[block_y]);
    //     //

    //     input_ = OPERATOR_GPU_64::sub(input_, last_pk, modulus[block_y]);

    //     input_ = OPERATOR_GPU_64::mult(input_, last_q_modinv[block_y],
    //                                    modulus[block_y]);

    //     if (block_z == 0)
    //     {
    //         Data64 message = plain[idx];
    //         Data64 fix = message * Q_mod_t;
    //         fix = fix + upper_threshold;
    //         fix = int(fix / plain_mod.value);

    //         Data64 ct_0 = OPERATOR_GPU_64::mult(
    //             message, coeffdiv_plain[block_y], modulus[block_y]);
    //         ct_0 = OPERATOR_GPU_64::add(ct_0, fix, modulus[block_y]);

    //         input_ = OPERATOR_GPU_64::add(input_, ct_0, modulus[block_y]);

    //         ct[idx + (block_y << n_power) +
    //            (((decomp_mod_count) << n_power) * block_z)] = input_;
    //     }
    //     else
    //     {
    //         ct[idx + (block_y << n_power) +
    //            (((decomp_mod_count) << n_power) * block_z)] = input_;
    //     }
    // }
    void enc_div_lastq_bfv_cpu(Data64* pk, Data64* e, Data64* plain, Data64* ct,
                            Modulus64* modulus, Data64* half, Data64* half_mod,
                            Data64* last_q_modinv, Modulus64 plain_mod,
                            Data64 Q_mod_t, Data64 upper_threshold,
                            Data64* coeffdiv_plain, int n_power,
                            int Q_prime_size, int Q_size, int P_size,
                            int grid_x, int grid_y, int grid_z, int block_size) {
        
        // 模拟GPU的网格和块结构
        for (int block_z = 0; block_z < grid_z; block_z++) {
            for (int block_y = 0; block_y < grid_y; block_y++) {
                for (int block_x = 0; block_x < grid_x; block_x++) {
                    for (int thread_idx = 0; thread_idx < block_size; thread_idx++) {
                        // 计算全局索引
                        int idx = block_x * block_size + thread_idx;
                        
                        // 检查索引是否越界（根据实际情况调整）
                        // 这里假设索引总是有效的
                        
                        // Max P size is 15.
                        Data64 last_pk[15]={0};
                        for (int i = 0; i < P_size; i++) {
                            Data64 last_pk_ = pk[idx + ((Q_size + i) << n_power) +
                                                ((Q_prime_size << n_power) * block_z)];
                            Data64 last_e_ = e[idx + ((Q_size + i) << n_power) +
                                            ((Q_prime_size << n_power) * block_z)];
                            last_pk[i] = OPERATOR64::add(last_pk_, last_e_, modulus[Q_size + i]);
                        }
                        
                        Data64 input_ = pk[idx + (block_y << n_power) +
                                        ((Q_prime_size << n_power) * block_z)];
                        Data64 e_ = e[idx + (block_y << n_power) +
                                    ((Q_prime_size << n_power) * block_z)];
                        input_ = OPERATOR64::add(input_, e_, modulus[block_y]);
                        
                        Data64 zero_ = 0;
                        int location_ = 0;
                        
                        for (int i = 0; i < P_size; i++) {
                            Data64 last_pk_add_half_ = last_pk[(P_size - 1 - i)];
                            last_pk_add_half_ = OPERATOR64::add(
                                last_pk_add_half_, half[i], modulus[(Q_prime_size - 1 - i)]);
                            
                            for (int j = 0; j < (P_size - 1 - i); j++) {
                                Data64 temp1 = OPERATOR64::add(last_pk_add_half_, zero_,
                                                                    modulus[Q_size + j]);
                                temp1 = OPERATOR64::sub(temp1,
                                                            half_mod[location_ + Q_size + j],
                                                            modulus[Q_size + j]);
                                temp1 = OPERATOR64::sub(last_pk[j], temp1,
                                                            modulus[Q_size + j]);
                                last_pk[j] = OPERATOR64::mult(
                                    temp1, last_q_modinv[location_ + Q_size + j],
                                    modulus[Q_size + j]);
                            }
                            
                            Data64 temp1 = OPERATOR64::add(last_pk_add_half_, zero_,
                                                                modulus[block_y]);
                            temp1 = OPERATOR64::sub(temp1, half_mod[location_ + block_y],
                                                        modulus[block_y]);
                            temp1 = OPERATOR64::sub(input_, temp1, modulus[block_y]);
                            input_ = OPERATOR64::mult(
                                temp1, last_q_modinv[location_ + block_y], modulus[block_y]);
                            location_ += Q_prime_size - 1 - i;
                        }
                        
                        if (block_z == 0) {
                            Data64 message = plain[idx];
                            Data64 fix = message * Q_mod_t;
                            fix = fix + upper_threshold;
                            fix = static_cast<Data64>(fix / plain_mod.value);
                            Data64 ct_0 = OPERATOR64::mult(
                                message, coeffdiv_plain[block_y], modulus[block_y]);
                            ct_0 = OPERATOR64::add(ct_0, fix, modulus[block_y]);
                            input_ = OPERATOR64::add(input_, ct_0, modulus[block_y]);
                            ct[idx + (block_y << n_power) + (((Q_size) << n_power) * block_z)] = input_;
                        } else {
                            ct[idx + (block_y << n_power) + (((Q_size) << n_power) * block_z)] = input_;
                        }
                    }
                }
            }
        }
    }
    // __global__ void
    // enc_div_lastq_bfv_kernel(Data64* pk, Data64* e, Data64* plain, Data64* ct,
    //                          Modulus64* modulus, Data64* half, Data64* half_mod,
    //                          Data64* last_q_modinv, Modulus64 plain_mod,
    //                          Data64 Q_mod_t, Data64 upper_threshold,
    //                          Data64* coeffdiv_plain, int n_power,
    //                          int Q_prime_size, int Q_size, int P_size)
    // {
    //     int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
    //     int block_y = blockIdx.y; // Decomposition Modulus Count (Q_size)
    //     int block_z = blockIdx.z; // Cipher Size (2)

    //     // Max P size is 15.
    //     Data64 last_pk[15];
    //     for (int i = 0; i < P_size; i++)
    //     {
    //         Data64 last_pk_ = pk[idx + ((Q_size + i) << n_power) +
    //                              ((Q_prime_size << n_power) * block_z)];
    //         Data64 last_e_ = e[idx + ((Q_size + i) << n_power) +
    //                            ((Q_prime_size << n_power) * block_z)];
    //         last_pk[i] =
    //             OPERATOR_GPU_64::add(last_pk_, last_e_, modulus[Q_size + i]);
    //     }

    //     Data64 input_ = pk[idx + (block_y << n_power) +
    //                        ((Q_prime_size << n_power) * block_z)];
    //     Data64 e_ = e[idx + (block_y << n_power) +
    //                   ((Q_prime_size << n_power) * block_z)];
    //     input_ = OPERATOR_GPU_64::add(input_, e_, modulus[block_y]);

    //     Data64 zero_ = 0;
    //     int location_ = 0;
    //     for (int i = 0; i < P_size; i++)
    //     {
    //         Data64 last_pk_add_half_ = last_pk[(P_size - 1 - i)];
    //         last_pk_add_half_ = OPERATOR_GPU_64::add(
    //             last_pk_add_half_, half[i], modulus[(Q_prime_size - 1 - i)]);
    //         for (int j = 0; j < (P_size - 1 - i); j++)
    //         {
    //             Data64 temp1 = OPERATOR_GPU_64::add(last_pk_add_half_, zero_,
    //                                                 modulus[Q_size + j]);
    //             temp1 = OPERATOR_GPU_64::sub(temp1,
    //                                          half_mod[location_ + Q_size + j],
    //                                          modulus[Q_size + j]);

    //             temp1 = OPERATOR_GPU_64::sub(last_pk[j], temp1,
    //                                          modulus[Q_size + j]);

    //             last_pk[j] = OPERATOR_GPU_64::mult(
    //                 temp1, last_q_modinv[location_ + Q_size + j],
    //                 modulus[Q_size + j]);
    //         }

    //         Data64 temp1 = OPERATOR_GPU_64::add(last_pk_add_half_, zero_,
    //                                             modulus[block_y]);
    //         temp1 = OPERATOR_GPU_64::sub(temp1, half_mod[location_ + block_y],
    //                                      modulus[block_y]);

    //         temp1 = OPERATOR_GPU_64::sub(input_, temp1, modulus[block_y]);

    //         input_ = OPERATOR_GPU_64::mult(
    //             temp1, last_q_modinv[location_ + block_y], modulus[block_y]);

    //         location_ = location_ + (Q_prime_size - 1 - i);
    //     }

    //     if (block_z == 0)
    //     {
    //         Data64 message = plain[idx];
    //         Data64 fix = message * Q_mod_t;
    //         fix = fix + upper_threshold;
    //         fix = int(fix / plain_mod.value);

    //         Data64 ct_0 = OPERATOR_GPU_64::mult(
    //             message, coeffdiv_plain[block_y], modulus[block_y]);
    //         ct_0 = OPERATOR_GPU_64::add(ct_0, fix, modulus[block_y]);

    //         input_ = OPERATOR_GPU_64::add(input_, ct_0, modulus[block_y]);

    //         ct[idx + (block_y << n_power) + (((Q_size) << n_power) * block_z)] =
    //             input_;
    //     }
    //     else
    //     {
    //         ct[idx + (block_y << n_power) + (((Q_size) << n_power) * block_z)] =
    //             input_;
    //     }
    // }

    // __global__ void enc_div_lastq_ckks_kernel(Data64* pk, Data64* e, Data64* ct,
    //                                           Modulus64* modulus, Data64* half,
    //                                           Data64* half_mod,
    //                                           Data64* last_q_modinv,
    //                                           int n_power, int Q_prime_size,
    //                                           int Q_size, int P_size)
    // {
    //     int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
    //     int block_y = blockIdx.y; // Decomposition Modulus Count (Q_size)
    //     int block_z = blockIdx.z; // Cipher Size (2)

    //     // Max P size is 15.
    //     Data64 last_pk[15];
    //     for (int i = 0; i < P_size; i++)
    //     {
    //         Data64 last_pk_ = pk[idx + ((Q_size + i) << n_power) +
    //                              ((Q_prime_size << n_power) * block_z)];
    //         Data64 last_e_ = e[idx + ((Q_size + i) << n_power) +
    //                            ((Q_prime_size << n_power) * block_z)];
    //         last_pk[i] =
    //             OPERATOR_GPU_64::add(last_pk_, last_e_, modulus[Q_size + i]);
    //     }

    //     Data64 input_ = pk[idx + (block_y << n_power) +
    //                        ((Q_prime_size << n_power) * block_z)];
    //     Data64 e_ = e[idx + (block_y << n_power) +
    //                   ((Q_prime_size << n_power) * block_z)];
    //     input_ = OPERATOR_GPU_64::add(input_, e_, modulus[block_y]);

    //     int location_ = 0;
    //     for (int i = 0; i < P_size; i++)
    //     {
    //         Data64 last_pk_add_half_ = last_pk[(P_size - 1 - i)];
    //         last_pk_add_half_ = OPERATOR_GPU_64::add(
    //             last_pk_add_half_, half[i], modulus[(Q_prime_size - 1 - i)]);

    //         for (int j = 0; j < (P_size - 1 - i); j++)
    //         {
    //             Data64 temp1 = OPERATOR_GPU_64::reduce(last_pk_add_half_,
    //                                                    modulus[Q_size + j]);

    //             temp1 = OPERATOR_GPU_64::sub(temp1,
    //                                          half_mod[location_ + Q_size + j],
    //                                          modulus[Q_size + j]);

    //             temp1 = OPERATOR_GPU_64::sub(last_pk[j], temp1,
    //                                          modulus[Q_size + j]);

    //             last_pk[j] = OPERATOR_GPU_64::mult(
    //                 temp1, last_q_modinv[location_ + Q_size + j],
    //                 modulus[Q_size + j]);
    //         }

    //         // Data64 temp1 = OPERATOR_GPU_64::reduce(last_pk_add_half_,
    //         // modulus[block_y]);
    //         Data64 temp1 = OPERATOR_GPU_64::reduce_forced(last_pk_add_half_,
    //                                                       modulus[block_y]);

    //         temp1 = OPERATOR_GPU_64::sub(temp1, half_mod[location_ + block_y],
    //                                      modulus[block_y]);

    //         temp1 = OPERATOR_GPU_64::sub(input_, temp1, modulus[block_y]);

    //         input_ = OPERATOR_GPU_64::mult(
    //             temp1, last_q_modinv[location_ + block_y], modulus[block_y]);

    //         location_ = location_ + (Q_prime_size - 1 - i);
    //     }

    //     ct[idx + (block_y << n_power) + (((Q_size) << n_power) * block_z)] =
    //         input_;
    // }

    // __global__ void cipher_message_add_kernel(Data64* ciphertext,
    //                                           Data64* plaintext,
    //                                           Modulus64* modulus, int n_power)
    // {
    //     int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
    //     int block_y = blockIdx.y; // Decomposition Modulus Count (Q_size)

    //     Data64 ct_0 = ciphertext[idx + (block_y << n_power)];
    //     Data64 plaintext_ = plaintext[idx + (block_y << n_power)];

    //     ct_0 = OPERATOR_GPU_64::add(ct_0, plaintext_, modulus[block_y]);

    //     ciphertext[idx + (block_y << n_power)] = ct_0;
    // }

} // namespace heongpu
