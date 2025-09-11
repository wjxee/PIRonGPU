#include "decryption_cpu.hpp"

namespace heoncpu{

    void sk_multiplication_cpu(
        Data64* ct1, Data64* sk, Data64* output,
        Modulus64* modulus, int n_power,
        int decomp_mod_count,
        int grid_x, int grid_y, int grid_z, int block_size) {
        
        // 计算n = 2^(n_power)
        int n = 1 << n_power;
        
        // 计算总线程数
        int total_threads = grid_x * grid_y * grid_z * block_size;
        
        // 确保不会越界访问
        if (total_threads > n * decomp_mod_count) {
            total_threads = n * decomp_mod_count;
        }
        
        // 模拟CUDA的线程网格结构
        for (int block_z = 0; block_z < grid_z; block_z++) {
            for (int block_y = 0; block_y < grid_y; block_y++) {
                for (int block_x = 0; block_x < grid_x; block_x++) {
                    for (int thread_id = 0; thread_id < block_size; thread_id++) {
                        // 计算全局线程索引（模拟CUDA的线程索引计算）
                        int idx = ((block_x + block_y * grid_x + block_z * grid_x * grid_y) * block_size) + thread_id;
                        
                        // 确保不超出数组范围
                        if (idx >= n * decomp_mod_count) continue;
                        
                        // 计算块索引（模拟blockIdx.y）
                        int block_y_idx = block_y;
                        
                        // 计算索引（模拟核函数中的索引计算）
                        int index = idx + (block_y_idx << n_power);
                        
                        // 再次检查索引是否越界
                        if (index >= n * decomp_mod_count) continue;
                        
                        // 获取数据
                        Data64 ct_1 = ct1[index];
                        Data64 sk_ = sk[index];
                        
                        // 确保模数索引在有效范围内
                        if (block_y_idx >= decomp_mod_count) {
                            throw std::out_of_range("Modulus index out of range: " + std::to_string(block_y_idx));
                        }
                        
                        // 执行乘法运算
                        output[index] = OPERATOR_GPU_64::mult(ct_1, sk_, modulus[block_y_idx]);
                    }
                }
            }
        }
    }

    void decryption_kernel_cpu(
        Data64* ct0, Data64* ct1, Data64* plain,
        Modulus64* modulus, Modulus64 plain_mod,
        Modulus64 gamma, Data64* Qi_t,
        Data64* Qi_gamma, Data64* Qi_inverse,
        Data64 mulq_inv_t, Data64 mulq_inv_gamma,
        Data64 inv_gamma, int n_power,
        int decomp_mod_count,
        int grid_x, int grid_y, int grid_z, int block_size) {
        
        // 计算n = 2^(n_power)
        int n = 1 << n_power;
        
        // 计算总线程数
        int total_threads = grid_x * grid_y * grid_z * block_size;
        
        // 确保不会越界访问
        if (total_threads > n) {
            total_threads = n;
        }
        
        // 模拟CUDA的线程网格结构
        for (int block_z = 0; block_z < grid_z; block_z++) {
            for (int block_y = 0; block_y < grid_y; block_y++) {
                for (int block_x = 0; block_x < grid_x; block_x++) {
                    for (int thread_id = 0; thread_id < block_size; thread_id++) {
                        // 计算全局线程索引（模拟CUDA的线程索引计算）
                        int idx = ((block_x + block_y * grid_x + block_z * grid_x * grid_y) * block_size) + thread_id;
                        
                        // 确保不超出数组范围
                        if (idx >= n) continue;
                        
                        Data64 sum_t = 0;
                        Data64 sum_gamma = 0;
                        
                        // 处理每个模数
                        for (int i = 0; i < decomp_mod_count; i++) {
                            int location = idx + (i << n_power);
                            
                            // 检查位置是否越界
                            if (location >= n * decomp_mod_count) {
                                throw std::out_of_range("Location out of range: " + std::to_string(location));
                            }
                            
                            // 确保模数索引在有效范围内
                            if (i >= decomp_mod_count) {
                                throw std::out_of_range("Modulus index out of range: " + std::to_string(i));
                            }
                            
                            // 执行解密运算
                            Data64 mt = OPERATOR_GPU_64::add(ct0[location], ct1[location], modulus[i]);
                            
                            Data64 gamma_ = OPERATOR_GPU_64::reduce_forced(gamma.value, modulus[i]);
                            
                            mt = OPERATOR_GPU_64::mult(mt, plain_mod.value, modulus[i]);
                            
                            mt = OPERATOR_GPU_64::mult(mt, gamma_, modulus[i]);
                            
                            mt = OPERATOR_GPU_64::mult(mt, Qi_inverse[i], modulus[i]);
                            
                            Data64 mt_in_t = OPERATOR_GPU_64::reduce_forced(mt, plain_mod);
                            Data64 mt_in_gamma = OPERATOR_GPU_64::reduce_forced(mt, gamma);
                            
                            mt_in_t = OPERATOR_GPU_64::mult(mt_in_t, Qi_t[i], plain_mod);
                            mt_in_gamma = OPERATOR_GPU_64::mult(mt_in_gamma, Qi_gamma[i], gamma);
                            
                            sum_t = OPERATOR_GPU_64::add(sum_t, mt_in_t, plain_mod);
                            sum_gamma = OPERATOR_GPU_64::add(sum_gamma, mt_in_gamma, gamma);
                        }
                        
                        sum_t = OPERATOR_GPU_64::mult(sum_t, mulq_inv_t, plain_mod);
                        sum_gamma = OPERATOR_GPU_64::mult(sum_gamma, mulq_inv_gamma, gamma);
                        
                        Data64 gamma_2 = gamma.value >> 1;
                        
                        if (sum_gamma > gamma_2) {
                            Data64 gamma_ = OPERATOR_GPU_64::reduce_forced(gamma.value, plain_mod);
                            Data64 sum_gamma_ = OPERATOR_GPU_64::reduce_forced(sum_gamma, plain_mod);
                            
                            Data64 result = OPERATOR_GPU_64::sub(gamma_, sum_gamma_, plain_mod);
                            result = OPERATOR_GPU_64::add(sum_t, result, plain_mod);
                            result = OPERATOR_GPU_64::mult(result, inv_gamma, plain_mod);
                            
                            plain[idx] = result;
                        } else {
                            Data64 sum_t_ = OPERATOR_GPU_64::reduce_forced(sum_t, plain_mod);
                            Data64 sum_gamma_ = OPERATOR_GPU_64::reduce_forced(sum_gamma, plain_mod);
                            
                            Data64 result = OPERATOR_GPU_64::sub(sum_t_, sum_gamma_, plain_mod);
                            result = OPERATOR_GPU_64::mult(result, inv_gamma, plain_mod);
                            
                            plain[idx] = result;
                        }
                    }
                }
            }
        }
    }
}