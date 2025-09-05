#include "random_cpu.hpp"

namespace heoncpu
{
    // Not cryptographically secure, will be fixed later.
//     __global__ void modular_uniform_random_number_generation_kernel(
//         Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
//         int seed, int offset)
//     {
//         int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
//         int block_y = blockIdx.y;

//         int subsequence = idx + (block_y << n_power);
//         curandState_t state;
//         curand_init(seed, subsequence, offset, &state);

//         int out_offset = (block_y * rns_mod_count) << n_power;
// #pragma unroll
//         for (int i = 0; i < rns_mod_count; i++)
//         {
//             int in_offset = i << n_power;

//             uint32_t rn_lo = curand(&state);
//             uint32_t rn_hi = curand(&state);

//             uint64_t combined = (static_cast<uint64_t>(rn_hi) << 32) |
//                                 static_cast<uint64_t>(rn_lo);
//             Data64 rn_ULL = static_cast<Data64>(combined);
//             rn_ULL = OPERATOR_GPU_64::reduce_forced(rn_ULL, modulus[i]);

//             output[idx + in_offset + out_offset] = rn_ULL;
//         }
//     }
    void modular_uniform_random_number_generation_cpu(
        Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
        int seed, int offset)
    {
        int n = 1 << n_power; // 计算环的大小
        int grid_dim_x = n >> 8; // 根据调用参数计算网格x维度
        
        std::uniform_int_distribution<uint64_t> distribution;
        
        // 遍历所有块和线程
        for (int block_x = 0; block_x < grid_dim_x; block_x++) {
            for (int thread_idx = 0; thread_idx < 256; thread_idx++) {
                int idx = block_x * 256 + thread_idx;
                if (idx >= n) continue; // 确保不越界
                
                int subsequence = idx; // 因为 gridDim.y = 1, 所以 block_y = 0
                std::mt19937_64 engine(seed + subsequence + offset);
                
                int block_y = 0; // 因为 gridDim.y = 1
                int out_offset = (block_y * rns_mod_count) << n_power;
                
                // 为每个RNS模数生成随机数
                for (int i = 0; i < rns_mod_count; i++) {
                    int in_offset = i << n_power;
                    
                    // 生成64位随机数
                    uint32_t rn_lo = static_cast<uint32_t>(distribution(engine));
                    uint32_t rn_hi = static_cast<uint32_t>(distribution(engine));
                    
                    uint64_t combined = (static_cast<uint64_t>(rn_hi) << 32) |
                                        static_cast<uint64_t>(rn_lo);
                    Data64 rn_ULL = static_cast<Data64>(combined);
                    rn_ULL = OPERATOR64::reduce_forced(rn_ULL, modulus[i]);
                    
                    output[idx + in_offset + out_offset] = rn_ULL;
                }
            }
        }
    }
    
//     // Not cryptographically secure, will be fixed later.
//     __global__ void modular_uniform_random_number_generation_kernel(
//         Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
//         int seed, int offset, int* mod_index)
//     {
//         int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
//         int block_y = blockIdx.y;

//         int subsequence = idx + (block_y << n_power);
//         curandState_t state;
//         curand_init(seed, subsequence, offset, &state);

//         int out_offset = (block_y * rns_mod_count) << n_power;
// #pragma unroll
//         for (int i = 0; i < rns_mod_count; i++)
//         {
//             int in_offset = i << n_power;
//             int index_mod = mod_index[i];

//             uint32_t rn_lo = curand(&state);
//             uint32_t rn_hi = curand(&state);

//             uint64_t combined = (static_cast<uint64_t>(rn_hi) << 32) |
//                                 static_cast<uint64_t>(rn_lo);
//             Data64 rn_ULL = static_cast<Data64>(combined);
//             rn_ULL = OPERATOR_GPU_64::reduce_forced(rn_ULL, modulus[index_mod]);

//             output[idx + in_offset + out_offset] = rn_ULL;
//         }
//     }

    // Not cryptographically secure, will be fixed later.
//     __global__ void modular_gaussian_random_number_generation_kernel(
//         Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
//         int seed, int offset)
//     {
//         int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
//         int block_y = blockIdx.y;

//         int subsequence = idx + (block_y << n_power);
//         curandState_t state;
//         curand_init(seed, subsequence, offset, &state);

//         float noise = curand_normal(&state);
//         noise = noise * error_std_dev; // SIGMA

//         uint64_t flag = static_cast<uint64_t>(-static_cast<int64_t>(noise < 0));

//         int out_offset = (block_y * rns_mod_count) << n_power;
// #pragma unroll
//         for (int i = 0; i < rns_mod_count; i++)
//         {
//             Data64 rn_ULL =
//                 static_cast<Data64>(noise) + (flag & modulus[i].value);
//             int in_offset = i << n_power;
//             output[idx + in_offset + out_offset] = rn_ULL;
//         }
//     }

    void modular_gaussian_random_number_generation_cpu(
        Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
        int seed, int offset)
    {
        int n = 1 << n_power; // 计算环的大小
        int grid_dim_x = n >> 8; // 根据调用参数计算网格x维度
        
        // 遍历所有块和线程
        for (int block_x = 0; block_x < grid_dim_x; block_x++) {
            for (int thread_idx = 0; thread_idx < 256; thread_idx++) {
                int idx = block_x * 256 + thread_idx;
                if (idx >= n) continue; // 确保不越界
                
                int subsequence = idx; // 因为 gridDim.y = 1, 所以 block_y = 0
                std::mt19937_64 engine(seed + subsequence + offset);
                std::normal_distribution<float> distribution(0.0f, 1.0f); // 均值为0，标准差为1
                
                int block_y = 0; // 因为 gridDim.y = 1
                
                // 生成高斯随机数并缩放
                float noise = distribution(engine);
                noise = noise * error_std_dev; // 乘以标准差
                
                // 处理负数标志
                uint64_t flag = static_cast<uint64_t>(-static_cast<int64_t>(noise < 0));
                
                int out_offset = (block_y * rns_mod_count) << n_power;
                
                // 为每个RNS模数生成随机数
                for (int i = 0; i < rns_mod_count; i++) {
                    int in_offset = i << n_power;
                    
                    // 处理负数：如果噪声为负，加上模数值
                    Data64 rn_ULL = static_cast<Data64>(noise) + (flag & modulus[i].value);
                    
                    output[idx + in_offset + out_offset] = rn_ULL;
                }
            }
        }
    }
//     // Not cryptographically secure, will be fixed later.
//     __global__ void modular_gaussian_random_number_generation_kernel(
//         Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
//         int seed, int offset, int* mod_index)
//     {
//         int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
//         int block_y = blockIdx.y;

//         int subsequence = idx + (block_y << n_power);
//         curandState_t state;
//         curand_init(seed, subsequence, offset, &state);

//         float noise = curand_normal(&state);
//         noise = noise * error_std_dev; // SIGMA

//         uint64_t flag = static_cast<uint64_t>(-static_cast<int64_t>(noise < 0));

//         int out_offset = (block_y * rns_mod_count) << n_power;
// #pragma unroll
//         for (int i = 0; i < rns_mod_count; i++)
//         {
//             int index_mod = mod_index[i];
//             Data64 rn_ULL =
//                 static_cast<Data64>(noise) + (flag & modulus[index_mod].value);
//             int in_offset = i << n_power;
//             output[idx + in_offset + out_offset] = rn_ULL;
//         }
//     }

//     // Not cryptographically secure, will be fixed later.
//     __global__ void modular_ternary_random_number_generation_kernel(
//         Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
//         int seed, int offset)
//     {
//         int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes

//         curandState_t state;
//         curand_init(seed, idx, offset, &state);

//         // TODO: make it efficient
//         Data64 random_number = curand(&state) & 3; // 0,1,2,3
//         if (random_number == 3)
//         {
//             random_number -= 3; // 0,1,2
//         }

//         uint64_t flag =
//             static_cast<uint64_t>(-static_cast<int64_t>(random_number == 0));

// #pragma unroll
//         for (int i = 0; i < rns_mod_count; i++)
//         {
//             int location = i << n_power;
//             Data64 result = random_number;
//             result = result + (flag & modulus[i].value) - 1;
//             output[idx + location] = result;
//         }
//     }

//     // Not cryptographically secure, will be fixed later.
//     __global__ void modular_ternary_random_number_generation_kernel(
//         Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
//         int seed, int offset, int* mod_index)
//     {
//         int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes

//         curandState_t state;
//         curand_init(seed, idx, offset, &state);

//         // TODO: make it efficient
//         Data64 random_number = curand(&state) & 3; // 0,1,2,3
//         if (random_number == 3)
//         {
//             random_number -= 3; // 0,1,2
//         }

//         uint64_t flag =
//             static_cast<uint64_t>(-static_cast<int64_t>(random_number == 0));

// #pragma unroll
//         for (int i = 0; i < rns_mod_count; i++)
//         {
//             int index_mod = mod_index[i];
//             int location = i << n_power;
//             Data64 result = random_number;
//             result = result + (flag & modulus[index_mod].value) - 1;
//             output[idx + location] = result;
//         }
//     }
} // namespace heongpu
