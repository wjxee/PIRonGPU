// #include "random_cpu.hpp"

// namespace heoncpu
// {
//     // Not cryptographically secure, will be fixed later.
// //     __global__ void modular_uniform_random_number_generation_kernel(
// //         Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
// //         int seed, int offset)
// //     {
// //         int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
// //         int block_y = blockIdx.y;

// //         int subsequence = idx + (block_y << n_power);
// //         curandState_t state;
// //         curand_init(seed, subsequence, offset, &state);

// //         int out_offset = (block_y * rns_mod_count) << n_power;
// // #pragma unroll
// //         for (int i = 0; i < rns_mod_count; i++)
// //         {
// //             int in_offset = i << n_power;

// //             uint32_t rn_lo = curand(&state);
// //             uint32_t rn_hi = curand(&state);

// //             uint64_t combined = (static_cast<uint64_t>(rn_hi) << 32) |
// //                                 static_cast<uint64_t>(rn_lo);
// //             Data64 rn_ULL = static_cast<Data64>(combined);
// //             rn_ULL = OPERATOR_GPU_64::reduce_forced(rn_ULL, modulus[i]);

// //             output[idx + in_offset + out_offset] = rn_ULL;
// //         }
// //     }
    
//     void modular_uniform_random_number_generation_cpu(
//         Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
//         int grid_x, int grid_y, int grid_z, int block_size,
//         int seed, int offset) {
        
//         int n = 1 << n_power; // 计算n
        
//         // 遍历所有网格和块
//         for (int block_z = 0; block_z < grid_z; block_z++) {
//             for (int block_y = 0; block_y < grid_y; block_y++) {
//                 for (int block_x = 0; block_x < grid_x; block_x++) {
//                     // 处理当前块中的所有线程
//                     for (int thread_idx = 0; thread_idx < block_size; thread_idx++) {
//                         // 计算全局索引
//                         int idx = block_x * block_size + thread_idx;
                        
//                         if (idx >= n) {
//                             continue; // 确保不超出范围
//                         }
                        
//                         // 计算子序列号（模拟CUDA中的subsequence）
//                         int subsequence = idx + (block_y << n_power) + (block_z << (n_power + 8));
                        
//                         // 为每个子序列创建随机数生成器
//                         std::mt19937_64 gen(seed + subsequence + offset);
                        
//                         // 计算输出偏移量
//                         int out_offset = ((block_y * grid_z + block_z) * rns_mod_count) << n_power;
                        
//                         // 为每个RNS模数生成随机数
//                         for (int i = 0; i < rns_mod_count; i++) {
//                             int in_offset = i << n_power;
                            
//                             // 生成64位随机数
//                             uint64_t rn_ULL = gen();
                            
//                             // 强制约减
//                             rn_ULL = OPERATOR64::reduce_forced(rn_ULL, modulus[i]);
                            
//                             // 存储结果
//                             output[idx + in_offset + out_offset] = rn_ULL;
//                         }
//                     }
//                 }
//             }
//         }
//     }

// //     // Not cryptographically secure, will be fixed later.
// //     __global__ void modular_uniform_random_number_generation_kernel(
// //         Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
// //         int seed, int offset, int* mod_index)
// //     {
// //         int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
// //         int block_y = blockIdx.y;

// //         int subsequence = idx + (block_y << n_power);
// //         curandState_t state;
// //         curand_init(seed, subsequence, offset, &state);

// //         int out_offset = (block_y * rns_mod_count) << n_power;
// // #pragma unroll
// //         for (int i = 0; i < rns_mod_count; i++)
// //         {
// //             int in_offset = i << n_power;
// //             int index_mod = mod_index[i];

// //             uint32_t rn_lo = curand(&state);
// //             uint32_t rn_hi = curand(&state);

// //             uint64_t combined = (static_cast<uint64_t>(rn_hi) << 32) |
// //                                 static_cast<uint64_t>(rn_lo);
// //             Data64 rn_ULL = static_cast<Data64>(combined);
// //             rn_ULL = OPERATOR_GPU_64::reduce_forced(rn_ULL, modulus[index_mod]);

// //             output[idx + in_offset + out_offset] = rn_ULL;
// //         }
// //     }

//     // Not cryptographically secure, will be fixed later.
// //     __global__ void modular_gaussian_random_number_generation_kernel(
// //         Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
// //         int seed, int offset)
// //     {
// //         int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
// //         int block_y = blockIdx.y;

// //         int subsequence = idx + (block_y << n_power);
// //         curandState_t state;
// //         curand_init(seed, subsequence, offset, &state);

// //         float noise = curand_normal(&state);
// //         noise = noise * error_std_dev; // SIGMA

// //         uint64_t flag = static_cast<uint64_t>(-static_cast<int64_t>(noise < 0));

// //         int out_offset = (block_y * rns_mod_count) << n_power;
// // #pragma unroll
// //         for (int i = 0; i < rns_mod_count; i++)
// //         {
// //             Data64 rn_ULL =
// //                 static_cast<Data64>(noise) + (flag & modulus[i].value);
// //             int in_offset = i << n_power;
// //             output[idx + in_offset + out_offset] = rn_ULL;
// //         }
// //     }

//     void modular_gaussian_random_number_generation_cpu(
//         Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
//         int grid_x, int grid_y, int grid_z, int block_size,
//         int seed, int offset) {
        
//         int n = 1 << n_power; // 计算n
        
//         // 使用固定种子创建主随机数生成器
//         std::mt19937_64 main_gen(seed);
        
//         // 遍历所有网格和块
//         for (int block_z = 0; block_z < grid_z; block_z++) {
//             for (int block_y = 0; block_y < grid_y; block_y++) {
//                 for (int block_x = 0; block_x < grid_x; block_x++) {
//                     // 处理当前块中的所有线程
//                     for (int thread_idx = 0; thread_idx < block_size; thread_idx++) {
//                         // 计算全局索引
//                         int idx = block_x * block_size + thread_idx;
                        
//                         if (idx >= n) {
//                             continue; // 确保不超出范围
//                         }
                        
//                         // 计算子序列号（模拟CUDA中的subsequence）
//                         int subsequence = idx + (block_y << n_power) + (block_z << (n_power + 8));
                        
//                         // 为每个子序列创建随机数生成器
//                         std::mt19937_64 gen(seed + subsequence + offset);
//                         std::normal_distribution<float> normal_dist(0.0f, 1.0f); // 均值为0，标准差为1
                        
//                         // 生成高斯随机数并缩放
//                         float noise = normal_dist(gen);
//                         noise = noise * error_std_dev; // 乘以SIGMA
                        
//                         // 处理负数情况
//                         uint64_t flag = static_cast<uint64_t>(-static_cast<int64_t>(noise < 0));
                        
//                         // 计算输出偏移量
//                         int out_offset = ((block_y * grid_z + block_z) * rns_mod_count) << n_power;
                        
//                         // 为每个RNS模数生成随机数
//                         for (int i = 0; i < rns_mod_count; i++) {
//                             int in_offset = i << n_power;
                            
//                             // 计算随机数并处理模数
//                             Data64 rn_ULL = static_cast<Data64>(noise) + (flag & modulus[i].value);
                            
//                             // 存储结果
//                             output[idx + in_offset + out_offset] = rn_ULL;
//                         }
//                     }
//                 }
//             }
//         }
//     }
// //     // Not cryptographically secure, will be fixed later.
// //     __global__ void modular_gaussian_random_number_generation_kernel(
// //         Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
// //         int seed, int offset, int* mod_index)
// //     {
// //         int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
// //         int block_y = blockIdx.y;

// //         int subsequence = idx + (block_y << n_power);
// //         curandState_t state;
// //         curand_init(seed, subsequence, offset, &state);

// //         float noise = curand_normal(&state);
// //         noise = noise * error_std_dev; // SIGMA

// //         uint64_t flag = static_cast<uint64_t>(-static_cast<int64_t>(noise < 0));

// //         int out_offset = (block_y * rns_mod_count) << n_power;
// // #pragma unroll
// //         for (int i = 0; i < rns_mod_count; i++)
// //         {
// //             int index_mod = mod_index[i];
// //             Data64 rn_ULL =
// //                 static_cast<Data64>(noise) + (flag & modulus[index_mod].value);
// //             int in_offset = i << n_power;
// //             output[idx + in_offset + out_offset] = rn_ULL;
// //         }
// //     }

//     // Not cryptographically secure, will be fixed later.
// //     __global__ void modular_ternary_random_number_generation_kernel(
// //         Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
// //         int seed, int offset)
// //     {
// //         int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes

// //         curandState_t state;
// //         curand_init(seed, idx, offset, &state);

// //         // TODO: make it efficient
// //         Data64 random_number = curand(&state) & 3; // 0,1,2,3
// //         if (random_number == 3)
// //         {
// //             random_number -= 3; // 0,1,2
// //         }

// //         uint64_t flag =
// //             static_cast<uint64_t>(-static_cast<int64_t>(random_number == 0));

// // #pragma unroll
// //         for (int i = 0; i < rns_mod_count; i++)
// //         {
// //             int location = i << n_power;
// //             Data64 result = random_number;
// //             result = result + (flag & modulus[i].value) - 1;
// //             output[idx + location] = result;
// //         }
// //     }
//     void modular_ternary_random_number_generation_cpu(
//         Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
//         int seed, int offset,
//         int grid_x, int grid_y, int grid_z, int block_size) {
        
//         int n = 1 << n_power; // 计算n
        
//         // 使用固定种子创建主随机数生成器
//         // std::mt19937_64 main_gen(seed);
        
//         // 遍历所有网格和块
//         for (int block_z = 0; block_z < grid_z; block_z++) {
//             for (int block_y = 0; block_y < grid_y; block_y++) {
//                 for (int block_x = 0; block_x < grid_x; block_x++) {
//                     // 处理当前块中的所有线程
//                     for (int thread_idx = 0; thread_idx < block_size; thread_idx++) {
//                         // 计算全局索引
//                         int idx = block_x * block_size + thread_idx;
                        
//                         if (idx >= n) {
//                             continue; // 确保不超出范围
//                         }
                        
//                         // 为每个线程创建独立的随机数生成器
//                         // 使用seed + idx + offset作为种子，模拟CUDA的curand_init
//                         std::mt19937_64 gen(seed + idx + offset);
                        
//                         // 生成0-3的随机数
//                         Data64 random_number = gen() & 3; // 取最低2位，得到0,1,2,3
                        
//                         // 将3转换为0
//                         if (random_number == 3) {
//                             random_number = 0; // 0,1,2
//                         }
                        
//                         // 处理标志位
//                         uint64_t flag = static_cast<uint64_t>(-static_cast<int64_t>(random_number == 0));
                        
//                         // 为每个RNS模数生成结果
//                         for (int i = 0; i < rns_mod_count; i++) {
//                             int location = i << n_power;
                            
//                             // 计算结果
//                             Data64 result = random_number;
//                             result = result + (flag & modulus[i].value) - 1;
                            
//                             // 存储结果
//                             output[idx + location] = result;
//                         }
//                     }
//                 }
//             }
//         }
//     }

// //     // Not cryptographically secure, will be fixed later.
// //     __global__ void modular_ternary_random_number_generation_kernel(
// //         Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
// //         int seed, int offset, int* mod_index)
// //     {
// //         int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes

// //         curandState_t state;
// //         curand_init(seed, idx, offset, &state);

// //         // TODO: make it efficient
// //         Data64 random_number = curand(&state) & 3; // 0,1,2,3
// //         if (random_number == 3)
// //         {
// //             random_number -= 3; // 0,1,2
// //         }

// //         uint64_t flag =
// //             static_cast<uint64_t>(-static_cast<int64_t>(random_number == 0));

// // #pragma unroll
// //         for (int i = 0; i < rns_mod_count; i++)
// //         {
// //             int index_mod = mod_index[i];
// //             int location = i << n_power;
// //             Data64 result = random_number;
// //             result = result + (flag & modulus[index_mod].value) - 1;
// //             output[idx + location] = result;
// //         }
// //     }
// } // namespace heongpu
