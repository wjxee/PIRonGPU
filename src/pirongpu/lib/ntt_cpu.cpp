#include "ntt_cpu.hpp"
#include <iostream>
#include <cassert>

namespace heoncpu
{
    template <typename T>
    std::vector<T>
    schoolbook_poly_multiplication(std::vector<T> a, std::vector<T> b,
                                   Modulus<T> modulus,
                                   ReductionPolynomial reduction_poly)
    {
        int length = a.size();
        std::vector<T> mult_vector(length * 2, 0);

        for (int i = 0; i < length; i++)
        {
            for (int j = 0; j < length; j++)
            {
                T mult_result = OPERATOR<T>::mult(a[i], b[j], modulus);
                mult_vector[i + j] =
                    OPERATOR<T>::add(mult_vector[i + j], mult_result, modulus);
            }
        }

        std::vector<T> result(length, 0);
        if (reduction_poly == ReductionPolynomial::X_N_minus)
        {
            for (int i = 0; i < length; i++)
            {
                result[i] = OPERATOR<T>::add(mult_vector[i],
                                             mult_vector[i + length], modulus);
            }
        }
        else if (reduction_poly == ReductionPolynomial::X_N_plus)
        {
            for (int i = 0; i < length; i++)
            {
                result[i] = OPERATOR<T>::sub(mult_vector[i],
                                             mult_vector[i + length], modulus);
            }
        }
        else
        {
            throw std::runtime_error("Poly reduction type is not supported!");
        }

        return result;
    }

    template std::vector<Data32> schoolbook_poly_multiplication<Data32>(
        std::vector<Data32> a, std::vector<Data32> b, Modulus<Data32> modulus,
        ReductionPolynomial reduction_poly);

    template std::vector<Data64> schoolbook_poly_multiplication<Data64>(
        std::vector<Data64> a, std::vector<Data64> b, Modulus<Data64> modulus,
        ReductionPolynomial reduction_poly);

    template <typename T> NTTCPU<T>::NTTCPU(NTTParameters<T> parameters_)
    {
        parameters = parameters_;
    }

    template <typename T>
    std::vector<T> NTTCPU<T>::mult(std::vector<T>& input1,
                                   std::vector<T>& input2)
    {
        std::vector<T> output;
        for (int i = 0; i < parameters.n; i++)
        {
            output.push_back(
                OPERATOR<T>::mult(input1[i], input2[i], parameters.modulus));
        }

        return output;
    }

    template <typename T> std::vector<T> NTTCPU<T>::ntt(std::vector<T>& input)
    {
        // Merged NTT with pre-processing (optimized) (iterative)
        // This is not NTT, this is pre-processing + NTT
        // (see: https://eprint.iacr.org/2016/504.pdf)

        std::vector<T> output = input;

        int t = parameters.n;
        int m = 1;

        while (m < parameters.n)
        {
            t = t >> 1;

            for (int i = 0; i < m; i++)
            {
                int j1 = 2 * i * t;
                int j2 = j1 + t - 1;

                int index;
                if (parameters.poly_reduction == ReductionPolynomial::X_N_minus)
                {
                    index = bitreverse(i, parameters.logn - 1);
                }
                else
                { // poly_reduce_type = ReductionPolynomial::X_N_plus
                    index = bitreverse(m + i, parameters.logn);
                }

                T S = parameters.forward_root_of_unity_table[index];

                for (int j = j1; j < (j2 + 1); j++)
                {
                    T U = output[j];
                    T V =
                        OPERATOR<T>::mult(output[j + t], S, parameters.modulus);

                    output[j] = OPERATOR<T>::add(U, V, parameters.modulus);
                    output[j + t] = OPERATOR<T>::sub(U, V, parameters.modulus);
                }
            }

            m = m << 1;
        }

        return output;
    }

    template <typename T> std::vector<T> NTTCPU<T>::intt(std::vector<T>& input)
    {
        // Merged INTT with post-processing (optimized) (iterative)
        // This is not NTT, this is pre-processing + NTT
        // (see: https://eprint.iacr.org/2016/504.pdf)

        std::vector<T> output = input;

        int t = 1;
        int m = parameters.n;
        while (m > 1)
        {
            int j1 = 0;
            int h = m >> 1;
            for (int i = 0; i < h; i++)
            {
                int j2 = j1 + t - 1;
                int index;
                if (parameters.poly_reduction == ReductionPolynomial::X_N_minus)
                {
                    index = bitreverse(i, parameters.logn - 1);
                }
                else
                { // poly_reduce_type = ReductionPolynomial::X_N_plus
                    index = bitreverse(h + i, parameters.logn);
                }

                T S = parameters.inverse_root_of_unity_table[index];

                for (int j = j1; j < (j2 + 1); j++)
                {
                    T U = output[j];
                    T V = output[j + t];

                    output[j] = OPERATOR<T>::add(U, V, parameters.modulus);
                    output[j + t] = OPERATOR<T>::sub(U, V, parameters.modulus);
                    output[j + t] =
                        OPERATOR<T>::mult(output[j + t], S, parameters.modulus);
                }

                j1 = j1 + (t << 1);
            }

            t = t << 1;
            m = m >> 1;
        }

        T n_inv = OPERATOR<T>::modinv(parameters.n, parameters.modulus);

        for (int i = 0; i < parameters.n; i++)
        {
            output[i] = OPERATOR<T>::mult(output[i], n_inv, parameters.modulus);
        }

        return output;
    }

    template class NTTCPU<Data32>;
    template class NTTCPU<Data64>;
    
    template <typename T>
    void CooleyTukeyUnit(T& U, T& V, const Root<T>& root,
                                    const Modulus<T>& modulus)
    {
        // std::cout << "DEBUG: U=" << U << ", V=" << V 
        //       << ", root.value=" << root 
        //       << ", modulus.value=" << modulus.value << std::endl;
        T u_ = U;
        T v_ = OPERATOR_GPU<T>::mult(V, root, modulus);

        U = OPERATOR<T>::add(u_, v_, modulus);
        V = OPERATOR<T>::sub(u_, v_, modulus);
    }

    template <typename T>
    void GentlemanSandeUnit(T& U, T& V, const Root<T>& root,
                                       const Modulus<T>& modulus)
    {
        T u_ = U;
        T v_ = V;

        U = OPERATOR<T>::add(u_, v_, modulus);

        v_ = OPERATOR<T>::sub(u_, v_, modulus);
        V = OPERATOR_GPU<T>::mult(v_, root, modulus);
    }

//     template <typename T>
//     __global__ void
//     ForwardCoreLowRing(T* polynomial_in,
//                        typename std::make_unsigned<T>::type* polynomial_out,
//                        const Root<typename std::make_unsigned<
//                            T>::type>* __restrict__ root_of_unity_table,
//                        Modulus<typename std::make_unsigned<T>::type> modulus,
//                        int shared_index, int N_power, bool zero_padding,
//                        bool reduction_poly_check, int total_batch)
//     {
//         using TU = typename std::make_unsigned<T>::type;

//         const int idx_x = threadIdx.x;
//         const int idx_y = threadIdx.y;
//         const int block_x = blockIdx.x;
//         const int block_thread = idx_x + (idx_y * blockDim.x);
//         const int batch_index = (block_x * blockDim.y) + idx_y;

//         if (batch_index >= total_batch)
//             return;

//         int batch_offset = ((block_x + 1) * blockDim.y);
//         int batch_offset_size =
//             (batch_offset > total_batch)
//                 ? (blockDim.y - (batch_offset - total_batch))
//                 : blockDim.y;
//         int block_size = blockDim.x * batch_offset_size;

//         // extern __shared__ T shared_memory[];
//         extern __shared__ char shared_memory_typed[];
//         TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

//         const Modulus<TU> modulus_reg = modulus;

//         int t_2 = N_power - 1;
//         int offset = idx_y << N_power;
//         int t_ = shared_index;
//         int m = 1;

//         location_t global_addresss =
//             block_thread + (location_t) ((blockDim.y * block_x) << N_power);

//         location_t omega_addresss = idx_x;

//         // Load T from global & store to shared
//         if constexpr (std::is_signed<T>::value)
//         {
//             T input1_reg = polynomial_in[global_addresss];
//             T input2_reg = polynomial_in[global_addresss + block_size];
//             shared_memory[block_thread] =
//                 OPERATOR_GPU<TU>::reduce(input1_reg, modulus_reg);
//             shared_memory[block_thread + block_size] =
//                 OPERATOR_GPU<TU>::reduce(input2_reg, modulus_reg);
//         }
//         else
//         {
//             shared_memory[block_thread] = polynomial_in[global_addresss];
//             shared_memory[block_thread + block_size] =
//                 polynomial_in[global_addresss + block_size];
//         }

//         int shared_addresss = idx_x;

//         int t = 1 << t_;
//         int in_shared_address =
//             ((shared_addresss >> t_) << t_) + shared_addresss;
//         location_t current_root_index;
//         __syncthreads();

// #pragma unroll
//         for (int lp = 0; lp < (shared_index + 1); lp++)
//         {
//             int group_in_shared_address = in_shared_address + offset;
//             if (reduction_poly_check)
//             { // X_N_minus
//                 current_root_index = (omega_addresss >> t_2);
//             }
//             else
//             { // X_N_plus
//                 current_root_index = m + (omega_addresss >> t_2);
//             }

//             CooleyTukeyUnit(shared_memory[group_in_shared_address],
//                             shared_memory[group_in_shared_address + t],
//                             root_of_unity_table[current_root_index],
//                             modulus_reg);

//             t = t >> 1;
//             t_2 -= 1;
//             t_ -= 1;
//             m <<= 1;

//             in_shared_address =
//                 ((shared_addresss >> t_) << t_) + shared_addresss;
//             __syncthreads();
//         }

//         __syncthreads();

//         polynomial_out[global_addresss] = shared_memory[block_thread];
//         polynomial_out[global_addresss + block_size] =
//             shared_memory[block_thread + block_size];
//     }

//     template <typename T>
//     __global__ void ForwardCoreLowRing(
//         T* polynomial_in, typename std::make_unsigned<T>::type* polynomial_out,
//         const Root<typename std::make_unsigned<
//             T>::type>* __restrict__ root_of_unity_table,
//         Modulus<typename std::make_unsigned<T>::type>* modulus,
//         int shared_index, int N_power, bool zero_padding,
//         bool reduction_poly_check, int total_batch, int mod_count)
//     {
//         using TU = typename std::make_unsigned<T>::type;

//         const int idx_x = threadIdx.x;
//         const int idx_y = threadIdx.y;
//         const int block_x = blockIdx.x;
//         const int block_thread = idx_x + (idx_y * blockDim.x);
//         const int batch_index = (block_x * blockDim.y) + idx_y;

//         if (batch_index >= total_batch)
//             return;

//         int mod_index = batch_index % mod_count;
//         int batch_offset = ((block_x + 1) * blockDim.y);
//         int batch_offset_size =
//             (batch_offset > total_batch)
//                 ? (blockDim.y - (batch_offset - total_batch))
//                 : blockDim.y;
//         int block_size = blockDim.x * batch_offset_size;

//         // extern __shared__ T shared_memory[];
//         extern __shared__ char shared_memory_typed[];
//         TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

//         const Modulus<TU> modulus_reg = modulus[mod_index];

//         int t_2 = N_power - 1;
//         int offset = idx_y << N_power;
//         int t_ = shared_index;
//         int m = 1;

//         location_t global_addresss =
//             block_thread + (location_t) ((blockDim.y * block_x) << N_power);

//         location_t omega_addresss = idx_x;

//         // Load T from global & store to shared
//         if constexpr (std::is_signed<T>::value)
//         {
//             T input1_reg = polynomial_in[global_addresss];
//             T input2_reg = polynomial_in[global_addresss + block_size];
//             shared_memory[block_thread] =
//                 OPERATOR_GPU<TU>::reduce(input1_reg, modulus_reg);
//             shared_memory[block_thread + block_size] =
//                 OPERATOR_GPU<TU>::reduce(input2_reg, modulus_reg);
//         }
//         else
//         {
//             shared_memory[block_thread] = polynomial_in[global_addresss];
//             shared_memory[block_thread + block_size] =
//                 polynomial_in[global_addresss + block_size];
//         }

//         int shared_addresss = idx_x;

//         int t = 1 << t_;
//         int in_shared_address =
//             ((shared_addresss >> t_) << t_) + shared_addresss;
//         location_t current_root_index;
//         __syncthreads();

// #pragma unroll
//         for (int lp = 0; lp < (shared_index + 1); lp++)
//         {
//             int group_in_shared_address = in_shared_address + offset;
//             if (reduction_poly_check)
//             { // X_N_minus
//                 current_root_index = (omega_addresss >> t_2) +
//                                      (location_t) (mod_index << N_power);
//             }
//             else
//             { // X_N_plus
//                 current_root_index = m + (omega_addresss >> t_2) +
//                                      (location_t) (mod_index << N_power);
//             }

//             CooleyTukeyUnit(shared_memory[group_in_shared_address],
//                             shared_memory[group_in_shared_address + t],
//                             root_of_unity_table[current_root_index],
//                             modulus_reg);

//             t = t >> 1;
//             t_2 -= 1;
//             t_ -= 1;
//             m <<= 1;

//             in_shared_address =
//                 ((shared_addresss >> t_) << t_) + shared_addresss;
//             __syncthreads();
//         }

//         __syncthreads();

//         polynomial_out[global_addresss] = shared_memory[block_thread];
//         polynomial_out[global_addresss + block_size] =
//             shared_memory[block_thread + block_size];
//     }

    template <typename T>
    void ForwardCoreLowRing(KernelConfig current_kernel_params,
        T* polynomial_in, typename std::make_unsigned<T>::type* polynomial_out,
        const Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
        Modulus<typename std::make_unsigned<T>::type>* modulus,
        int shared_index, int N_power, bool zero_padding,
        bool reduction_poly_check, int total_batch, int mod_count)
    {
        using TU = typename std::make_unsigned<T>::type;
        
        // 计算块大小和线程数（在CPU上模拟）
        int blockdim_x = current_kernel_params.blockdim_x; // 估计值，需要根据实际情况调整
        int blockdim_y = current_kernel_params.blockdim_y;
        
        // 遍历所有批次
        for (int batch_index = 0; batch_index < total_batch; batch_index++)
        {
            int mod_index = batch_index % mod_count;
            const Modulus<TU> modulus_reg = modulus[mod_index];
            
            // 计算偏移量
            int offset = (batch_index / blockdim_y) << N_power;
            
            // 分配本地内存（模拟共享内存）
            std::vector<TU> shared_memory(2 * blockdim_x * blockdim_y);
            
            // 加载数据到本地内存
            for (int idx_x = 0; idx_x < blockdim_x; idx_x++)
            {
                for (int idx_y = 0; idx_y < blockdim_y; idx_y++)
                {
                    int block_thread = idx_x + (idx_y * blockdim_x);
                    location_t global_address = block_thread + offset;
                    
                    if constexpr (std::is_signed<T>::value)
                    {
                        T input1_reg = polynomial_in[global_address];
                        T input2_reg = polynomial_in[global_address + blockdim_x * blockdim_y];
                        shared_memory[block_thread] = 
                            OPERATOR<TU>::reduce(input1_reg, modulus_reg);
                        shared_memory[block_thread + blockdim_x * blockdim_y] = 
                            OPERATOR<TU>::reduce(input2_reg, modulus_reg);
                    }
                    else
                    {
                        shared_memory[block_thread] = polynomial_in[global_address];
                        shared_memory[block_thread + blockdim_x * blockdim_y] = 
                            polynomial_in[global_address + blockdim_x * blockdim_y];
                    }
                }
            }
            
            // 执行蝴蝶操作
            int t_2 = N_power - 1;
            int t = 1 << shared_index;
            int m = 1;
            int t_ = shared_index;
            
            for (int lp = 0; lp < (shared_index + 1); lp++)
            {
                for (int idx_x = 0; idx_x < blockdim_x; idx_x++)
                {
                    for (int idx_y = 0; idx_y < blockdim_y; idx_y++)
                    {
                        int shared_addresss = idx_x;
                        int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
                        int group_in_shared_address = in_shared_address + (idx_y << N_power);
                        
                        location_t current_root_index;
                        if (reduction_poly_check)
                        { // X_N_minus
                            current_root_index = (idx_x >> t_2) + (mod_index << N_power);
                        }
                        else
                        { // X_N_plus
                            current_root_index = m + (idx_x >> t_2) + (mod_index << N_power);
                        }
                        
                        // 执行蝴蝶操作
                        CooleyTukeyUnit(
                            shared_memory[group_in_shared_address],
                            shared_memory[group_in_shared_address + t],
                            root_of_unity_table[current_root_index],
                            modulus_reg);
                    }
                }
                
                // 更新变量
                t = t >> 1;
                t_2 -= 1;
                t_ -= 1;
                m <<= 1;
            }
            
            // 将结果写回输出
            for (int idx_x = 0; idx_x < blockdim_x; idx_x++)
            {
                for (int idx_y = 0; idx_y < blockdim_y; idx_y++)
                {
                    int block_thread = idx_x + (idx_y * blockdim_x);
                    location_t global_address = block_thread + offset;
                    
                    polynomial_out[global_address] = shared_memory[block_thread];
                    polynomial_out[global_address + blockdim_x * blockdim_y] = 
                        shared_memory[block_thread + blockdim_x * blockdim_y];
                }
            }
        }
    }

//     template <typename T>
//     __global__ void InverseCoreLowRing(
//         typename std::make_unsigned<T>::type* polynomial_in, T* polynomial_out,
//         const Root<typename std::make_unsigned<
//             T>::type>* __restrict__ inverse_root_of_unity_table,
//         Modulus<typename std::make_unsigned<T>::type> modulus, int shared_index,
//         int N_power, Ninverse<typename std::make_unsigned<T>::type> n_inverse,
//         bool reduction_poly_check, int total_batch)
//     {
//         using TU = typename std::make_unsigned<T>::type;

//         const int idx_x = threadIdx.x;
//         const int idx_y = threadIdx.y;
//         const int block_x = blockIdx.x;
//         const int block_thread = idx_x + (idx_y * blockDim.x);
//         const int batch_index = (block_x * blockDim.y) + idx_y;

//         if (batch_index >= total_batch)
//             return;

//         int batch_offset = ((block_x + 1) * blockDim.y);
//         int batch_offset_size =
//             (batch_offset > total_batch)
//                 ? (blockDim.y - (batch_offset - total_batch))
//                 : blockDim.y;
//         int block_size = blockDim.x * batch_offset_size;

//         // extern __shared__ T shared_memory[];
//         extern __shared__ char shared_memory_typed[];
//         TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

//         const Modulus<TU> modulus_reg = modulus;

//         int t_2 = 0;
//         int t_ = 0;
//         int offset = idx_y << N_power;
//         int loops = N_power;
//         int m = (int) 1 << (N_power - 1);

//         location_t global_addresss =
//             block_thread + (location_t) ((blockDim.y * block_x) << N_power);

//         location_t omega_addresss = idx_x;

//         // Load T from global & store to shared
//         shared_memory[block_thread] = polynomial_in[global_addresss];
//         shared_memory[block_thread + block_size] =
//             polynomial_in[global_addresss + block_size];

//         int shared_addresss = idx_x;

//         int t = 1 << t_;
//         int in_shared_address =
//             ((shared_addresss >> t_) << t_) + shared_addresss;
//         location_t current_root_index;
//         __syncthreads();

// #pragma unroll
//         for (int lp = 0; lp < loops; lp++)
//         {
//             int group_in_shared_address = in_shared_address + offset;
//             if (reduction_poly_check)
//             { // X_N_minus
//                 current_root_index = (omega_addresss >> t_2);
//             }
//             else
//             { // X_N_plus
//                 current_root_index = m + (omega_addresss >> t_2);
//             }

//             GentlemanSandeUnit(shared_memory[group_in_shared_address],
//                                shared_memory[group_in_shared_address + t],
//                                inverse_root_of_unity_table[current_root_index],
//                                modulus_reg);

//             t = t << 1;
//             t_2 += 1;
//             t_ += 1;
//             m >>= 1;

//             in_shared_address =
//                 ((shared_addresss >> t_) << t_) + shared_addresss;
//             __syncthreads();
//         }
//         __syncthreads();

//         TU output1_reg = OPERATOR_GPU<TU>::mult(shared_memory[block_thread],
//                                                 n_inverse, modulus_reg);
//         TU output2_reg = OPERATOR_GPU<TU>::mult(
//             shared_memory[block_thread + block_size], n_inverse, modulus_reg);

//         if constexpr (std::is_signed<T>::value)
//         {
//             polynomial_out[global_addresss] =
//                 OPERATOR_GPU<TU>::centered_reduction(output1_reg, modulus_reg);
//             polynomial_out[global_addresss + block_size] =
//                 OPERATOR_GPU<TU>::centered_reduction(output2_reg, modulus_reg);
//         }
//         else
//         {
//             polynomial_out[global_addresss] = output1_reg;
//             polynomial_out[global_addresss + block_size] = output2_reg;
//         }
//     }

//     template <typename T>
//     __global__ void InverseCoreLowRing(
//         typename std::make_unsigned<T>::type* polynomial_in, T* polynomial_out,
//         const Root<typename std::make_unsigned<
//             T>::type>* __restrict__ inverse_root_of_unity_table,
//         Modulus<typename std::make_unsigned<T>::type>* modulus,
//         int shared_index, int N_power,
//         Ninverse<typename std::make_unsigned<T>::type>* n_inverse,
//         bool reduction_poly_check, int total_batch, int mod_count)
//     {
//         using TU = typename std::make_unsigned<T>::type;

//         const int idx_x = threadIdx.x;
//         const int idx_y = threadIdx.y;
//         const int block_x = blockIdx.x;
//         const int block_thread = idx_x + (idx_y * blockDim.x);
//         const int batch_index = (block_x * blockDim.y) + idx_y;

//         if (batch_index >= total_batch)
//             return;

//         int mod_index = batch_index % mod_count;
//         int batch_offset = ((block_x + 1) * blockDim.y);
//         int batch_offset_size =
//             (batch_offset > total_batch)
//                 ? (blockDim.y - (batch_offset - total_batch))
//                 : blockDim.y;
//         int block_size = blockDim.x * batch_offset_size;

//         // extern __shared__ T shared_memory[];
//         extern __shared__ char shared_memory_typed[];
//         TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

//         const Modulus<TU> modulus_reg = modulus[mod_index];
//         const Ninverse<TU> n_inverse_reg = n_inverse[mod_index];

//         int t_2 = 0;
//         int t_ = 0;
//         int offset = idx_y << N_power;
//         int loops = N_power;
//         int m = (int) 1 << (N_power - 1);

//         location_t global_addresss =
//             block_thread + (location_t) ((blockDim.y * block_x) << N_power);

//         location_t omega_addresss = idx_x;

//         // Load T from global & store to shared
//         shared_memory[block_thread] = polynomial_in[global_addresss];
//         shared_memory[block_thread + block_size] =
//             polynomial_in[global_addresss + block_size];

//         int shared_addresss = idx_x;

//         int t = 1 << t_;
//         int in_shared_address =
//             ((shared_addresss >> t_) << t_) + shared_addresss;
//         location_t current_root_index;
//         __syncthreads();

// #pragma unroll
//         for (int lp = 0; lp < loops; lp++)
//         {
//             int group_in_shared_address = in_shared_address + offset;
//             if (reduction_poly_check)
//             { // X_N_minus
//                 current_root_index = (omega_addresss >> t_2);
//             }
//             else
//             { // X_N_plus
//                 current_root_index = m + (omega_addresss >> t_2);
//             }

//             GentlemanSandeUnit(shared_memory[group_in_shared_address],
//                                shared_memory[group_in_shared_address + t],
//                                inverse_root_of_unity_table[current_root_index],
//                                modulus_reg);

//             t = t << 1;
//             t_2 += 1;
//             t_ += 1;
//             m >>= 1;

//             in_shared_address =
//                 ((shared_addresss >> t_) << t_) + shared_addresss;
//             __syncthreads();
//         }
//         __syncthreads();

//         TU output1_reg = OPERATOR_GPU<TU>::mult(shared_memory[block_thread],
//                                                 n_inverse_reg, modulus_reg);
//         TU output2_reg =
//             OPERATOR_GPU<TU>::mult(shared_memory[block_thread + block_size],
//                                    n_inverse_reg, modulus_reg);

//         if constexpr (std::is_signed<T>::value)
//         {
//             polynomial_out[global_addresss] =
//                 OPERATOR_GPU<TU>::centered_reduction(output1_reg, modulus_reg);
//             polynomial_out[global_addresss + block_size] =
//                 OPERATOR_GPU<TU>::centered_reduction(output2_reg, modulus_reg);
//         }
//         else
//         {
//             polynomial_out[global_addresss] = output1_reg;
//             polynomial_out[global_addresss + block_size] = output2_reg;
//         }
//     }

//     template <typename T>
//     __global__ void ForwardCore(
//         T* polynomial_in, typename std::make_unsigned<T>::type* polynomial_out,
//         const Root<typename std::make_unsigned<
//             T>::type>* __restrict__ root_of_unity_table,
//         Modulus<typename std::make_unsigned<T>::type> modulus, int shared_index,
//         int logm, int outer_iteration_count, int N_power, bool zero_padding,
//         bool not_last_kernel, bool reduction_poly_check)
//     {
//         using TU = typename std::make_unsigned<T>::type;

//         const int idx_x = threadIdx.x;
//         const int idx_y = threadIdx.y;
//         const int block_x = blockIdx.x;
//         const int block_y = blockIdx.y;
//         const int block_z = blockIdx.z;

//         // extern __shared__ T shared_memory[];
//         extern __shared__ char shared_memory_typed[];
//         TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

//         const Modulus<TU> modulus_reg = modulus;

//         int t_2 = N_power - logm - 1;
//         location_t offset = 1 << (N_power - logm - 1);
//         int t_ = shared_index;
//         location_t m = (location_t) 1 << logm;

//         location_t global_addresss =
//             idx_x +
//             (location_t) (idx_y *
//                           (offset / (1 << (outer_iteration_count - 1)))) +
//             (location_t) (blockDim.x * block_x) +
//             (location_t) (2 * block_y * offset) +
//             (location_t) (block_z << N_power);

//         location_t omega_addresss =
//             idx_x +
//             (location_t) (idx_y *
//                           (offset / (1 << (outer_iteration_count - 1)))) +
//             (location_t) (blockDim.x * block_x) +
//             (location_t) (block_y * offset);

//         location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

//         // Load T from global & store to shared
//         if constexpr (std::is_signed<T>::value)
//         {
//             T input1_reg = polynomial_in[global_addresss];
//             T input2_reg = polynomial_in[global_addresss + offset];
//             shared_memory[shared_addresss] =
//                 OPERATOR_GPU<TU>::reduce(input1_reg, modulus_reg);
//             shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
//                 OPERATOR_GPU<TU>::reduce(input2_reg, modulus_reg);
//         }
//         else
//         {
//             shared_memory[shared_addresss] = polynomial_in[global_addresss];
//             shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
//                 polynomial_in[global_addresss + offset];
//         }

//         int t = 1 << t_;
//         int in_shared_address =
//             ((shared_addresss >> t_) << t_) + shared_addresss;
//         location_t current_root_index;
//         if (not_last_kernel)
//         {
// #pragma unroll
//             for (int lp = 0; lp < outer_iteration_count; lp++)
//             {
//                 __syncthreads();
//                 if (reduction_poly_check)
//                 { // X_N_minus
//                     current_root_index = (omega_addresss >> t_2);
//                 }
//                 else
//                 { // X_N_plus
//                     current_root_index = m + (omega_addresss >> t_2);
//                 }
//                 CooleyTukeyUnit(shared_memory[in_shared_address],
//                                 shared_memory[in_shared_address + t],
//                                 root_of_unity_table[current_root_index],
//                                 modulus_reg);

//                 t = t >> 1;
//                 t_2 -= 1;
//                 t_ -= 1;
//                 m <<= 1;

//                 in_shared_address =
//                     ((shared_addresss >> t_) << t_) + shared_addresss;
//                 //__syncthreads();
//             }
//             __syncthreads();
//         }
//         else
//         {
// #pragma unroll
//             for (int lp = 0; lp < (shared_index - 5); lp++) // 4 for 512 thread
//             {
//                 __syncthreads();
//                 if (reduction_poly_check)
//                 { // X_N_minus
//                     current_root_index = (omega_addresss >> t_2);
//                 }
//                 else
//                 { // X_N_plus
//                     current_root_index = m + (omega_addresss >> t_2);
//                 }

//                 CooleyTukeyUnit(shared_memory[in_shared_address],
//                                 shared_memory[in_shared_address + t],
//                                 root_of_unity_table[current_root_index],
//                                 modulus_reg);

//                 t = t >> 1;
//                 t_2 -= 1;
//                 t_ -= 1;
//                 m <<= 1;

//                 in_shared_address =
//                     ((shared_addresss >> t_) << t_) + shared_addresss;
//                 //__syncthreads();
//             }
//             __syncthreads();

// #pragma unroll
//             for (int lp = 0; lp < 6; lp++)
//             {
//                 if (reduction_poly_check)
//                 { // X_N_minus
//                     current_root_index = (omega_addresss >> t_2);
//                 }
//                 else
//                 { // X_N_plus
//                     current_root_index = m + (omega_addresss >> t_2);
//                 }
//                 CooleyTukeyUnit(shared_memory[in_shared_address],
//                                 shared_memory[in_shared_address + t],
//                                 root_of_unity_table[current_root_index],
//                                 modulus_reg);

//                 t = t >> 1;
//                 t_2 -= 1;
//                 t_ -= 1;
//                 m <<= 1;

//                 in_shared_address =
//                     ((shared_addresss >> t_) << t_) + shared_addresss;
//             }
//             __syncthreads();
//         }

//         polynomial_out[global_addresss] = shared_memory[shared_addresss];
//         polynomial_out[global_addresss + offset] =
//             shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
//     }

//     template <typename T>
//     __global__ void
//     ForwardCore(T* polynomial_in,
//                 typename std::make_unsigned<T>::type* polynomial_out,
//                 const Root<typename std::make_unsigned<
//                     T>::type>* __restrict__ root_of_unity_table,
//                 Modulus<typename std::make_unsigned<T>::type>* modulus,
//                 int shared_index, int logm, int outer_iteration_count,
//                 int N_power, bool zero_padding, bool not_last_kernel,
//                 bool reduction_poly_check, int mod_count)
//     {
//         using TU = typename std::make_unsigned<T>::type;

//         const int idx_x = threadIdx.x;
//         const int idx_y = threadIdx.y;
//         const int block_x = blockIdx.x;
//         const int block_y = blockIdx.y;
//         const int block_z = blockIdx.z;

//         const int mod_index = block_z % mod_count;

//         // extern __shared__ T shared_memory[];
//         extern __shared__ char shared_memory_typed[];
//         TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

//         const Modulus<TU> modulus_reg = modulus[mod_index];

//         int t_2 = N_power - logm - 1;
//         location_t offset = 1 << (N_power - logm - 1);
//         int t_ = shared_index;
//         location_t m = (location_t) 1 << logm;

//         location_t global_addresss =
//             idx_x +
//             (location_t) (idx_y *
//                           (offset / (1 << (outer_iteration_count - 1)))) +
//             (location_t) (blockDim.x * block_x) +
//             (location_t) (2 * block_y * offset) +
//             (location_t) (block_z << N_power);

//         location_t omega_addresss =
//             idx_x +
//             (location_t) (idx_y *
//                           (offset / (1 << (outer_iteration_count - 1)))) +
//             (location_t) (blockDim.x * block_x) +
//             (location_t) (block_y * offset);

//         location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

//         // Load T from global & store to shared
//         if constexpr (std::is_signed<T>::value)
//         {
//             T input1_reg = polynomial_in[global_addresss];
//             T input2_reg = polynomial_in[global_addresss + offset];
//             shared_memory[shared_addresss] =
//                 OPERATOR_GPU<TU>::reduce(input1_reg, modulus_reg);
//             shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
//                 OPERATOR_GPU<TU>::reduce(input2_reg, modulus_reg);
//         }
//         else
//         {
//             shared_memory[shared_addresss] = polynomial_in[global_addresss];
//             shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
//                 polynomial_in[global_addresss + offset];
//         }

//         int t = 1 << t_;
//         int in_shared_address =
//             ((shared_addresss >> t_) << t_) + shared_addresss;
//         location_t current_root_index;
//         if (not_last_kernel)
//         {
// #pragma unroll
//             for (int lp = 0; lp < outer_iteration_count; lp++)
//             {
//                 __syncthreads();
//                 if (reduction_poly_check)
//                 { // X_N_minus
//                     current_root_index = (omega_addresss >> t_2) +
//                                          (location_t) (mod_index << N_power);
//                 }
//                 else
//                 { // X_N_plus
//                     current_root_index = m + (omega_addresss >> t_2) +
//                                          (location_t) (mod_index << N_power);
//                 }
//                 CooleyTukeyUnit(shared_memory[in_shared_address],
//                                 shared_memory[in_shared_address + t],
//                                 root_of_unity_table[current_root_index],
//                                 modulus_reg);

//                 t = t >> 1;
//                 t_2 -= 1;
//                 t_ -= 1;
//                 m <<= 1;

//                 in_shared_address =
//                     ((shared_addresss >> t_) << t_) + shared_addresss;
//                 //__syncthreads();
//             }
//             __syncthreads();
//         }
//         else
//         {
// #pragma unroll
//             for (int lp = 0; lp < (shared_index - 5); lp++) // 4 for 512 thread
//             {
//                 __syncthreads();
//                 if (reduction_poly_check)
//                 { // X_N_minus
//                     current_root_index = (omega_addresss >> t_2) +
//                                          (location_t) (mod_index << N_power);
//                 }
//                 else
//                 { // X_N_plus
//                     current_root_index = m + (omega_addresss >> t_2) +
//                                          (location_t) (mod_index << N_power);
//                 }

//                 CooleyTukeyUnit(shared_memory[in_shared_address],
//                                 shared_memory[in_shared_address + t],
//                                 root_of_unity_table[current_root_index],
//                                 modulus_reg);

//                 t = t >> 1;
//                 t_2 -= 1;
//                 t_ -= 1;
//                 m <<= 1;

//                 in_shared_address =
//                     ((shared_addresss >> t_) << t_) + shared_addresss;
//                 //__syncthreads();
//             }
//             __syncthreads();

// #pragma unroll
//             for (int lp = 0; lp < 6; lp++)
//             {
//                 if (reduction_poly_check)
//                 { // X_N_minus
//                     current_root_index = (omega_addresss >> t_2) +
//                                          (location_t) (mod_index << N_power);
//                 }
//                 else
//                 { // X_N_plus
//                     current_root_index = m + (omega_addresss >> t_2) +
//                                          (location_t) (mod_index << N_power);
//                 }
//                 CooleyTukeyUnit(shared_memory[in_shared_address],
//                                 shared_memory[in_shared_address + t],
//                                 root_of_unity_table[current_root_index],
//                                 modulus_reg);

//                 t = t >> 1;
//                 t_2 -= 1;
//                 t_ -= 1;
//                 m <<= 1;

//                 in_shared_address =
//                     ((shared_addresss >> t_) << t_) + shared_addresss;
//             }
//             __syncthreads();
//         }

//         polynomial_out[global_addresss] = shared_memory[shared_addresss];
//         polynomial_out[global_addresss + offset] =
//             shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
//     }

    template <typename T>
    void ForwardCoreCPU(
        T* polynomial_in, T* polynomial_out,
        const Root<T>* root_of_unity_table,
        Modulus<T>* modulus, int shared_index, int logm,
        int outer_iteration_count, int N_power,
        bool zero_padding, bool not_last_kernel,
        bool reduction_poly_check, int mod_count,
        int grid_x, int grid_y, int grid_z, 
        int block_size_x, int block_size_y,
        size_t shared_memory_size) {
        
        // 计算总多项式大小
        int n = 1 << N_power;
        int total_polynomials = grid_z * mod_count;
        
        // 分配共享内存（模拟）
        int shared_memory_elements = shared_memory_size / sizeof(T);
        if (shared_memory_elements < 2 * block_size_x * block_size_y) {
            throw std::invalid_argument("Shared memory size is too small");
        }
        
        std::vector<T> shared_memory(shared_memory_elements);
        
        // 遍历所有block和thread
        for (int block_z = 0; block_z < grid_z; block_z++) {
            for (int block_y = 0; block_y < grid_y; block_y++) {
                for (int block_x = 0; block_x < grid_x; block_x++) {
                    // 处理当前block中的所有线程
                    for (int thread_y = 0; thread_y < block_size_y; thread_y++) {
                        for (int thread_x = 0; thread_x < block_size_x; thread_x++) {
                            // 模拟线程索引
                            int idx_x = thread_x;
                            int idx_y = thread_y;
                            
                            // 模拟block索引
                            int block_x_idx = block_x;
                            int block_y_idx = block_y;
                            int block_z_idx = block_z;
                            
                            // 计算模数索引
                            int mod_index = block_z_idx % mod_count;
                            
                            // 计算各种位置和偏移
                            int t_2 = N_power - logm - 1;
                            location_t offset = 1 << (N_power - logm - 1);
                            int t_ = shared_index;
                            location_t m = (location_t) 1 << logm;
                            
                            // 计算全局地址
                            location_t global_addresss =
                                idx_x +
                                (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
                                (location_t)(block_size_x * block_x_idx) +
                                (location_t)(2 * block_y_idx * offset) + 
                                (location_t)(block_z_idx << N_power);
                            
                            // 检查全局地址是否越界
                            if (global_addresss >= n * total_polynomials) {
                                throw std::out_of_range("Global address out of range: " + std::to_string(global_addresss));
                            }
                            
                            // 计算omega地址
                            location_t omega_addresss =
                                idx_x +
                                (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
                                (location_t)(block_size_x * block_x_idx) + 
                                (location_t)(block_y_idx * offset);
                            
                            // 计算共享内存地址
                            location_t shared_addresss = (idx_x + (idx_y * block_size_x));
                            
                            // 检查共享内存地址是否越界
                            if (shared_addresss >= shared_memory_elements || 
                                shared_addresss + (block_size_x * block_size_y) >= shared_memory_elements) {
                                throw std::out_of_range("Shared memory address out of range");
                            }
                            
                            // 从全局内存加载到共享内存（模拟）
                            shared_memory[shared_addresss] = polynomial_in[global_addresss];
                            shared_memory[shared_addresss + (block_size_x * block_size_y)] =
                                polynomial_in[global_addresss + offset];
                            
                            int t = 1 << t_;
                            int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
                            
                            // 检查in_shared_address是否越界
                            if (in_shared_address >= shared_memory_elements || 
                                in_shared_address + t >= shared_memory_elements) {
                                throw std::out_of_range("In-shared address out of range");
                            }
                            
                            location_t current_root_index;
                            
                            if (not_last_kernel) {
                                for (int lp = 0; lp < outer_iteration_count; lp++) {
                                    if (reduction_poly_check) { // X_N_minus
                                        current_root_index = (omega_addresss >> t_2) +
                                                            (location_t)(mod_index << N_power);
                                    } else { // X_N_plus
                                        current_root_index = m + (omega_addresss >> t_2) +
                                                            (location_t)(mod_index << N_power);
                                    }
                                    
                                    // 检查根索引是否越界
                                    if (current_root_index >= n * mod_count) {
                                        throw std::out_of_range("Root index out of range: " + std::to_string(current_root_index));
                                    }
                                    
                                    CooleyTukeyUnit<T>(
                                        shared_memory[in_shared_address],
                                        shared_memory[in_shared_address + t],
                                        root_of_unity_table[current_root_index],
                                        modulus[mod_index]);
                                    
                                    t = t >> 1;
                                    t_2 -= 1;
                                    t_ -= 1;
                                    m <<= 1;
                                    
                                    in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
                                    
                                    // 检查更新后的in_shared_address是否越界
                                    if (in_shared_address >= shared_memory_elements || 
                                        in_shared_address + t >= shared_memory_elements) {
                                        throw std::out_of_range("Updated in-shared address out of range");
                                    }
                                }
                            } else {
                                for (int lp = 0; lp < (shared_index - 5); lp++) {
                                    if (reduction_poly_check) { // X_N_minus
                                        current_root_index = (omega_addresss >> t_2) +
                                                            (location_t)(mod_index << N_power);
                                    } else { // X_N_plus
                                        current_root_index = m + (omega_addresss >> t_2) +
                                                            (location_t)(mod_index << N_power);
                                    }
                                    
                                    // 检查根索引是否越界
                                    if (current_root_index >= n * mod_count) {
                                        throw std::out_of_range("Root index out of range: " + std::to_string(current_root_index));
                                    }
                                    
                                    CooleyTukeyUnit<T>(
                                        shared_memory[in_shared_address],
                                        shared_memory[in_shared_address + t],
                                        root_of_unity_table[current_root_index],
                                        modulus[mod_index]);
                                    
                                    t = t >> 1;
                                    t_2 -= 1;
                                    t_ -= 1;
                                    m <<= 1;
                                    
                                    in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
                                    
                                    // 检查更新后的in_shared_address是否越界
                                    if (in_shared_address >= shared_memory_elements || 
                                        in_shared_address + t >= shared_memory_elements) {
                                        throw std::out_of_range("Updated in-shared address out of range");
                                    }
                                }
                                
                                for (int lp = 0; lp < 6; lp++) {
                                    if (reduction_poly_check) { // X_N_minus
                                        current_root_index = (omega_addresss >> t_2) +
                                                            (location_t)(mod_index << N_power);
                                    } else { // X_N_plus
                                        current_root_index = m + (omega_addresss >> t_2) +
                                                            (location_t)(mod_index << N_power);
                                    }
                                    
                                    // 检查根索引是否越界
                                    if (current_root_index >= n * mod_count) {
                                        throw std::out_of_range("Root index out of range: " + std::to_string(current_root_index));
                                    }
                                    
                                    CooleyTukeyUnit<T>(
                                        shared_memory[in_shared_address],
                                        shared_memory[in_shared_address + t],
                                        root_of_unity_table[current_root_index],
                                        modulus[mod_index]);
                                    
                                    t = t >> 1;
                                    t_2 -= 1;
                                    t_ -= 1;
                                    m <<= 1;
                                    
                                    in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
                                    
                                    // 检查更新后的in_shared_address是否越界
                                    if (in_shared_address >= shared_memory_elements || 
                                        in_shared_address + t >= shared_memory_elements) {
                                        throw std::out_of_range("Updated in-shared address out of range");
                                    }
                                }
                            }
                            
                            // 将结果写回全局内存
                            polynomial_out[global_addresss] = shared_memory[shared_addresss];
                            polynomial_out[global_addresss + offset] =
                                shared_memory[shared_addresss + (block_size_x * block_size_y)];
                        }
                    }
                }
            }
        }
    }
//     template <typename T>
//     __global__ void ForwardCore_(
//         T* polynomial_in, typename std::make_unsigned<T>::type* polynomial_out,
//         const Root<typename std::make_unsigned<
//             T>::type>* __restrict__ root_of_unity_table,
//         Modulus<typename std::make_unsigned<T>::type> modulus, int shared_index,
//         int logm, int outer_iteration_count, int N_power, bool zero_padding,
//         bool not_last_kernel, bool reduction_poly_check)
//     {
//         using TU = typename std::make_unsigned<T>::type;

//         const int idx_x = threadIdx.x;
//         const int idx_y = threadIdx.y;
//         const int block_x = blockIdx.x;
//         const int block_y = blockIdx.y;
//         const int block_z = blockIdx.z;

//         // extern __shared__ T shared_memory[];
//         extern __shared__ char shared_memory_typed[];
//         TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

//         const Modulus<TU> modulus_reg = modulus;

//         int t_2 = N_power - logm - 1;
//         location_t offset = 1 << (N_power - logm - 1);
//         int t_ = shared_index;
//         location_t m = (location_t) 1 << logm;

//         location_t global_addresss =
//             idx_x +
//             (location_t) (idx_y *
//                           (offset / (1 << (outer_iteration_count - 1)))) +
//             (location_t) (blockDim.x * block_y) +
//             (location_t) (2 * block_x * offset) +
//             (location_t) (block_z << N_power);
//         location_t omega_addresss =
//             idx_x +
//             (location_t) (idx_y *
//                           (offset / (1 << (outer_iteration_count - 1)))) +
//             (location_t) (blockDim.x * block_y) +
//             (location_t) (block_x * offset);
//         location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

//         // Load T from global & store to shared
//         if constexpr (std::is_signed<T>::value)
//         {
//             T input1_reg = polynomial_in[global_addresss];
//             T input2_reg = polynomial_in[global_addresss + offset];
//             shared_memory[shared_addresss] =
//                 OPERATOR_GPU<TU>::reduce(input1_reg, modulus_reg);
//             shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
//                 OPERATOR_GPU<TU>::reduce(input2_reg, modulus_reg);
//         }
//         else
//         {
//             shared_memory[shared_addresss] = polynomial_in[global_addresss];
//             shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
//                 polynomial_in[global_addresss + offset];
//         }

//         int t = 1 << t_;
//         int in_shared_address =
//             ((shared_addresss >> t_) << t_) + shared_addresss;
//         location_t current_root_index;
//         if (not_last_kernel)
//         {
// #pragma unroll
//             for (int lp = 0; lp < outer_iteration_count; lp++)
//             {
//                 __syncthreads();
//                 if (reduction_poly_check)
//                 { // X_N_minus
//                     current_root_index = (omega_addresss >> t_2);
//                 }
//                 else
//                 { // X_N_plus
//                     current_root_index = m + (omega_addresss >> t_2);
//                 }
//                 CooleyTukeyUnit(shared_memory[in_shared_address],
//                                 shared_memory[in_shared_address + t],
//                                 root_of_unity_table[current_root_index],
//                                 modulus_reg);

//                 t = t >> 1;
//                 t_2 -= 1;
//                 t_ -= 1;
//                 m <<= 1;

//                 in_shared_address =
//                     ((shared_addresss >> t_) << t_) + shared_addresss;
//                 //__syncthreads();
//             }
//             __syncthreads();
//         }
//         else
//         {
// #pragma unroll
//             for (int lp = 0; lp < (shared_index - 5); lp++)
//             {
//                 __syncthreads();
//                 if (reduction_poly_check)
//                 { // X_N_minus
//                     current_root_index = (omega_addresss >> t_2);
//                 }
//                 else
//                 { // X_N_plus
//                     current_root_index = m + (omega_addresss >> t_2);
//                 }

//                 CooleyTukeyUnit(shared_memory[in_shared_address],
//                                 shared_memory[in_shared_address + t],
//                                 root_of_unity_table[current_root_index],
//                                 modulus_reg);

//                 t = t >> 1;
//                 t_2 -= 1;
//                 t_ -= 1;
//                 m <<= 1;

//                 in_shared_address =
//                     ((shared_addresss >> t_) << t_) + shared_addresss;
//                 //__syncthreads();
//             }
//             __syncthreads();

// #pragma unroll
//             for (int lp = 0; lp < 6; lp++)
//             {
//                 if (reduction_poly_check)
//                 { // X_N_minus
//                     current_root_index = (omega_addresss >> t_2);
//                 }
//                 else
//                 { // X_N_plus
//                     current_root_index = m + (omega_addresss >> t_2);
//                 }
//                 CooleyTukeyUnit(shared_memory[in_shared_address],
//                                 shared_memory[in_shared_address + t],
//                                 root_of_unity_table[current_root_index],
//                                 modulus_reg);

//                 t = t >> 1;
//                 t_2 -= 1;
//                 t_ -= 1;
//                 m <<= 1;

//                 in_shared_address =
//                     ((shared_addresss >> t_) << t_) + shared_addresss;
//             }
//             __syncthreads();
//         }

//         polynomial_out[global_addresss] = shared_memory[shared_addresss];
//         polynomial_out[global_addresss + offset] =
//             shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
//     }

//     template <typename T>
//     __global__ void
//     ForwardCore_(T* polynomial_in,
//                  typename std::make_unsigned<T>::type* polynomial_out,
//                  const Root<typename std::make_unsigned<
//                      T>::type>* __restrict__ root_of_unity_table,
//                  Modulus<typename std::make_unsigned<T>::type>* modulus,
//                  int shared_index, int logm, int outer_iteration_count,
//                  int N_power, bool zero_padding, bool not_last_kernel,
//                  bool reduction_poly_check, int mod_count)
//     {
//         using TU = typename std::make_unsigned<T>::type;

//         const int idx_x = threadIdx.x;
//         const int idx_y = threadIdx.y;
//         const int block_x = blockIdx.x;
//         const int block_y = blockIdx.y;
//         const int block_z = blockIdx.z;

//         const int mod_index = block_z % mod_count;

//         // extern __shared__ T shared_memory[];
//         extern __shared__ char shared_memory_typed[];
//         TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

//         const Modulus<TU> modulus_reg = modulus[mod_index];

//         int t_2 = N_power - logm - 1;
//         location_t offset = 1 << (N_power - logm - 1);
//         int t_ = shared_index;
//         location_t m = (location_t) 1 << logm;

//         location_t global_addresss =
//             idx_x +
//             (location_t) (idx_y *
//                           (offset / (1 << (outer_iteration_count - 1)))) +
//             (location_t) (blockDim.x * block_y) +
//             (location_t) (2 * block_x * offset) +
//             (location_t) (block_z << N_power);
//         location_t omega_addresss =
//             idx_x +
//             (location_t) (idx_y *
//                           (offset / (1 << (outer_iteration_count - 1)))) +
//             (location_t) (blockDim.x * block_y) +
//             (location_t) (block_x * offset);
//         location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

//         // Load T from global & store to shared
//         if constexpr (std::is_signed<T>::value)
//         {
//             T input1_reg = polynomial_in[global_addresss];
//             T input2_reg = polynomial_in[global_addresss + offset];
//             shared_memory[shared_addresss] =
//                 OPERATOR_GPU<TU>::reduce(input1_reg, modulus_reg);
//             shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
//                 OPERATOR_GPU<TU>::reduce(input2_reg, modulus_reg);
//         }
//         else
//         {
//             shared_memory[shared_addresss] = polynomial_in[global_addresss];
//             shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
//                 polynomial_in[global_addresss + offset];
//         }

//         int t = 1 << t_;
//         int in_shared_address =
//             ((shared_addresss >> t_) << t_) + shared_addresss;
//         location_t current_root_index;
//         if (not_last_kernel)
//         {
// #pragma unroll
//             for (int lp = 0; lp < outer_iteration_count; lp++)
//             {
//                 __syncthreads();
//                 if (reduction_poly_check)
//                 { // X_N_minus
//                     current_root_index = (omega_addresss >> t_2) +
//                                          (location_t) (mod_index << N_power);
//                 }
//                 else
//                 { // X_N_plus
//                     current_root_index = m + (omega_addresss >> t_2) +
//                                          (location_t) (mod_index << N_power);
//                 }
//                 CooleyTukeyUnit(shared_memory[in_shared_address],
//                                 shared_memory[in_shared_address + t],
//                                 root_of_unity_table[current_root_index],
//                                 modulus_reg);

//                 t = t >> 1;
//                 t_2 -= 1;
//                 t_ -= 1;
//                 m <<= 1;

//                 in_shared_address =
//                     ((shared_addresss >> t_) << t_) + shared_addresss;
//                 //__syncthreads();
//             }
//             __syncthreads();
//         }
//         else
//         {
// #pragma unroll
//             for (int lp = 0; lp < (shared_index - 5); lp++)
//             {
//                 __syncthreads();
//                 if (reduction_poly_check)
//                 { // X_N_minus
//                     current_root_index = (omega_addresss >> t_2) +
//                                          (location_t) (mod_index << N_power);
//                 }
//                 else
//                 { // X_N_plus
//                     current_root_index = m + (omega_addresss >> t_2) +
//                                          (location_t) (mod_index << N_power);
//                 }
//                 CooleyTukeyUnit(shared_memory[in_shared_address],
//                                 shared_memory[in_shared_address + t],
//                                 root_of_unity_table[current_root_index],
//                                 modulus_reg);

//                 t = t >> 1;
//                 t_2 -= 1;
//                 t_ -= 1;
//                 m <<= 1;

//                 in_shared_address =
//                     ((shared_addresss >> t_) << t_) + shared_addresss;
//                 //__syncthreads();
//             }
//             __syncthreads();

// #pragma unroll
//             for (int lp = 0; lp < 6; lp++)
//             {
//                 if (reduction_poly_check)
//                 { // X_N_minus
//                     current_root_index = (omega_addresss >> t_2) +
//                                          (location_t) (mod_index << N_power);
//                 }
//                 else
//                 { // X_N_plus
//                     current_root_index = m + (omega_addresss >> t_2) +
//                                          (location_t) (mod_index << N_power);
//                 }
//                 CooleyTukeyUnit(shared_memory[in_shared_address],
//                                 shared_memory[in_shared_address + t],
//                                 root_of_unity_table[current_root_index],
//                                 modulus_reg);

//                 t = t >> 1;
//                 t_2 -= 1;
//                 t_ -= 1;
//                 m <<= 1;

//                 in_shared_address =
//                     ((shared_addresss >> t_) << t_) + shared_addresss;
//             }
//             __syncthreads();
//         }

//         polynomial_out[global_addresss] = shared_memory[shared_addresss];
//         polynomial_out[global_addresss + offset] =
//             shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
//     }

    template <typename T>
    void ForwardCore_CPU(T* polynomial_in, T* polynomial_out,
                        const Root<T>* root_of_unity_table,
                        Modulus<T>* modulus, int shared_index, int logm,
                        int outer_iteration_count, int N_power,
                        bool zero_padding, bool not_last_kernel,
                        bool reduction_poly_check, int mod_count,
                        int griddim_x, int griddim_y, int batch_size,
                        int blockdim_x, int blockdim_y) {
        
        // 计算共享内存大小（以T为单位）
        int shared_memory_size = blockdim_x * blockdim_y * 2 * 1024 * sizeof(T);
        
        // 遍历所有网格和块
        for (int block_z = 0; block_z < batch_size; block_z++) {
            for (int block_y = 0; block_y < griddim_y; block_y++) {
                for (int block_x = 0; block_x < griddim_x; block_x++) {
                    
                    // 分配共享内存
                    std::vector<T> shared_memory(shared_memory_size);
                    
                    const int mod_index = block_z % mod_count;
                    
                    int t_2 = N_power - logm - 1;
                    location_t offset = 1 << (N_power - logm - 1);
                    int t_ = shared_index;
                    location_t m = (location_t) 1 << logm;
                    
                    // 处理所有线程
                    for (int idx_y = 0; idx_y < blockdim_y; idx_y++) {
                        for (int idx_x = 0; idx_x < blockdim_x; idx_x++) {
                            location_t global_addresss =
                                idx_x +
                                (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
                                (location_t) (blockdim_x * block_y) +
                                (location_t) (2 * block_x * offset) + (location_t) (block_z << N_power);
                            
                            location_t omega_addresss =
                                idx_x +
                                (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
                                (location_t) (blockdim_x * block_y) + (location_t) (block_x * offset);
                            
                            location_t shared_addresss = (idx_x + (idx_y * blockdim_x));
                            
                            // 从全局内存加载数据到共享内存
                            shared_memory[shared_addresss] = polynomial_in[global_addresss];
                            shared_memory[shared_addresss + (blockdim_x * blockdim_y)] =
                                polynomial_in[global_addresss + offset];
                            
                            int t = 1 << t_;
                            int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
                            location_t current_root_index;
                            
                            if (not_last_kernel) {
                                for (int lp = 0; lp < outer_iteration_count; lp++) {
                                    if (reduction_poly_check) { // X_N_minus
                                        current_root_index = (omega_addresss >> t_2) +
                                                            (location_t) (mod_index << N_power);
                                    } else { // X_N_plus
                                        current_root_index = m + (omega_addresss >> t_2) +
                                                            (location_t) (mod_index << N_power);
                                    }
                                    
                                    CooleyTukeyUnit(shared_memory[in_shared_address],
                                                    shared_memory[in_shared_address + t],
                                                    root_of_unity_table[current_root_index],
                                                    modulus[mod_index]);
                                    
                                    t = t >> 1;
                                    t_2 -= 1;
                                    t_ -= 1;
                                    m <<= 1;
                                    
                                    in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
                                }
                            } else {
                                for (int lp = 0; lp < (shared_index - 5); lp++) {
                                    if (reduction_poly_check) { // X_N_minus
                                        current_root_index = (omega_addresss >> t_2) +
                                                            (location_t) (mod_index << N_power);
                                    } else { // X_N_plus
                                        current_root_index = m + (omega_addresss >> t_2) +
                                                            (location_t) (mod_index << N_power);
                                    }
                                    
                                    CooleyTukeyUnit(shared_memory[in_shared_address],
                                                    shared_memory[in_shared_address + t],
                                                    root_of_unity_table[current_root_index],
                                                    modulus[mod_index]);
                                    
                                    t = t >> 1;
                                    t_2 -= 1;
                                    t_ -= 1;
                                    m <<= 1;
                                    
                                    in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
                                }
                                
                                for (int lp = 0; lp < 6; lp++) {
                                    if (reduction_poly_check) { // X_N_minus
                                        current_root_index = (omega_addresss >> t_2) +
                                                            (location_t) (mod_index << N_power);
                                    } else { // X_N_plus
                                        current_root_index = m + (omega_addresss >> t_2) +
                                                            (location_t) (mod_index << N_power);
                                    }
                                    
                                    CooleyTukeyUnit(shared_memory[in_shared_address],
                                                    shared_memory[in_shared_address + t],
                                                    root_of_unity_table[current_root_index],
                                                    modulus[mod_index]);
                                    
                                    t = t >> 1;
                                    t_2 -= 1;
                                    t_ -= 1;
                                    m <<= 1;
                                    
                                    in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
                                }
                            }
                            
                            // 将结果写回全局内存
                            polynomial_out[global_addresss] = shared_memory[shared_addresss];
                            polynomial_out[global_addresss + offset] =
                                shared_memory[shared_addresss + (blockdim_x * blockdim_y)];
                        }
                    }
                }
            }
        }
    }

//     template <typename T>
//     __global__ void InverseCore(
//         typename std::make_unsigned<T>::type* polynomial_in, T* polynomial_out,
//         const Root<typename std::make_unsigned<
//             T>::type>* __restrict__ inverse_root_of_unity_table,
//         Modulus<typename std::make_unsigned<T>::type> modulus, int shared_index,
//         int logm, int k, int outer_iteration_count, int N_power,
//         Ninverse<typename std::make_unsigned<T>::type> n_inverse,
//         bool last_kernel, bool reduction_poly_check)
//     {
//         using TU = typename std::make_unsigned<T>::type;

//         const int idx_x = threadIdx.x;
//         const int idx_y = threadIdx.y;
//         const int block_x = blockIdx.x;
//         const int block_y = blockIdx.y;
//         const int block_z = blockIdx.z;

//         // extern __shared__ T shared_memory[];
//         extern __shared__ char shared_memory_typed[];
//         TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

//         const Modulus<TU> modulus_reg = modulus;
//         const Ninverse<TU> n_inverse_reg = n_inverse;

//         int t_2 = N_power - logm - 1;
//         location_t offset = 1 << (N_power - k - 1);
//         // int t_ = 9 - outer_iteration_count;
//         int t_ = (shared_index + 1) - outer_iteration_count;
//         int loops = outer_iteration_count;
//         location_t m = (location_t) 1 << logm;

//         location_t global_addresss =
//             idx_x +
//             (location_t) (idx_y *
//                           (offset / (1 << (outer_iteration_count - 1)))) +
//             (location_t) (blockDim.x * block_x) +
//             (location_t) (2 * block_y * offset) +
//             (location_t) (block_z << N_power);

//         location_t omega_addresss =
//             idx_x +
//             (location_t) (idx_y *
//                           (offset / (1 << (outer_iteration_count - 1)))) +
//             (location_t) (blockDim.x * block_x) +
//             (location_t) (block_y * offset);
//         location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

//         shared_memory[shared_addresss] = polynomial_in[global_addresss];
//         shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
//             polynomial_in[global_addresss + offset];

//         int t = 1 << t_;
//         int in_shared_address =
//             ((shared_addresss >> t_) << t_) + shared_addresss;
//         location_t current_root_index;
// #pragma unroll
//         for (int lp = 0; lp < loops; lp++)
//         {
//             __syncthreads();
//             if (reduction_poly_check)
//             { // X_N_minus
//                 current_root_index = (omega_addresss >> t_2);
//             }
//             else
//             { // X_N_plus
//                 current_root_index = m + (omega_addresss >> t_2);
//             }

//             GentlemanSandeUnit(shared_memory[in_shared_address],
//                                shared_memory[in_shared_address + t],
//                                inverse_root_of_unity_table[current_root_index],
//                                modulus_reg);

//             t = t << 1;
//             t_2 += 1;
//             t_ += 1;
//             m >>= 1;

//             in_shared_address =
//                 ((shared_addresss >> t_) << t_) + shared_addresss;
//         }
//         __syncthreads();

//         if (last_kernel)
//         {
//             TU output1_reg = OPERATOR_GPU<TU>::mult(
//                 shared_memory[shared_addresss], n_inverse_reg, modulus_reg);
//             TU output2_reg = OPERATOR_GPU<TU>::mult(
//                 shared_memory[shared_addresss + (blockDim.x * blockDim.y)],
//                 n_inverse_reg, modulus_reg);

//             if constexpr (std::is_signed<T>::value)
//             {
//                 polynomial_out[global_addresss] =
//                     OPERATOR_GPU<TU>::centered_reduction(output1_reg,
//                                                          modulus_reg);
//                 polynomial_out[global_addresss + offset] =
//                     OPERATOR_GPU<TU>::centered_reduction(output2_reg,
//                                                          modulus_reg);
//             }
//             else
//             {
//                 polynomial_out[global_addresss] = output1_reg;
//                 polynomial_out[global_addresss + offset] = output2_reg;
//             }
//         }
//         else
//         {
//             polynomial_out[global_addresss] = shared_memory[shared_addresss];
//             polynomial_out[global_addresss + offset] =
//                 shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
//         }
//     }

//     template <typename T>
//     __global__ void InverseCore(
//         typename std::make_unsigned<T>::type* polynomial_in, T* polynomial_out,
//         const Root<typename std::make_unsigned<
//             T>::type>* __restrict__ inverse_root_of_unity_table,
//         Modulus<typename std::make_unsigned<T>::type>* modulus,
//         int shared_index, int logm, int k, int outer_iteration_count,
//         int N_power, Ninverse<typename std::make_unsigned<T>::type>* n_inverse,
//         bool last_kernel, bool reduction_poly_check, int mod_count)
//     {
//         using TU = typename std::make_unsigned<T>::type;

//         const int idx_x = threadIdx.x;
//         const int idx_y = threadIdx.y;
//         const int block_x = blockIdx.x;
//         const int block_y = blockIdx.y;
//         const int block_z = blockIdx.z;

//         const int mod_index = block_z % mod_count;

//         // extern __shared__ T shared_memory[];
//         extern __shared__ char shared_memory_typed[];
//         TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

//         const Modulus<TU> modulus_reg = modulus[mod_index];
//         const Ninverse<TU> n_inverse_reg = n_inverse[mod_index];

//         int t_2 = N_power - logm - 1;
//         location_t offset = 1 << (N_power - k - 1);
//         // int t_ = 9 - outer_iteration_count;
//         int t_ = (shared_index + 1) - outer_iteration_count;
//         int loops = outer_iteration_count;
//         location_t m = (location_t) 1 << logm;

//         location_t global_addresss =
//             idx_x +
//             (location_t) (idx_y *
//                           (offset / (1 << (outer_iteration_count - 1)))) +
//             (location_t) (blockDim.x * block_x) +
//             (location_t) (2 * block_y * offset) +
//             (location_t) (block_z << N_power);

//         location_t omega_addresss =
//             idx_x +
//             (location_t) (idx_y *
//                           (offset / (1 << (outer_iteration_count - 1)))) +
//             (location_t) (blockDim.x * block_x) +
//             (location_t) (block_y * offset);
//         location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

//         shared_memory[shared_addresss] = polynomial_in[global_addresss];
//         shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
//             polynomial_in[global_addresss + offset];

//         int t = 1 << t_;
//         int in_shared_address =
//             ((shared_addresss >> t_) << t_) + shared_addresss;
//         location_t current_root_index;
// #pragma unroll
//         for (int lp = 0; lp < loops; lp++)
//         {
//             __syncthreads();
//             if (reduction_poly_check)
//             { // X_N_minus
//                 current_root_index = (omega_addresss >> t_2) +
//                                      (location_t) (mod_index << N_power);
//             }
//             else
//             { // X_N_plus
//                 current_root_index = m + (omega_addresss >> t_2) +
//                                      (location_t) (mod_index << N_power);
//             }

//             GentlemanSandeUnit(shared_memory[in_shared_address],
//                                shared_memory[in_shared_address + t],
//                                inverse_root_of_unity_table[current_root_index],
//                                modulus[mod_index]);

//             t = t << 1;
//             t_2 += 1;
//             t_ += 1;
//             m >>= 1;

//             in_shared_address =
//                 ((shared_addresss >> t_) << t_) + shared_addresss;
//         }
//         __syncthreads();

//         if (last_kernel)
//         {
//             TU output1_reg = OPERATOR_GPU<TU>::mult(
//                 shared_memory[shared_addresss], n_inverse_reg, modulus_reg);
//             TU output2_reg = OPERATOR_GPU<TU>::mult(
//                 shared_memory[shared_addresss + (blockDim.x * blockDim.y)],
//                 n_inverse_reg, modulus_reg);

//             if constexpr (std::is_signed<T>::value)
//             {
//                 polynomial_out[global_addresss] =
//                     OPERATOR_GPU<TU>::centered_reduction(output1_reg,
//                                                          modulus_reg);
//                 polynomial_out[global_addresss + offset] =
//                     OPERATOR_GPU<TU>::centered_reduction(output2_reg,
//                                                          modulus_reg);
//             }
//             else
//             {
//                 polynomial_out[global_addresss] = output1_reg;
//                 polynomial_out[global_addresss + offset] = output2_reg;
//             }
//         }
//         else
//         {
//             polynomial_out[global_addresss] = shared_memory[shared_addresss];
//             polynomial_out[global_addresss + offset] =
//                 shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
//         }
//     }
    template <typename T>
    void InverseCoreCPU(
        T* polynomial_in, T* polynomial_out,
        const Root<T>* inverse_root_of_unity_table,
        Modulus<T>* modulus, int shared_index, int logm, int k,
        int outer_iteration_count, int N_power, Ninverse<T>* n_inverse,
        bool last_kernel, bool reduction_poly_check, int mod_count,
        int grid_x, int grid_y, int grid_z, 
        int block_size_x, int block_size_y,
        int shared_memory_size) {
        
        // 计算总多项式大小
        int n = 1 << N_power;
        int total_polynomials = grid_z * mod_count;
        
        // 分配共享内存（模拟）
        int shared_memory_elements = shared_memory_size / sizeof(T);
        if (shared_memory_elements < 2 * block_size_x * block_size_y) {
            throw std::invalid_argument("Shared memory size is too small");
        }
        
        std::vector<T> shared_memory(shared_memory_elements);
        
        // 遍历所有block和thread
        for (int block_z = 0; block_z < grid_z; block_z++) {
            for (int block_y = 0; block_y < grid_y; block_y++) {
                for (int block_x = 0; block_x < grid_x; block_x++) {
                    // 处理当前block中的所有线程
                    for (int thread_y = 0; thread_y < block_size_y; thread_y++) {
                        for (int thread_x = 0; thread_x < block_size_x; thread_x++) {
                            // 模拟线程索引
                            int idx_x = thread_x;
                            int idx_y = thread_y;
                            
                            // 模拟block索引
                            int block_x_idx = block_x;
                            int block_y_idx = block_y;
                            int block_z_idx = block_z;
                            
                            // 计算模数索引
                            int mod_index = block_z_idx % mod_count;
                            
                            // 计算各种位置和偏移
                            int t_2 = N_power - logm - 1;
                            location_t offset = 1 << (N_power - k - 1);
                            int t_ = (shared_index + 1) - outer_iteration_count;
                            int loops = outer_iteration_count;
                            location_t m = (location_t) 1 << logm;
                            
                            // 计算全局地址
                            location_t global_addresss =
                                idx_x +
                                (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
                                (location_t)(block_size_x * block_x_idx) +
                                (location_t)(2 * block_y_idx * offset) + 
                                (location_t)(block_z_idx << N_power);
                            
                            // 检查全局地址是否越界
                            if (global_addresss >= n * total_polynomials) {
                                throw std::out_of_range("Global address out of range: " + std::to_string(global_addresss));
                            }
                            
                            // 计算omega地址
                            location_t omega_addresss =
                                idx_x +
                                (location_t)(idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
                                (location_t)(block_size_x * block_x_idx) + 
                                (location_t)(block_y_idx * offset);
                            
                            // 计算共享内存地址
                            location_t shared_addresss = (idx_x + (idx_y * block_size_x));
                            
                            // 检查共享内存地址是否越界
                            if (shared_addresss >= shared_memory_elements || 
                                shared_addresss + (block_size_x * block_size_y) >= shared_memory_elements) {
                                throw std::out_of_range("Shared memory address out of range");
                            }
                            
                            // 从全局内存加载到共享内存（模拟）
                            shared_memory[shared_addresss] = polynomial_in[global_addresss];
                            shared_memory[shared_addresss + (block_size_x * block_size_y)] =
                                polynomial_in[global_addresss + offset];
                            
                            int t = 1 << t_;
                            int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
                            
                            // 检查in_shared_address是否越界
                            if (in_shared_address >= shared_memory_elements || 
                                in_shared_address + t >= shared_memory_elements) {
                                throw std::out_of_range("In-shared address out of range");
                            }
                            
                            location_t current_root_index;
                            
                            for (int lp = 0; lp < loops; lp++) {
                                if (reduction_poly_check) { // X_N_minus
                                    current_root_index =
                                        (omega_addresss >> t_2) + (location_t)(mod_index << N_power);
                                } else { // X_N_plus
                                    current_root_index = m + (omega_addresss >> t_2) +
                                                        (location_t)(mod_index << N_power);
                                }
                                
                                // 检查根索引是否越界
                                if (current_root_index >= n * mod_count) {
                                    throw std::out_of_range("Root index out of range: " + std::to_string(current_root_index));
                                }
                                
                                GentlemanSandeUnit<T>(
                                    shared_memory[in_shared_address],
                                    shared_memory[in_shared_address + t],
                                    inverse_root_of_unity_table[current_root_index],
                                    modulus[mod_index]);
                                
                                t = t << 1;
                                t_2 += 1;
                                t_ += 1;
                                m >>= 1;
                                
                                in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
                                
                                // 检查更新后的in_shared_address是否越界
                                if (in_shared_address >= shared_memory_elements || 
                                    in_shared_address + t >= shared_memory_elements) {
                                    throw std::out_of_range("Updated in-shared address out of range");
                                }
                            }
                            
                            // 将结果写回全局内存
                            if (last_kernel) {
                                polynomial_out[global_addresss] =
                                    OPERATOR_GPU<T>::mult(shared_memory[shared_addresss],
                                                        n_inverse[mod_index], modulus[mod_index]);
                                polynomial_out[global_addresss + offset] = 
                                    OPERATOR_GPU<T>::mult(
                                        shared_memory[shared_addresss + (block_size_x * block_size_y)],
                                        n_inverse[mod_index], modulus[mod_index]);
                            } else {
                                polynomial_out[global_addresss] = shared_memory[shared_addresss];
                                polynomial_out[global_addresss + offset] =
                                    shared_memory[shared_addresss + (block_size_x * block_size_y)];
                            }
                        }
                    }
                }
            }
        }
    }

//     template <typename T>
//     __global__ void InverseCore_(
//         typename std::make_unsigned<T>::type* polynomial_in, T* polynomial_out,
//         const Root<typename std::make_unsigned<
//             T>::type>* __restrict__ inverse_root_of_unity_table,
//         Modulus<typename std::make_unsigned<T>::type> modulus, int shared_index,
//         int logm, int k, int outer_iteration_count, int N_power,
//         Ninverse<typename std::make_unsigned<T>::type> n_inverse,
//         bool last_kernel, bool reduction_poly_check)
//     {
//         using TU = typename std::make_unsigned<T>::type;

//         const int idx_x = threadIdx.x;
//         const int idx_y = threadIdx.y;
//         const int block_x = blockIdx.x;
//         const int block_y = blockIdx.y;
//         const int block_z = blockIdx.z;

//         // extern __shared__ T shared_memory[];
//         extern __shared__ char shared_memory_typed[];
//         TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

//         const Modulus<TU> modulus_reg = modulus;
//         const Ninverse<TU> n_inverse_reg = n_inverse;

//         int t_2 = N_power - logm - 1;
//         location_t offset = 1 << (N_power - k - 1);
//         // int t_ = 9 - outer_iteration_count;
//         int t_ = (shared_index + 1) - outer_iteration_count;
//         int loops = outer_iteration_count;
//         location_t m = (location_t) 1 << logm;

//         location_t global_addresss =
//             idx_x +
//             (location_t) (idx_y *
//                           (offset / (1 << (outer_iteration_count - 1)))) +
//             (location_t) (blockDim.x * block_y) +
//             (location_t) (2 * block_x * offset) +
//             (location_t) (block_z << N_power);

//         location_t omega_addresss =
//             idx_x +
//             (location_t) (idx_y *
//                           (offset / (1 << (outer_iteration_count - 1)))) +
//             (location_t) (blockDim.x * block_y) +
//             (location_t) (block_x * offset);
//         location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

//         shared_memory[shared_addresss] = polynomial_in[global_addresss];
//         shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
//             polynomial_in[global_addresss + offset];

//         int t = 1 << t_;
//         int in_shared_address =
//             ((shared_addresss >> t_) << t_) + shared_addresss;
//         location_t current_root_index;
// #pragma unroll
//         for (int lp = 0; lp < loops; lp++)
//         {
//             __syncthreads();
//             if (reduction_poly_check)
//             { // X_N_minus
//                 current_root_index = (omega_addresss >> t_2);
//             }
//             else
//             { // X_N_plus
//                 current_root_index = m + (omega_addresss >> t_2);
//             }

//             GentlemanSandeUnit(shared_memory[in_shared_address],
//                                shared_memory[in_shared_address + t],
//                                inverse_root_of_unity_table[current_root_index],
//                                modulus);

//             t = t << 1;
//             t_2 += 1;
//             t_ += 1;
//             m >>= 1;

//             in_shared_address =
//                 ((shared_addresss >> t_) << t_) + shared_addresss;
//         }
//         __syncthreads();

//         if (last_kernel)
//         {
//             TU output1_reg = OPERATOR_GPU<TU>::mult(
//                 shared_memory[shared_addresss], n_inverse_reg, modulus_reg);
//             TU output2_reg = OPERATOR_GPU<TU>::mult(
//                 shared_memory[shared_addresss + (blockDim.x * blockDim.y)],
//                 n_inverse_reg, modulus_reg);

//             if constexpr (std::is_signed<T>::value)
//             {
//                 polynomial_out[global_addresss] =
//                     OPERATOR_GPU<TU>::centered_reduction(output1_reg,
//                                                          modulus_reg);
//                 polynomial_out[global_addresss + offset] =
//                     OPERATOR_GPU<TU>::centered_reduction(output2_reg,
//                                                          modulus_reg);
//             }
//             else
//             {
//                 polynomial_out[global_addresss] = output1_reg;
//                 polynomial_out[global_addresss + offset] = output2_reg;
//             }
//         }
//         else
//         {
//             polynomial_out[global_addresss] = shared_memory[shared_addresss];
//             polynomial_out[global_addresss + offset] =
//                 shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
//         }
//     }

//     template <typename T>
//     __global__ void InverseCore_(
//         typename std::make_unsigned<T>::type* polynomial_in, T* polynomial_out,
//         const Root<typename std::make_unsigned<
//             T>::type>* __restrict__ inverse_root_of_unity_table,
//         Modulus<typename std::make_unsigned<T>::type>* modulus,
//         int shared_index, int logm, int k, int outer_iteration_count,
//         int N_power, Ninverse<typename std::make_unsigned<T>::type>* n_inverse,
//         bool last_kernel, bool reduction_poly_check, int mod_count)
//     {
//         using TU = typename std::make_unsigned<T>::type;

//         const int idx_x = threadIdx.x;
//         const int idx_y = threadIdx.y;
//         const int block_x = blockIdx.x;
//         const int block_y = blockIdx.y;
//         const int block_z = blockIdx.z;

//         const int mod_index = block_z % mod_count;

//         // extern __shared__ T shared_memory[];
//         extern __shared__ char shared_memory_typed[];
//         TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);

//         const Modulus<TU> modulus_reg = modulus[mod_index];
//         const Ninverse<TU> n_inverse_reg = n_inverse[mod_index];

//         int t_2 = N_power - logm - 1;
//         location_t offset = 1 << (N_power - k - 1);
//         // int t_ = 9 - outer_iteration_count;
//         int t_ = (shared_index + 1) - outer_iteration_count;
//         int loops = outer_iteration_count;
//         location_t m = (location_t) 1 << logm;

//         location_t global_addresss =
//             idx_x +
//             (location_t) (idx_y *
//                           (offset / (1 << (outer_iteration_count - 1)))) +
//             (location_t) (blockDim.x * block_y) +
//             (location_t) (2 * block_x * offset) +
//             (location_t) (block_z << N_power);

//         location_t omega_addresss =
//             idx_x +
//             (location_t) (idx_y *
//                           (offset / (1 << (outer_iteration_count - 1)))) +
//             (location_t) (blockDim.x * block_y) +
//             (location_t) (block_x * offset);
//         location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

//         shared_memory[shared_addresss] = polynomial_in[global_addresss];
//         shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
//             polynomial_in[global_addresss + offset];

//         int t = 1 << t_;
//         int in_shared_address =
//             ((shared_addresss >> t_) << t_) + shared_addresss;
//         location_t current_root_index;
// #pragma unroll
//         for (int lp = 0; lp < loops; lp++)
//         {
//             __syncthreads();
//             if (reduction_poly_check)
//             { // X_N_minus
//                 current_root_index = (omega_addresss >> t_2) +
//                                      (location_t) (mod_index << N_power);
//             }
//             else
//             { // X_N_plus
//                 current_root_index = m + (omega_addresss >> t_2) +
//                                      (location_t) (mod_index << N_power);
//             }

//             GentlemanSandeUnit(shared_memory[in_shared_address],
//                                shared_memory[in_shared_address + t],
//                                inverse_root_of_unity_table[current_root_index],
//                                modulus[mod_index]);

//             t = t << 1;
//             t_2 += 1;
//             t_ += 1;
//             m >>= 1;

//             in_shared_address =
//                 ((shared_addresss >> t_) << t_) + shared_addresss;
//         }
//         __syncthreads();

//         if (last_kernel)
//         {
//             TU output1_reg = OPERATOR_GPU<TU>::mult(
//                 shared_memory[shared_addresss], n_inverse_reg, modulus_reg);
//             TU output2_reg = OPERATOR_GPU<TU>::mult(
//                 shared_memory[shared_addresss + (blockDim.x * blockDim.y)],
//                 n_inverse_reg, modulus_reg);

//             if constexpr (std::is_signed<T>::value)
//             {
//                 polynomial_out[global_addresss] =
//                     OPERATOR_GPU<TU>::centered_reduction(output1_reg,
//                                                          modulus_reg);
//                 polynomial_out[global_addresss + offset] =
//                     OPERATOR_GPU<TU>::centered_reduction(output2_reg,
//                                                          modulus_reg);
//             }
//             else
//             {
//                 polynomial_out[global_addresss] = output1_reg;
//                 polynomial_out[global_addresss + offset] = output2_reg;
//             }
//         }
//         else
//         {
//             polynomial_out[global_addresss] = shared_memory[shared_addresss];
//             polynomial_out[global_addresss + offset] =
//                 shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
//         }
//     }
    template <typename T>
    void InverseCore_CPU(T* polynomial_in, T* polynomial_out,
                        const Root<T>* inverse_root_of_unity_table,
                        Modulus<T>* modulus, int shared_index, int logm, int k,
                        int outer_iteration_count, int N_power, Ninverse<T>* n_inverse,
                        bool last_kernel, bool reduction_poly_check, int mod_count,
                        int griddim_x, int griddim_y, int batch_size,
                        int blockdim_x, int blockdim_y) {
        
        // 计算共享内存大小（以T为单位）
        int shared_memory_size = blockdim_x * blockdim_y * 2 * sizeof(T);
        
        // 遍历所有网格和块
        for (int block_z = 0; block_z < batch_size; block_z++) {
            for (int block_y = 0; block_y < griddim_y; block_y++) {
                for (int block_x = 0; block_x < griddim_x; block_x++) {
                    
                    // 分配共享内存
                    std::vector<T> shared_memory(shared_memory_size);
                    
                    const int mod_index = block_z % mod_count;
                    
                    int t_2 = N_power - logm - 1;
                    location_t offset = 1 << (N_power - k - 1);
                    int t_ = (shared_index + 1) - outer_iteration_count;
                    int loops = outer_iteration_count;
                    location_t m = (location_t) 1 << logm;
                    
                    // 处理所有线程
                    for (int idx_y = 0; idx_y < blockdim_y; idx_y++) {
                        for (int idx_x = 0; idx_x < blockdim_x; idx_x++) {
                            location_t global_addresss =
                                idx_x +
                                (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
                                (location_t) (blockdim_x * block_y) +
                                (location_t) (2 * block_x * offset) + (location_t) (block_z << N_power);
                            
                            location_t omega_addresss =
                                idx_x +
                                (location_t) (idx_y * (offset / (1 << (outer_iteration_count - 1)))) +
                                (location_t) (blockdim_x * block_y) + (location_t) (block_x * offset);
                            
                            location_t shared_addresss = (idx_x + (idx_y * blockdim_x));
                            
                            // 从全局内存加载数据到共享内存
                            shared_memory[shared_addresss] = polynomial_in[global_addresss];
                            shared_memory[shared_addresss + (blockdim_x * blockdim_y)] =
                                polynomial_in[global_addresss + offset];
                            
                            int t = 1 << t_;
                            int in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
                            location_t current_root_index;
                            
                            for (int lp = 0; lp < loops; lp++) {
                                if (reduction_poly_check) { // X_N_minus
                                    current_root_index =
                                        (omega_addresss >> t_2) + (location_t) (mod_index << N_power);
                                } else { // X_N_plus
                                    current_root_index = m + (omega_addresss >> t_2) +
                                                        (location_t) (mod_index << N_power);
                                }
                                
                                GentlemanSandeUnit(shared_memory[in_shared_address],
                                                shared_memory[in_shared_address + t],
                                                inverse_root_of_unity_table[current_root_index],
                                                modulus[mod_index]);
                                
                                t = t << 1;
                                t_2 += 1;
                                t_ += 1;
                                m >>= 1;
                                
                                in_shared_address = ((shared_addresss >> t_) << t_) + shared_addresss;
                            }
                            
                            // 处理最后一步
                            if (last_kernel) {
                                polynomial_out[global_addresss] =
                                    OPERATOR<T>::mult(shared_memory[shared_addresss],
                                            n_inverse[mod_index], modulus[mod_index]);
                                polynomial_out[global_addresss + offset] = OPERATOR<T>::mult(
                                    shared_memory[shared_addresss + (blockdim_x * blockdim_y)],
                                    n_inverse[mod_index], modulus[mod_index]);
                            } else {
                                polynomial_out[global_addresss] = shared_memory[shared_addresss];
                                polynomial_out[global_addresss + offset] =
                                    shared_memory[shared_addresss + (blockdim_x * blockdim_y)];
                            }
                        }
                    }
                }
            }
        }
    }
//     template <typename T>
//     __global__ void ForwardCoreTranspose(
//         T* polynomial_in, typename std::make_unsigned<T>::type* polynomial_out,
//         const Root<typename std::make_unsigned<
//             T>::type>* __restrict__ root_of_unity_table,
//         const Modulus<typename std::make_unsigned<T>::type> modulus,
//         int log_row, int log_column, bool reduction_poly_check)
//     {
//         using TU = typename std::make_unsigned<T>::type;

//         const int idx_x = threadIdx.x;
//         const int idx_y = threadIdx.y;
//         const int block_x = blockIdx.x;

//         extern __shared__ char shared_memory_typed[];
//         TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);
//         TU* root_shared_memory = shared_memory + (1024);

//         const Modulus<TU> modulus_thread = modulus;

//         if (idx_x == 0)
//         {
//             if (reduction_poly_check)
//             { // X_N_minus
//                 root_shared_memory[idx_y] = root_of_unity_table[idx_y];
//             }
//             else
//             { // X_N_plus
//                 root_shared_memory[idx_y] = root_of_unity_table[idx_y];
//                 root_shared_memory[idx_y + blockDim.y] =
//                     root_of_unity_table[idx_y + blockDim.y];
//             }
//         }

//         location_t global_addresss =
//             (idx_y << log_column) + (blockDim.x * block_x) + idx_x;
//         location_t global_offset = (blockDim.y << log_column);

//         int t_ = log_row - 1;
//         int m = 1;

//         // Load T from global & store to shared
//         const int transpose_block = idx_y + (idx_x * (2 * blockDim.y));
//         if constexpr (std::is_signed<T>::value)
//         {
//             T input1_reg = polynomial_in[global_addresss];
//             T input2_reg = polynomial_in[global_addresss + global_offset];
//             shared_memory[transpose_block] =
//                 OPERATOR_GPU<TU>::reduce(input1_reg, modulus_thread);
//             shared_memory[transpose_block + blockDim.y] =
//                 OPERATOR_GPU<TU>::reduce(input2_reg, modulus_thread);
//         }
//         else
//         {
//             shared_memory[transpose_block] = polynomial_in[global_addresss];
//             shared_memory[transpose_block + blockDim.y] =
//                 polynomial_in[global_addresss + global_offset];
//         }

//         const int block_thread_index = idx_x + (idx_y * blockDim.x);
//         const int ntt_thread_index = block_thread_index & ((1 << t_) - 1);
//         const int ntt_block_index = block_thread_index >> t_;
//         int offset = ntt_block_index << log_row;

//         int shared_addresss = ntt_thread_index;
//         int omega_addresss = ntt_thread_index;
//         int t = 1 << t_;
//         int in_shared_address =
//             ((shared_addresss >> t_) << t_) + shared_addresss;
//         location_t current_root_index;
//         __syncthreads();

//         int loop = ((log_row - 6) <= 0) ? log_row : 6;

// #pragma unroll
//         for (int lp = 0; lp < (log_row - 6); lp++)
//         {
//             int group_in_shared_address = in_shared_address + offset;

//             if (reduction_poly_check)
//             { // X_N_minus
//                 current_root_index = (omega_addresss >> t_);
//             }
//             else
//             { // X_N_plus
//                 current_root_index = m + (omega_addresss >> t_);
//             }

//             Root<TU> root = root_shared_memory[current_root_index];

//             CooleyTukeyUnit(shared_memory[group_in_shared_address],
//                             shared_memory[group_in_shared_address + t], root,
//                             modulus_thread);

//             t = t >> 1;
//             t_ -= 1;
//             m <<= 1;

//             in_shared_address =
//                 ((shared_addresss >> t_) << t_) + shared_addresss;
//             __syncthreads();
//         }

// #pragma unroll
//         for (int lp = 0; lp < loop; lp++)
//         {
//             int group_in_shared_address = in_shared_address + offset;

//             if (reduction_poly_check)
//             { // X_N_minus
//                 current_root_index = (omega_addresss >> t_);
//             }
//             else
//             { // X_N_plus
//                 current_root_index = m + (omega_addresss >> t_);
//             }

//             Root<TU> root = root_shared_memory[current_root_index];

//             CooleyTukeyUnit(shared_memory[group_in_shared_address],
//                             shared_memory[group_in_shared_address + t], root,
//                             modulus_thread);

//             t = t >> 1;
//             t_ -= 1;
//             m <<= 1;

//             in_shared_address =
//                 ((shared_addresss >> t_) << t_) + shared_addresss;
//         }

//         __syncthreads();

//         polynomial_out[global_addresss] = shared_memory[transpose_block];
//         polynomial_out[global_addresss + global_offset] =
//             shared_memory[transpose_block + blockDim.y];
//     }

//     template <typename T>
//     __global__ void ForwardCoreTranspose(
//         T* polynomial_in, typename std::make_unsigned<T>::type* polynomial_out,
//         const Root<typename std::make_unsigned<
//             T>::type>* __restrict__ root_of_unity_table,
//         const Modulus<typename std::make_unsigned<T>::type>* modulus,
//         int log_row, int log_column, bool reduction_poly_check, int mod_count)
//     {
//         using TU = typename std::make_unsigned<T>::type;

//         const int idx_x = threadIdx.x;
//         const int idx_y = threadIdx.y;
//         const int block_x = blockIdx.x;
//         const int total_block_thread = blockDim.x * blockDim.y;

//         extern __shared__ char shared_memory_typed[];
//         TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);
//         TU* root_shared_memory = shared_memory + (1024);

//         if (idx_x == 0)
//         {
//             if (reduction_poly_check)
//             { // X_N_minus
//                 root_shared_memory[idx_y] = root_of_unity_table[idx_y];
//             }
//             else
//             { // X_N_plus
//                 root_shared_memory[idx_y] = root_of_unity_table[idx_y];
//                 root_shared_memory[idx_y + blockDim.y] =
//                     root_of_unity_table[idx_y + blockDim.y];
//             }
//         }

//         location_t global_addresss =
//             (idx_y << log_column) + (blockDim.x * block_x) + idx_x;
//         location_t global_offset = (blockDim.y << log_column);

//         int t_ = log_row - 1;
//         int m = 1;

//         const int block_thread_index = idx_x + (idx_y * blockDim.x);
//         const int ntt_thread_index = block_thread_index & ((1 << t_) - 1);
//         const int ntt_block_index = block_thread_index >> t_;
//         int offset = ntt_block_index << log_row;

//         const int batch_index =
//             (block_x * (total_block_thread >> t_)) + ntt_block_index;
//         int mod_index = batch_index % mod_count;

//         const Modulus<TU> modulus_thread = modulus[mod_index];

//         // Load T from global & store to shared
//         const int transpose_block = idx_y + (idx_x * (2 * blockDim.y));
//         if constexpr (std::is_signed<T>::value)
//         {
//             T input1_reg = polynomial_in[global_addresss];
//             T input2_reg = polynomial_in[global_addresss + global_offset];
//             shared_memory[transpose_block] =
//                 OPERATOR_GPU<TU>::reduce(input1_reg, modulus_thread);
//             shared_memory[transpose_block + blockDim.y] =
//                 OPERATOR_GPU<TU>::reduce(input2_reg, modulus_thread);
//         }
//         else
//         {
//             shared_memory[transpose_block] = polynomial_in[global_addresss];
//             shared_memory[transpose_block + blockDim.y] =
//                 polynomial_in[global_addresss + global_offset];
//         }

//         int shared_addresss = ntt_thread_index;
//         int omega_addresss = ntt_thread_index;
//         int t = 1 << t_;
//         int in_shared_address =
//             ((shared_addresss >> t_) << t_) + shared_addresss;
//         location_t current_root_index;
//         __syncthreads();

//         int loop = ((log_row - 6) <= 0) ? log_row : 6;

// #pragma unroll
//         for (int lp = 0; lp < (log_row - 6); lp++)
//         {
//             int group_in_shared_address = in_shared_address + offset;

//             if (reduction_poly_check)
//             { // X_N_minus
//                 current_root_index = (omega_addresss >> t_);
//             }
//             else
//             { // X_N_plus
//                 current_root_index = m + (omega_addresss >> t_);
//             }

//             Root<TU> root = root_shared_memory[current_root_index];

//             CooleyTukeyUnit(shared_memory[group_in_shared_address],
//                             shared_memory[group_in_shared_address + t], root,
//                             modulus_thread);

//             t = t >> 1;
//             t_ -= 1;
//             m <<= 1;

//             in_shared_address =
//                 ((shared_addresss >> t_) << t_) + shared_addresss;
//             __syncthreads();
//         }

// #pragma unroll
//         for (int lp = 0; lp < loop; lp++)
//         {
//             int group_in_shared_address = in_shared_address + offset;

//             if (reduction_poly_check)
//             { // X_N_minus
//                 current_root_index =
//                     (omega_addresss >> t_) + (mod_index << log_row);
//             }
//             else
//             { // X_N_plus
//                 current_root_index =
//                     m + (omega_addresss >> t_) + (mod_index << log_row);
//             }

//             Root<TU> root = root_shared_memory[current_root_index];

//             CooleyTukeyUnit(shared_memory[group_in_shared_address],
//                             shared_memory[group_in_shared_address + t], root,
//                             modulus_thread);

//             t = t >> 1;
//             t_ -= 1;
//             m <<= 1;

//             in_shared_address =
//                 ((shared_addresss >> t_) << t_) + shared_addresss;
//         }

//         __syncthreads();

//         polynomial_out[global_addresss] = shared_memory[transpose_block];
//         polynomial_out[global_addresss + global_offset] =
//             shared_memory[transpose_block + blockDim.y];
//     }

    template <typename T>
    void ForwardCoreTranspose(int total_block_thread, int total_block_count,
        int blockdim_x, int blockdim_y,
        T* polynomial_in, typename std::make_unsigned<T>::type* polynomial_out,
        const Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
        const Modulus<typename std::make_unsigned<T>::type>* modulus,
        int log_row, int log_column, bool reduction_poly_check, int mod_count)
    {
        using TU = typename std::make_unsigned<T>::type;
         
        
        // 分配共享内存（模拟CUDA共享内存）
        std::vector<TU> shared_memory(total_block_thread * 2 + (1 << log_row));
        TU* root_shared_memory = shared_memory.data() + 1024; // 模拟CUDA中的偏移
        
        // 初始化根表到共享内存
        if (reduction_poly_check)
        { // X_N_minus
            for (int idx_y = 0; idx_y < blockdim_y; idx_y++)
            {
                root_shared_memory[idx_y] = root_of_unity_table[idx_y];
            }
        }
        else
        { // X_N_plus
            for (int idx_y = 0; idx_y < blockdim_y; idx_y++)
            {
                root_shared_memory[idx_y] = root_of_unity_table[idx_y];
                root_shared_memory[idx_y + blockdim_y] = 
                    root_of_unity_table[idx_y + blockdim_y];
            }
        }
        
        // 遍历所有块
        for (int block_x = 0; block_x < total_block_count; block_x++)
        {
            // 遍历所有线程
            for (int idx_y = 0; idx_y < blockdim_y; idx_y++)
            {
                for (int idx_x = 0; idx_x < blockdim_x; idx_x++)
                {
                    // 计算全局地址
                    location_t global_address = 
                        (idx_y << log_column) + (blockdim_x * block_x) + idx_x;
                    location_t global_offset = (blockdim_y << log_column);
                    
                    // 初始化变量
                    int t_ = log_row - 1;
                    int m = 1;
                    
                    // 计算线程索引
                    const int block_thread_index = idx_x + (idx_y * blockdim_x);
                    const int ntt_thread_index = block_thread_index & ((1 << t_) - 1);
                    const int ntt_block_index = block_thread_index >> t_;
                    int offset = ntt_block_index << log_row;
                    
                    // 计算批次索引和模数索引
                    const int batch_index = 
                        (block_x * (total_block_thread >> t_)) + ntt_block_index;
                    int mod_index = batch_index % mod_count;
                    const Modulus<TU> modulus_thread = modulus[mod_index];
                    
                    // 加载数据到共享内存（包含转置操作）
                    const int transpose_block = idx_y + (idx_x * (2 * blockdim_y));
                    if constexpr (std::is_signed<T>::value)
                    {
                        T input1_reg = polynomial_in[global_address];
                        T input2_reg = polynomial_in[global_address + global_offset];
                        shared_memory[transpose_block] = 
                            OPERATOR<TU>::reduce(input1_reg, modulus_thread);
                        shared_memory[transpose_block + blockdim_y] = 
                            OPERATOR<TU>::reduce(input2_reg, modulus_thread);
                    }
                    else
                    {
                        shared_memory[transpose_block] = polynomial_in[global_address];
                        shared_memory[transpose_block + blockdim_y] = 
                            polynomial_in[global_address + global_offset];
                    }
                    
                    // 初始化蝴蝶操作变量
                    int shared_address = ntt_thread_index;
                    int omega_address = ntt_thread_index;
                    int t = 1 << t_;
                    int in_shared_address = ((shared_address >> t_) << t_) + shared_address;
                    
                    // 确定循环次数
                    int loop = ((log_row - 6) <= 0) ? log_row : 6;
                    
                    // 第一个循环
                    for (int lp = 0; lp < (log_row - 6); lp++)
                    {
                        int group_in_shared_address = in_shared_address + offset;
                        
                        location_t current_root_index;
                        if (reduction_poly_check)
                        { // X_N_minus
                            current_root_index = (omega_address >> t_);
                        }
                        else
                        { // X_N_plus
                            current_root_index = m + (omega_address >> t_);
                        }
                        
                        Root<TU> root;
                        root = root_shared_memory[current_root_index];
                        
                        CooleyTukeyUnit(
                            shared_memory[group_in_shared_address],
                            shared_memory[group_in_shared_address + t],
                            root,
                            modulus_thread);
                        
                        // 更新变量
                        t = t >> 1;
                        t_ -= 1;
                        m <<= 1;
                        
                        in_shared_address = ((shared_address >> t_) << t_) + shared_address;
                    }
                    
                    // 第二个循环
                    for (int lp = 0; lp < loop; lp++)
                    {
                        int group_in_shared_address = in_shared_address + offset;
                        
                        location_t current_root_index;
                        if (reduction_poly_check)
                        { // X_N_minus
                            current_root_index = (omega_address >> t_) + (mod_index << log_row);
                        }
                        else
                        { // X_N_plus
                            current_root_index = m + (omega_address >> t_) + (mod_index << log_row);
                        }
                        
                        Root<TU> root;
                        root = root_shared_memory[current_root_index];
                        
                        CooleyTukeyUnit(
                            shared_memory[group_in_shared_address],
                            shared_memory[group_in_shared_address + t],
                            root,
                            modulus_thread);
                        
                        // 更新变量
                        t = t >> 1;
                        t_ -= 1;
                        m <<= 1;
                        
                        in_shared_address = ((shared_address >> t_) << t_) + shared_address;
                    }
                    
                    // 将结果写回输出
                    polynomial_out[global_address] = shared_memory[transpose_block];
                    polynomial_out[global_address + global_offset] = 
                        shared_memory[transpose_block + blockdim_y];
                }
            }
        }
    }
//     template <typename T>
//     __global__ void InverseCoreTranspose(
//         typename std::make_unsigned<T>::type* polynomial_in, T* polynomial_out,
//         const Root<typename std::make_unsigned<
//             T>::type>* __restrict__ inverse_root_of_unity_table,
//         Modulus<typename std::make_unsigned<T>::type> modulus,
//         Ninverse<typename std::make_unsigned<T>::type> n_inverse, int log_row,
//         int log_column, bool reduction_poly_check)
//     {
//         using TU = typename std::make_unsigned<T>::type;

//         const int idx_x = threadIdx.x;
//         const int idx_y = threadIdx.y;
//         const int block_x = blockIdx.x;

//         extern __shared__ char shared_memory_typed[];
//         TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);
//         TU* root_shared_memory = shared_memory + (1024);

//         if (idx_x == 0)
//         {
//             if (reduction_poly_check)
//             { // X_N_minus
//                 root_shared_memory[idx_y] = inverse_root_of_unity_table[idx_y];
//             }
//             else
//             { // X_N_plus
//                 root_shared_memory[idx_y] = inverse_root_of_unity_table[idx_y];
//                 root_shared_memory[idx_y + blockDim.y] =
//                     inverse_root_of_unity_table[idx_y + blockDim.y];
//             }
//         }

//         const Modulus<TU> modulus_thread = modulus;
//         const Ninverse<TU> n_inverse_thread = n_inverse;

//         location_t global_addresss =
//             (idx_y << log_column) + (blockDim.x * block_x) + idx_x;
//         location_t global_offset = (blockDim.y << log_column);

//         int t_ = 0;
//         int loops = log_row;
//         int m = (int) 1 << (log_row - 1);

//         // Load T from global & store to shared
//         const int transpose_block = idx_y + (idx_x * (2 * blockDim.y));
//         shared_memory[transpose_block] = polynomial_in[global_addresss];
//         shared_memory[transpose_block + blockDim.y] =
//             polynomial_in[global_addresss + global_offset];

//         int log_row_r = log_row - 1;
//         const int block_thread_index = idx_x + (idx_y * blockDim.x);
//         const int ntt_thread_index =
//             block_thread_index & ((1 << log_row_r) - 1);
//         const int ntt_block_index = block_thread_index >> log_row_r;
//         int offset = ntt_block_index << log_row;

//         int omega_addresss = ntt_thread_index;
//         int shared_addresss = ntt_thread_index;
//         int t = 1 << t_;
//         int in_shared_address =
//             ((shared_addresss >> t_) << t_) + shared_addresss;
//         location_t current_root_index;
//         __syncthreads();

// #pragma unroll
//         for (int lp = 0; lp < loops; lp++)
//         {
//             int group_in_shared_address = in_shared_address + offset;

//             if (reduction_poly_check)
//             { // X_N_minus
//                 current_root_index = (omega_addresss >> t_);
//             }
//             else
//             { // X_N_plus
//                 current_root_index = m + (omega_addresss >> t_);
//             }

//             Root<TU> root = root_shared_memory[current_root_index];

//             GentlemanSandeUnit(shared_memory[group_in_shared_address],
//                                shared_memory[group_in_shared_address + t], root,
//                                modulus_thread);

//             t = t << 1;
//             t_ += 1;
//             m >>= 1;

//             in_shared_address =
//                 ((shared_addresss >> t_) << t_) + shared_addresss;
//             __syncthreads();
//         }

//         TU output1_reg = OPERATOR_GPU<TU>::mult(
//             shared_memory[transpose_block], n_inverse_thread, modulus_thread);
//         TU output2_reg =
//             OPERATOR_GPU<TU>::mult(shared_memory[transpose_block + blockDim.y],
//                                    n_inverse_thread, modulus_thread);

//         if constexpr (std::is_signed<T>::value)
//         {
//             polynomial_out[global_addresss] =
//                 OPERATOR_GPU<TU>::centered_reduction(output1_reg,
//                                                      modulus_thread);
//             polynomial_out[global_addresss + global_offset] =
//                 OPERATOR_GPU<TU>::centered_reduction(output2_reg,
//                                                      modulus_thread);
//         }
//         else
//         {
//             polynomial_out[global_addresss] = output1_reg;
//             polynomial_out[global_addresss + global_offset] = output2_reg;
//         }
//     }

//     template <typename T>
//     __global__ void InverseCoreTranspose(
//         typename std::make_unsigned<T>::type* polynomial_in, T* polynomial_out,
//         const Root<typename std::make_unsigned<
//             T>::type>* __restrict__ inverse_root_of_unity_table,
//         Modulus<typename std::make_unsigned<T>::type>* modulus,
//         Ninverse<typename std::make_unsigned<T>::type>* n_inverse, int log_row,
//         int log_column, bool reduction_poly_check, int mod_count)
//     {
//         using TU = typename std::make_unsigned<T>::type;

//         const int idx_x = threadIdx.x;
//         const int idx_y = threadIdx.y;
//         const int block_x = blockIdx.x;
//         const int total_block_thread = blockDim.x * blockDim.y;

//         extern __shared__ char shared_memory_typed[];
//         TU* shared_memory = reinterpret_cast<TU*>(shared_memory_typed);
//         TU* root_shared_memory = shared_memory + (1024);

//         if (idx_x == 0)
//         {
//             if (reduction_poly_check)
//             { // X_N_minus
//                 root_shared_memory[idx_y] = inverse_root_of_unity_table[idx_y];
//             }
//             else
//             { // X_N_plus
//                 root_shared_memory[idx_y] = inverse_root_of_unity_table[idx_y];
//                 root_shared_memory[idx_y + blockDim.y] =
//                     inverse_root_of_unity_table[idx_y + blockDim.y];
//             }
//         }

//         location_t global_addresss =
//             (idx_y << log_column) + (blockDim.x * block_x) + idx_x;
//         location_t global_offset = (blockDim.y << log_column);

//         int t_ = 0;
//         int loops = log_row;
//         int m = (int) 1 << (log_row - 1);

//         // Load T from global & store to shared
//         const int transpose_block = idx_y + (idx_x * (2 * blockDim.y));
//         shared_memory[transpose_block] = polynomial_in[global_addresss];
//         shared_memory[transpose_block + blockDim.y] =
//             polynomial_in[global_addresss + global_offset];

//         int log_row_r = log_row - 1;
//         const int block_thread_index = idx_x + (idx_y * blockDim.x);
//         const int ntt_thread_index =
//             block_thread_index & ((1 << log_row_r) - 1);
//         const int ntt_block_index = block_thread_index >> log_row_r;
//         int offset = ntt_block_index << log_row;

//         const int batch_index =
//             (block_x * (total_block_thread >> t_)) + ntt_block_index;
//         int mod_index = batch_index % mod_count;

//         const Modulus<TU> modulus_thread = modulus[mod_index];
//         const Ninverse<TU> n_inverse_thread = n_inverse[mod_index];

//         int omega_addresss = ntt_thread_index;
//         int shared_addresss = ntt_thread_index;
//         int t = 1 << t_;
//         int in_shared_address =
//             ((shared_addresss >> t_) << t_) + shared_addresss;
//         location_t current_root_index;
//         __syncthreads();

// #pragma unroll
//         for (int lp = 0; lp < loops; lp++)
//         {
//             int group_in_shared_address = in_shared_address + offset;

//             if (reduction_poly_check)
//             { // X_N_minus
//                 current_root_index =
//                     (omega_addresss >> t_) + (mod_index << log_row);
//             }
//             else
//             { // X_N_plus
//                 current_root_index =
//                     m + (omega_addresss >> t_) + (mod_index << log_row);
//             }

//             Root<TU> root = root_shared_memory[current_root_index];

//             GentlemanSandeUnit(shared_memory[group_in_shared_address],
//                                shared_memory[group_in_shared_address + t], root,
//                                modulus_thread);

//             t = t << 1;
//             t_ += 1;
//             m >>= 1;

//             in_shared_address =
//                 ((shared_addresss >> t_) << t_) + shared_addresss;
//             __syncthreads();
//         }

//         TU output1_reg = OPERATOR_GPU<TU>::mult(
//             shared_memory[transpose_block], n_inverse_thread, modulus_thread);
//         TU output2_reg =
//             OPERATOR_GPU<TU>::mult(shared_memory[transpose_block + blockDim.y],
//                                    n_inverse_thread, modulus_thread);

//         if constexpr (std::is_signed<T>::value)
//         {
//             polynomial_out[global_addresss] =
//                 OPERATOR_GPU<TU>::centered_reduction(output1_reg,
//                                                      modulus_thread);
//             polynomial_out[global_addresss + global_offset] =
//                 OPERATOR_GPU<TU>::centered_reduction(output2_reg,
//                                                      modulus_thread);
//         }
//         else
//         {
//             polynomial_out[global_addresss] = output1_reg;
//             polynomial_out[global_addresss + global_offset] = output2_reg;
//         }
//     }

//     template <typename T>
//     __host__ void
//     GPU_NTT(T* device_in, typename std::make_unsigned<T>::type* device_out,
//             Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
//             Modulus<typename std::make_unsigned<T>::type> modulus,
//             ntt_configuration<typename std::make_unsigned<T>::type> cfg,
//             int batch_size)
//     {
//         switch (cfg.ntt_layout)
//         {
//             case PerPolynomial:
//             {
//                 if ((cfg.n_power <= 0 || cfg.n_power >= 29))
//                 {
//                     throw std::invalid_argument("Invalid n_power range!");
//                 }

//                 auto kernel_parameters = CreateForwardNTTKernel<
//                     typename std::make_unsigned<T>::type>();
//                 bool low_ring_size = (cfg.n_power < 10) ? true : false;
//                 bool standart_kernel = (cfg.n_power < 25) ? true : false;

//                 if (low_ring_size)
//                 {
//                     auto& current_kernel_params =
//                         kernel_parameters[cfg.n_power][0];
//                     ForwardCoreLowRing<<<
//                         dim3((batch_size +
//                               (current_kernel_params.blockdim_y - 1)) /
//                                  current_kernel_params.blockdim_y,
//                              1, 1),
//                         dim3(current_kernel_params.blockdim_x,
//                              current_kernel_params.blockdim_y),
//                         current_kernel_params.shared_memory, cfg.stream>>>(
//                         device_in, device_out, root_of_unity_table, modulus,
//                         current_kernel_params.shared_index, cfg.n_power,
//                         cfg.zero_padding,
//                         (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
//                         batch_size);
//                     GPUNTT_CUDA_CHECK(cudaGetLastError());
//                 }
//                 else
//                 {
//                     if (standart_kernel)
//                     {
//                         auto& current_kernel_params =
//                             kernel_parameters[cfg.n_power][0];
//                         ForwardCore<<<
//                             dim3(current_kernel_params.griddim_x,
//                                  current_kernel_params.griddim_y, batch_size),
//                             dim3(current_kernel_params.blockdim_x,
//                                  current_kernel_params.blockdim_y),
//                             current_kernel_params.shared_memory, cfg.stream>>>(
//                             device_in, device_out, root_of_unity_table, modulus,
//                             current_kernel_params.shared_index,
//                             current_kernel_params.logm,
//                             current_kernel_params.outer_iteration_count,
//                             cfg.n_power, cfg.zero_padding,
//                             current_kernel_params.not_last_kernel,
//                             (cfg.reduction_poly ==
//                              ReductionPolynomial::X_N_minus));
//                         GPUNTT_CUDA_CHECK(cudaGetLastError());

//                         for (int i = 1;
//                              i < kernel_parameters[cfg.n_power].size(); i++)
//                         {
//                             auto& current_kernel_params =
//                                 kernel_parameters[cfg.n_power][i];
//                             ForwardCore<<<dim3(current_kernel_params.griddim_x,
//                                                current_kernel_params.griddim_y,
//                                                batch_size),
//                                           dim3(
//                                               current_kernel_params.blockdim_x,
//                                               current_kernel_params.blockdim_y),
//                                           current_kernel_params.shared_memory,
//                                           cfg.stream>>>(
//                                 device_out, device_out, root_of_unity_table,
//                                 modulus, current_kernel_params.shared_index,
//                                 current_kernel_params.logm,
//                                 current_kernel_params.outer_iteration_count,
//                                 cfg.n_power, cfg.zero_padding,
//                                 current_kernel_params.not_last_kernel,
//                                 (cfg.reduction_poly ==
//                                  ReductionPolynomial::X_N_minus));
//                             GPUNTT_CUDA_CHECK(cudaGetLastError());
//                         }
//                     }
//                     else
//                     {
//                         auto& current_kernel_params =
//                             kernel_parameters[cfg.n_power][0];
//                         ForwardCore<<<
//                             dim3(current_kernel_params.griddim_x,
//                                  current_kernel_params.griddim_y, batch_size),
//                             dim3(current_kernel_params.blockdim_x,
//                                  current_kernel_params.blockdim_y),
//                             current_kernel_params.shared_memory, cfg.stream>>>(
//                             device_in, device_out, root_of_unity_table, modulus,
//                             current_kernel_params.shared_index,
//                             current_kernel_params.logm,
//                             current_kernel_params.outer_iteration_count,
//                             cfg.n_power, cfg.zero_padding,
//                             current_kernel_params.not_last_kernel,
//                             (cfg.reduction_poly ==
//                              ReductionPolynomial::X_N_minus));
//                         GPUNTT_CUDA_CHECK(cudaGetLastError());

//                         for (int i = 1;
//                              i < kernel_parameters[cfg.n_power].size() - 1; i++)
//                         {
//                             auto& current_kernel_params =
//                                 kernel_parameters[cfg.n_power][i];
//                             ForwardCore<<<dim3(current_kernel_params.griddim_x,
//                                                current_kernel_params.griddim_y,
//                                                batch_size),
//                                           dim3(
//                                               current_kernel_params.blockdim_x,
//                                               current_kernel_params.blockdim_y),
//                                           current_kernel_params.shared_memory,
//                                           cfg.stream>>>(
//                                 device_out, device_out, root_of_unity_table,
//                                 modulus, current_kernel_params.shared_index,
//                                 current_kernel_params.logm,
//                                 current_kernel_params.outer_iteration_count,
//                                 cfg.n_power, cfg.zero_padding,
//                                 current_kernel_params.not_last_kernel,
//                                 (cfg.reduction_poly ==
//                                  ReductionPolynomial::X_N_minus));
//                             GPUNTT_CUDA_CHECK(cudaGetLastError());
//                         }
//                         current_kernel_params = kernel_parameters
//                             [cfg.n_power]
//                             [kernel_parameters[cfg.n_power].size() - 1];
//                         ForwardCore_<<<
//                             dim3(current_kernel_params.griddim_x,
//                                  current_kernel_params.griddim_y, batch_size),
//                             dim3(current_kernel_params.blockdim_x,
//                                  current_kernel_params.blockdim_y),
//                             current_kernel_params.shared_memory, cfg.stream>>>(
//                             device_out, device_out, root_of_unity_table,
//                             modulus, current_kernel_params.shared_index,
//                             current_kernel_params.logm,
//                             current_kernel_params.outer_iteration_count,
//                             cfg.n_power, cfg.zero_padding,
//                             current_kernel_params.not_last_kernel,
//                             (cfg.reduction_poly ==
//                              ReductionPolynomial::X_N_minus));
//                         GPUNTT_CUDA_CHECK(cudaGetLastError());
//                     }
//                 }
//             }
//             break;
//             case PerCoefficient:
//             {
//                 if ((cfg.n_power <= 0 || cfg.n_power >= 10))
//                 {
//                     throw std::invalid_argument("Invalid n_power range!");
//                 }

//                 int log_batch_size = log2(batch_size);
//                 int total_size = 1 << (cfg.n_power + log_batch_size);
//                 int total_block_thread = 512;
//                 int total_block_count = total_size / (total_block_thread * 2);
//                 int blockdim_y = 1 << (cfg.n_power - 1);
//                 int blockdim_x = total_block_thread / blockdim_y;
//                 ForwardCoreTranspose<<<
//                     dim3(total_block_count, 1, 1),
//                     dim3(blockdim_x, blockdim_y, 1),
//                     ((total_block_thread * 2) + (1 << cfg.n_power)) * sizeof(T),
//                     cfg.stream>>>(
//                     device_in, device_out, root_of_unity_table, modulus,
//                     cfg.n_power, log_batch_size,
//                     (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
//                 GPUNTT_CUDA_CHECK(cudaGetLastError());
//             }
//             break;
//             default:
//                 throw std::invalid_argument("Invalid ntt_layout!");
//                 break;
//         }
//     }

//     template <typename T>
//     __host__ void
//     GPU_INTT(typename std::make_unsigned<T>::type* device_in, T* device_out,
//              Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
//              Modulus<typename std::make_unsigned<T>::type> modulus,
//              ntt_configuration<typename std::make_unsigned<T>::type> cfg,
//              int batch_size)
//     {
//         switch (cfg.ntt_layout)
//         {
//             case PerPolynomial:
//             {
//                 if ((cfg.n_power <= 0 || cfg.n_power >= 29))
//                 {
//                     throw std::invalid_argument("Invalid n_power range!");
//                 }

//                 auto kernel_parameters = CreateInverseNTTKernel<
//                     typename std::make_unsigned<T>::type>();
//                 bool low_ring_size = (cfg.n_power < 11) ? true : false;
//                 bool standart_kernel = (cfg.n_power < 25) ? true : false;

//                 if (low_ring_size)
//                 {
//                     auto& current_kernel_params =
//                         kernel_parameters[cfg.n_power][0];
//                     InverseCoreLowRing<<<
//                         dim3((batch_size +
//                               (current_kernel_params.blockdim_y - 1)) /
//                                  current_kernel_params.blockdim_y,
//                              1, 1),
//                         dim3(current_kernel_params.blockdim_x,
//                              current_kernel_params.blockdim_y),
//                         current_kernel_params.shared_memory, cfg.stream>>>(
//                         device_in, device_out, root_of_unity_table, modulus,
//                         current_kernel_params.shared_index, cfg.n_power,
//                         cfg.mod_inverse,
//                         (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
//                         batch_size);
//                     GPUNTT_CUDA_CHECK(cudaGetLastError());
//                 }
//                 else
//                 {
//                     if (standart_kernel)
//                     {
//                         if constexpr (std::is_signed<T>::value)
//                         {
//                             typename std::make_unsigned<T>::type* device_in_ =
//                                 device_in;
//                             for (int i = 0;
//                                  i < kernel_parameters[cfg.n_power].size() - 1;
//                                  i++)
//                             {
//                                 auto& current_kernel_params =
//                                     kernel_parameters[cfg.n_power][i];
//                                 InverseCore<<<
//                                     dim3(current_kernel_params.griddim_x,
//                                          current_kernel_params.griddim_y,
//                                          batch_size),
//                                     dim3(current_kernel_params.blockdim_x,
//                                          current_kernel_params.blockdim_y),
//                                     current_kernel_params.shared_memory,
//                                     cfg.stream>>>(
//                                     device_in_,
//                                     reinterpret_cast<
//                                         typename std::make_unsigned<T>::type*>(
//                                         device_out),
//                                     root_of_unity_table, modulus,
//                                     current_kernel_params.shared_index,
//                                     current_kernel_params.logm,
//                                     current_kernel_params.k,
//                                     current_kernel_params.outer_iteration_count,
//                                     cfg.n_power, cfg.mod_inverse,
//                                     current_kernel_params.not_last_kernel,
//                                     (cfg.reduction_poly ==
//                                      ReductionPolynomial::X_N_minus));
//                                 GPUNTT_CUDA_CHECK(cudaGetLastError());
//                                 device_in_ = reinterpret_cast<
//                                     typename std::make_unsigned<T>::type*>(
//                                     device_out);
//                             }

//                             auto& current_kernel_params = kernel_parameters
//                                 [cfg.n_power]
//                                 [kernel_parameters[cfg.n_power].size() - 1];
//                             InverseCore<<<dim3(current_kernel_params.griddim_x,
//                                                current_kernel_params.griddim_y,
//                                                batch_size),
//                                           dim3(
//                                               current_kernel_params.blockdim_x,
//                                               current_kernel_params.blockdim_y),
//                                           current_kernel_params.shared_memory,
//                                           cfg.stream>>>(
//                                 reinterpret_cast<
//                                     typename std::make_unsigned<T>::type*>(
//                                     device_out),
//                                 device_out, root_of_unity_table, modulus,
//                                 current_kernel_params.shared_index,
//                                 current_kernel_params.logm,
//                                 current_kernel_params.k,
//                                 current_kernel_params.outer_iteration_count,
//                                 cfg.n_power, cfg.mod_inverse,
//                                 current_kernel_params.not_last_kernel,
//                                 (cfg.reduction_poly ==
//                                  ReductionPolynomial::X_N_minus));
//                             GPUNTT_CUDA_CHECK(cudaGetLastError());
//                         }
//                         else
//                         {
//                             for (int i = 0;
//                                  i < kernel_parameters[cfg.n_power].size(); i++)
//                             {
//                                 auto& current_kernel_params =
//                                     kernel_parameters[cfg.n_power][i];
//                                 InverseCore<<<
//                                     dim3(current_kernel_params.griddim_x,
//                                          current_kernel_params.griddim_y,
//                                          batch_size),
//                                     dim3(current_kernel_params.blockdim_x,
//                                          current_kernel_params.blockdim_y),
//                                     current_kernel_params.shared_memory,
//                                     cfg.stream>>>(
//                                     device_in, device_out, root_of_unity_table,
//                                     modulus, current_kernel_params.shared_index,
//                                     current_kernel_params.logm,
//                                     current_kernel_params.k,
//                                     current_kernel_params.outer_iteration_count,
//                                     cfg.n_power, cfg.mod_inverse,
//                                     current_kernel_params.not_last_kernel,
//                                     (cfg.reduction_poly ==
//                                      ReductionPolynomial::X_N_minus));
//                                 GPUNTT_CUDA_CHECK(cudaGetLastError());
//                             }
//                         }
//                     }
//                     else
//                     {
//                         if constexpr (std::is_signed<T>::value)
//                         {
//                             auto& current_kernel_params =
//                                 kernel_parameters[cfg.n_power][0];
//                             InverseCore_<<<
//                                 dim3(current_kernel_params.griddim_x,
//                                      current_kernel_params.griddim_y,
//                                      batch_size),
//                                 dim3(current_kernel_params.blockdim_x,
//                                      current_kernel_params.blockdim_y),
//                                 current_kernel_params.shared_memory,
//                                 cfg.stream>>>(
//                                 device_in,
//                                 reinterpret_cast<
//                                     typename std::make_unsigned<T>::type*>(
//                                     device_out),
//                                 root_of_unity_table, modulus,
//                                 current_kernel_params.shared_index,
//                                 current_kernel_params.logm,
//                                 current_kernel_params.k,
//                                 current_kernel_params.outer_iteration_count,
//                                 cfg.n_power, cfg.mod_inverse,
//                                 current_kernel_params.not_last_kernel,
//                                 (cfg.reduction_poly ==
//                                  ReductionPolynomial::X_N_minus));
//                             GPUNTT_CUDA_CHECK(cudaGetLastError());
//                             for (int i = 1;
//                                  i < kernel_parameters[cfg.n_power].size() - 1;
//                                  i++)
//                             {
//                                 auto& current_kernel_params =
//                                     kernel_parameters[cfg.n_power][i];
//                                 InverseCore<<<
//                                     dim3(current_kernel_params.griddim_x,
//                                          current_kernel_params.griddim_y,
//                                          batch_size),
//                                     dim3(current_kernel_params.blockdim_x,
//                                          current_kernel_params.blockdim_y),
//                                     current_kernel_params.shared_memory,
//                                     cfg.stream>>>(
//                                     reinterpret_cast<
//                                         typename std::make_unsigned<T>::type*>(
//                                         device_out),
//                                     reinterpret_cast<
//                                         typename std::make_unsigned<T>::type*>(
//                                         device_out),
//                                     root_of_unity_table, modulus,
//                                     current_kernel_params.shared_index,
//                                     current_kernel_params.logm,
//                                     current_kernel_params.k,
//                                     current_kernel_params.outer_iteration_count,
//                                     cfg.n_power, cfg.mod_inverse,
//                                     current_kernel_params.not_last_kernel,
//                                     (cfg.reduction_poly ==
//                                      ReductionPolynomial::X_N_minus));
//                                 GPUNTT_CUDA_CHECK(cudaGetLastError());
//                             }

//                             current_kernel_params = kernel_parameters
//                                 [cfg.n_power]
//                                 [kernel_parameters[cfg.n_power].size() - 1];
//                             InverseCore<<<dim3(current_kernel_params.griddim_x,
//                                                current_kernel_params.griddim_y,
//                                                batch_size),
//                                           dim3(
//                                               current_kernel_params.blockdim_x,
//                                               current_kernel_params.blockdim_y),
//                                           current_kernel_params.shared_memory,
//                                           cfg.stream>>>(
//                                 reinterpret_cast<
//                                     typename std::make_unsigned<T>::type*>(
//                                     device_out),
//                                 device_out, root_of_unity_table, modulus,
//                                 current_kernel_params.shared_index,
//                                 current_kernel_params.logm,
//                                 current_kernel_params.k,
//                                 current_kernel_params.outer_iteration_count,
//                                 cfg.n_power, cfg.mod_inverse,
//                                 current_kernel_params.not_last_kernel,
//                                 (cfg.reduction_poly ==
//                                  ReductionPolynomial::X_N_minus));
//                             GPUNTT_CUDA_CHECK(cudaGetLastError());
//                         }
//                         else
//                         {
//                             auto& current_kernel_params =
//                                 kernel_parameters[cfg.n_power][0];
//                             InverseCore_<<<
//                                 dim3(current_kernel_params.griddim_x,
//                                      current_kernel_params.griddim_y,
//                                      batch_size),
//                                 dim3(current_kernel_params.blockdim_x,
//                                      current_kernel_params.blockdim_y),
//                                 current_kernel_params.shared_memory,
//                                 cfg.stream>>>(
//                                 device_in, device_out, root_of_unity_table,
//                                 modulus, current_kernel_params.shared_index,
//                                 current_kernel_params.logm,
//                                 current_kernel_params.k,
//                                 current_kernel_params.outer_iteration_count,
//                                 cfg.n_power, cfg.mod_inverse,
//                                 current_kernel_params.not_last_kernel,
//                                 (cfg.reduction_poly ==
//                                  ReductionPolynomial::X_N_minus));
//                             GPUNTT_CUDA_CHECK(cudaGetLastError());

//                             for (int i = 1;
//                                  i < kernel_parameters[cfg.n_power].size(); i++)
//                             {
//                                 auto& current_kernel_params =
//                                     kernel_parameters[cfg.n_power][i];
//                                 InverseCore<<<
//                                     dim3(current_kernel_params.griddim_x,
//                                          current_kernel_params.griddim_y,
//                                          batch_size),
//                                     dim3(current_kernel_params.blockdim_x,
//                                          current_kernel_params.blockdim_y),
//                                     current_kernel_params.shared_memory,
//                                     cfg.stream>>>(
//                                     device_out, device_out, root_of_unity_table,
//                                     modulus, current_kernel_params.shared_index,
//                                     current_kernel_params.logm,
//                                     current_kernel_params.k,
//                                     current_kernel_params.outer_iteration_count,
//                                     cfg.n_power, cfg.mod_inverse,
//                                     current_kernel_params.not_last_kernel,
//                                     (cfg.reduction_poly ==
//                                      ReductionPolynomial::X_N_minus));
//                                 GPUNTT_CUDA_CHECK(cudaGetLastError());
//                             }
//                         }
//                     }
//                 }
//             }
//             break;
//             case PerCoefficient:
//             {
//                 if ((cfg.n_power <= 0 || cfg.n_power >= 10))
//                 {
//                     throw std::invalid_argument("Invalid n_power range!");
//                 }

//                 int log_batch_size = log2(batch_size);
//                 int total_size = 1 << (cfg.n_power + log_batch_size);
//                 int total_block_thread = 512;
//                 int total_block_count = total_size / (total_block_thread * 2);
//                 int blockdim_y = 1 << (cfg.n_power - 1);
//                 int blockdim_x = total_block_thread / blockdim_y;
//                 InverseCoreTranspose<<<
//                     dim3(total_block_count, 1, 1),
//                     dim3(blockdim_x, blockdim_y, 1),
//                     ((total_block_thread * 2) + (1 << cfg.n_power)) * sizeof(T),
//                     cfg.stream>>>(
//                     device_in, device_out, root_of_unity_table, modulus,
//                     cfg.mod_inverse, cfg.n_power, log_batch_size,
//                     (cfg.reduction_poly == ReductionPolynomial::X_N_minus));
//                 GPUNTT_CUDA_CHECK(cudaGetLastError());
//             }
//             break;
//             default:
//                 throw std::invalid_argument("Invalid ntt_layout!");
//                 break;
//         }
//     }

    template <typename T>
    void GPU_NTT(T* device_in, T* device_out, Root<T>* root_of_unity_table,
                        Modulus<T>* modulus, ntt_rns_configuration<T> cfg,
                        int batch_size, int mod_count)
    {
        if ((cfg.n_power <= 11 || cfg.n_power >= 29))
        {
            throw std::invalid_argument("Invalid n_power range!");
        }

        auto kernel_parameters = (cfg.ntt_type == FORWARD)
                                    ? CreateForwardNTTKernel<T>()
                                    : CreateInverseNTTKernel<T>();
        bool standart_kernel = (cfg.n_power < 25) ? true : false;
        T* device_in_ = device_in;

        switch (cfg.ntt_type)
        {
            case FORWARD: //here
                if (standart_kernel)
                {
                    for (int i = 0; i < kernel_parameters[cfg.n_power].size(); i++)
                    {
                        auto& current_kernel_params =
                            kernel_parameters[cfg.n_power][i];
                        // ForwardCore<<<
                        //     dim3(current_kernel_params.griddim_x,
                        //         current_kernel_params.griddim_y, batch_size),
                        //     dim3(current_kernel_params.blockdim_x,
                        //         current_kernel_params.blockdim_y),
                        //     current_kernel_params.shared_memory, cfg.stream>>>(
                        //     device_in_, device_out, root_of_unity_table, modulus,
                        //     current_kernel_params.shared_index,
                        //     current_kernel_params.logm,
                        //     current_kernel_params.outer_iteration_count,
                        //     cfg.n_power, cfg.zero_padding,
                        //     current_kernel_params.not_last_kernel,
                        //     (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        //     mod_count);
                        // THROW_IF_CUDA_ERROR(cudaGetLastError());
                        try {
                            ForwardCoreCPU<T>(
                                device_in_, device_out,
                                root_of_unity_table , modulus ,
                                current_kernel_params.shared_index, current_kernel_params.logm, 
                                current_kernel_params.outer_iteration_count, cfg.n_power,
                                cfg.zero_padding, current_kernel_params.not_last_kernel, 
                                (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count,
                                current_kernel_params.griddim_x, current_kernel_params.griddim_y, batch_size, 
                                current_kernel_params.blockdim_x, current_kernel_params.blockdim_y,
                                current_kernel_params.shared_memory);
                            
                            std::cout << "ForwardCore completed successfully" << std::endl;
                        } catch (const std::exception& e) {
                            std::cerr << "Error in ForwardCore: " << e.what() << std::endl; 
                            std::exit(EXIT_FAILURE);
                        }
                        device_in_ = device_out;
                    }
                }
                else
                {
                    for (int i = 0; i < kernel_parameters[cfg.n_power].size() - 1;
                        i++)
                    {
                        auto& current_kernel_params =
                            kernel_parameters[cfg.n_power][i];
                        // ForwardCore<<<
                        //     dim3(current_kernel_params.griddim_x,
                        //         current_kernel_params.griddim_y, batch_size),
                        //     dim3(current_kernel_params.blockdim_x,
                        //         current_kernel_params.blockdim_y),
                        //     current_kernel_params.shared_memory, cfg.stream>>>(
                        //     device_in_, device_out, root_of_unity_table, modulus,
                        //     current_kernel_params.shared_index,
                        //     current_kernel_params.logm,
                        //     current_kernel_params.outer_iteration_count,
                        //     cfg.n_power, cfg.zero_padding,
                        //     current_kernel_params.not_last_kernel,
                        //     (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        //     mod_count);
                        // THROW_IF_CUDA_ERROR(cudaGetLastError());
                        try {
                            ForwardCoreCPU<T>(
                                device_in_, device_out,
                                root_of_unity_table , modulus ,
                                current_kernel_params.shared_index, current_kernel_params.logm, 
                                current_kernel_params.outer_iteration_count, cfg.n_power,
                                cfg.zero_padding, current_kernel_params.not_last_kernel, 
                                (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count,
                                current_kernel_params.griddim_x, current_kernel_params.griddim_y, batch_size, 
                                current_kernel_params.blockdim_x, current_kernel_params.blockdim_y,
                                current_kernel_params.shared_memory);
                            
                            std::cout << "ForwardCore completed successfully" << std::endl;
                        } catch (const std::exception& e) {
                            std::cerr << "Error in ForwardCore: " << e.what() << std::endl; 
                            std::exit(EXIT_FAILURE);
                        }
                        device_in_ = device_out;
                    }
                    auto& current_kernel_params =
                        kernel_parameters[cfg.n_power]
                                        [kernel_parameters[cfg.n_power].size() -
                                        1];
                    // ForwardCore_<<<
                    //     dim3(current_kernel_params.griddim_x,
                    //         current_kernel_params.griddim_y, batch_size),
                    //     dim3(current_kernel_params.blockdim_x,
                    //         current_kernel_params.blockdim_y),
                    //     current_kernel_params.shared_memory, cfg.stream>>>(
                    //     device_in_, device_out, root_of_unity_table, modulus,
                    //     current_kernel_params.shared_index,
                    //     current_kernel_params.logm,
                    //     current_kernel_params.outer_iteration_count, cfg.n_power,
                    //     cfg.zero_padding, current_kernel_params.not_last_kernel,
                    //     (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                    //     mod_count);
                    // THROW_IF_CUDA_ERROR(cudaGetLastError());
                    ForwardCore_CPU(
                        device_in_, device_out,
                        root_of_unity_table , modulus ,
                        current_kernel_params.shared_index, current_kernel_params.logm, 
                        current_kernel_params.outer_iteration_count, cfg.n_power,
                        cfg.zero_padding, current_kernel_params.not_last_kernel, 
                        (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count,
                        current_kernel_params.griddim_x, current_kernel_params.griddim_y, batch_size,
                        current_kernel_params.blockdim_x, current_kernel_params.blockdim_y
                    );
                }
                break;
            case INVERSE:
                if (standart_kernel)
                {
                    for (int i = 0; i < kernel_parameters[cfg.n_power].size(); i++)
                    {
                        auto& current_kernel_params =
                            kernel_parameters[cfg.n_power][i];
                        // InverseCore<<<
                        //     dim3(current_kernel_params.griddim_x,
                        //         current_kernel_params.griddim_y, batch_size),
                        //     dim3(current_kernel_params.blockdim_x,
                        //         current_kernel_params.blockdim_y),
                        //     current_kernel_params.shared_memory, cfg.stream>>>(
                        //     device_in_, device_out, root_of_unity_table, modulus,
                        //     current_kernel_params.shared_index,
                        //     current_kernel_params.logm, current_kernel_params.k,
                        //     current_kernel_params.outer_iteration_count,
                        //     cfg.n_power, cfg.mod_inverse,
                        //     current_kernel_params.not_last_kernel,
                        //     (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        //     mod_count);
                        // THROW_IF_CUDA_ERROR(cudaGetLastError());
                        InverseCoreCPU(
                            device_in_, device_out,
                            root_of_unity_table, modulus,
                            current_kernel_params.shared_index, current_kernel_params.logm, 
                            current_kernel_params.k, current_kernel_params.outer_iteration_count, 
                            cfg.n_power,cfg.mod_inverse, current_kernel_params.not_last_kernel, 
                            (cfg.reduction_poly == ReductionPolynomial::X_N_minus), mod_count,
                            current_kernel_params.griddim_x, current_kernel_params.griddim_y, batch_size,
                            current_kernel_params.blockdim_x, current_kernel_params.blockdim_y,
                            current_kernel_params.shared_memory
                        ); 
                        device_in_ = device_out;
                    }
                }
                else
                {
                    auto& current_kernel_params = kernel_parameters[cfg.n_power][0];
                    // InverseCore_<<<
                    //     dim3(current_kernel_params.griddim_x,
                    //         current_kernel_params.griddim_y, batch_size),
                    //     dim3(current_kernel_params.blockdim_x,
                    //         current_kernel_params.blockdim_y),
                    //     current_kernel_params.shared_memory, cfg.stream>>>(
                    //     device_in_, device_out, root_of_unity_table, modulus,
                    //     current_kernel_params.shared_index,
                    //     current_kernel_params.logm, current_kernel_params.k,
                    //     current_kernel_params.outer_iteration_count, cfg.n_power,
                    //     cfg.mod_inverse, current_kernel_params.not_last_kernel,
                    //     (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                    //     mod_count);
                    // THROW_IF_CUDA_ERROR(cudaGetLastError());
                    
                    //invercpucore
                    device_in_ = device_out;
                    for (int i = 1; i < kernel_parameters[cfg.n_power].size(); i++)
                    {
                        auto& current_kernel_params =
                            kernel_parameters[cfg.n_power][i];
                        // InverseCore<<<
                        //     dim3(current_kernel_params.griddim_x,
                        //         current_kernel_params.griddim_y, batch_size),
                        //     dim3(current_kernel_params.blockdim_x,
                        //         current_kernel_params.blockdim_y),
                        //     current_kernel_params.shared_memory, cfg.stream>>>(
                        //     device_in_, device_out, root_of_unity_table, modulus,
                        //     current_kernel_params.shared_index,
                        //     current_kernel_params.logm, current_kernel_params.k,
                        //     current_kernel_params.outer_iteration_count,
                        //     cfg.n_power, cfg.mod_inverse,
                        //     current_kernel_params.not_last_kernel,
                        //     (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
                        //     mod_count);
                        // THROW_IF_CUDA_ERROR(cudaGetLastError());
                        //inversecorecpu
                    }
                }
                break;
            default:
                throw std::invalid_argument("Invalid ntt_type!");
                break;
        }
    }

//     template <typename T>
//     __host__ void
//     GPU_INTT(typename std::make_unsigned<T>::type* device_in, T* device_out,
//              Root<typename std::make_unsigned<T>::type>* root_of_unity_table,
//              Modulus<typename std::make_unsigned<T>::type>* modulus,
//              ntt_rns_configuration<typename std::make_unsigned<T>::type> cfg,
//              int batch_size, int mod_count)
//     {
//         switch (cfg.ntt_layout)
//         {
//             case PerPolynomial:
//             {
//                 if ((cfg.n_power <= 0 || cfg.n_power >= 29))
//                 {
//                     throw std::invalid_argument("Invalid n_power range!");
//                 }

//                 auto kernel_parameters = CreateInverseNTTKernel<
//                     typename std::make_unsigned<T>::type>();
//                 bool low_ring_size = (cfg.n_power < 11) ? true : false;
//                 bool standart_kernel = (cfg.n_power < 25) ? true : false;

//                 if (low_ring_size)
//                 {
//                     auto& current_kernel_params =
//                         kernel_parameters[cfg.n_power][0];
//                     InverseCoreLowRing<<<
//                         dim3((batch_size +
//                               (current_kernel_params.blockdim_y - 1)) /
//                                  current_kernel_params.blockdim_y,
//                              1, 1),
//                         dim3(current_kernel_params.blockdim_x,
//                              current_kernel_params.blockdim_y),
//                         current_kernel_params.shared_memory, cfg.stream>>>(
//                         device_in, device_out, root_of_unity_table, modulus,
//                         current_kernel_params.shared_index, cfg.n_power,
//                         cfg.mod_inverse,
//                         (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
//                         batch_size, mod_count);
//                     GPUNTT_CUDA_CHECK(cudaGetLastError());
//                 }
//                 else
//                 {
//                     if (standart_kernel)
//                     {
//                         if constexpr (std::is_signed<T>::value)
//                         {
//                             typename std::make_unsigned<T>::type* device_in_ =
//                                 device_in;
//                             for (int i = 0;
//                                  i < kernel_parameters[cfg.n_power].size() - 1;
//                                  i++)
//                             {
//                                 auto& current_kernel_params =
//                                     kernel_parameters[cfg.n_power][i];
//                                 InverseCore<<<
//                                     dim3(current_kernel_params.griddim_x,
//                                          current_kernel_params.griddim_y,
//                                          batch_size),
//                                     dim3(current_kernel_params.blockdim_x,
//                                          current_kernel_params.blockdim_y),
//                                     current_kernel_params.shared_memory,
//                                     cfg.stream>>>(
//                                     device_in_,
//                                     reinterpret_cast<
//                                         typename std::make_unsigned<T>::type*>(
//                                         device_out),
//                                     root_of_unity_table, modulus,
//                                     current_kernel_params.shared_index,
//                                     current_kernel_params.logm,
//                                     current_kernel_params.k,
//                                     current_kernel_params.outer_iteration_count,
//                                     cfg.n_power, cfg.mod_inverse,
//                                     current_kernel_params.not_last_kernel,
//                                     (cfg.reduction_poly ==
//                                      ReductionPolynomial::X_N_minus),
//                                     mod_count);
//                                 GPUNTT_CUDA_CHECK(cudaGetLastError());
//                                 device_in_ = reinterpret_cast<
//                                     typename std::make_unsigned<T>::type*>(
//                                     device_out);
//                             }

//                             auto& current_kernel_params = kernel_parameters
//                                 [cfg.n_power]
//                                 [kernel_parameters[cfg.n_power].size() - 1];
//                             InverseCore<<<dim3(current_kernel_params.griddim_x,
//                                                current_kernel_params.griddim_y,
//                                                batch_size),
//                                           dim3(
//                                               current_kernel_params.blockdim_x,
//                                               current_kernel_params.blockdim_y),
//                                           current_kernel_params.shared_memory,
//                                           cfg.stream>>>(
//                                 reinterpret_cast<
//                                     typename std::make_unsigned<T>::type*>(
//                                     device_out),
//                                 device_out, root_of_unity_table, modulus,
//                                 current_kernel_params.shared_index,
//                                 current_kernel_params.logm,
//                                 current_kernel_params.k,
//                                 current_kernel_params.outer_iteration_count,
//                                 cfg.n_power, cfg.mod_inverse,
//                                 current_kernel_params.not_last_kernel,
//                                 (cfg.reduction_poly ==
//                                  ReductionPolynomial::X_N_minus),
//                                 mod_count);
//                             GPUNTT_CUDA_CHECK(cudaGetLastError());
//                         }
//                         else
//                         {
//                             T* device_in_ = device_in;
//                             for (int i = 0;
//                                  i < kernel_parameters[cfg.n_power].size(); i++)
//                             {
//                                 auto& current_kernel_params =
//                                     kernel_parameters[cfg.n_power][i];
//                                 InverseCore<<<
//                                     dim3(current_kernel_params.griddim_x,
//                                          current_kernel_params.griddim_y,
//                                          batch_size),
//                                     dim3(current_kernel_params.blockdim_x,
//                                          current_kernel_params.blockdim_y),
//                                     current_kernel_params.shared_memory,
//                                     cfg.stream>>>(
//                                     device_in_, device_out, root_of_unity_table,
//                                     modulus, current_kernel_params.shared_index,
//                                     current_kernel_params.logm,
//                                     current_kernel_params.k,
//                                     current_kernel_params.outer_iteration_count,
//                                     cfg.n_power, cfg.mod_inverse,
//                                     current_kernel_params.not_last_kernel,
//                                     (cfg.reduction_poly ==
//                                      ReductionPolynomial::X_N_minus),
//                                     mod_count);
//                                 GPUNTT_CUDA_CHECK(cudaGetLastError());
//                                 device_in_ = device_out;
//                             }
//                         }
//                     }
//                     else
//                     {
//                         if constexpr (std::is_signed<T>::value)
//                         {
//                             auto& current_kernel_params =
//                                 kernel_parameters[cfg.n_power][0];
//                             InverseCore_<<<
//                                 dim3(current_kernel_params.griddim_x,
//                                      current_kernel_params.griddim_y,
//                                      batch_size),
//                                 dim3(current_kernel_params.blockdim_x,
//                                      current_kernel_params.blockdim_y),
//                                 current_kernel_params.shared_memory,
//                                 cfg.stream>>>(
//                                 device_in,
//                                 reinterpret_cast<
//                                     typename std::make_unsigned<T>::type*>(
//                                     device_out),
//                                 root_of_unity_table, modulus,
//                                 current_kernel_params.shared_index,
//                                 current_kernel_params.logm,
//                                 current_kernel_params.k,
//                                 current_kernel_params.outer_iteration_count,
//                                 cfg.n_power, cfg.mod_inverse,
//                                 current_kernel_params.not_last_kernel,
//                                 (cfg.reduction_poly ==
//                                  ReductionPolynomial::X_N_minus),
//                                 mod_count);
//                             GPUNTT_CUDA_CHECK(cudaGetLastError());
//                             for (int i = 1;
//                                  i < kernel_parameters[cfg.n_power].size() - 1;
//                                  i++)
//                             {
//                                 auto& current_kernel_params =
//                                     kernel_parameters[cfg.n_power][i];
//                                 InverseCore<<<
//                                     dim3(current_kernel_params.griddim_x,
//                                          current_kernel_params.griddim_y,
//                                          batch_size),
//                                     dim3(current_kernel_params.blockdim_x,
//                                          current_kernel_params.blockdim_y),
//                                     current_kernel_params.shared_memory,
//                                     cfg.stream>>>(
//                                     reinterpret_cast<
//                                         typename std::make_unsigned<T>::type*>(
//                                         device_out),
//                                     reinterpret_cast<
//                                         typename std::make_unsigned<T>::type*>(
//                                         device_out),
//                                     root_of_unity_table, modulus,
//                                     current_kernel_params.shared_index,
//                                     current_kernel_params.logm,
//                                     current_kernel_params.k,
//                                     current_kernel_params.outer_iteration_count,
//                                     cfg.n_power, cfg.mod_inverse,
//                                     current_kernel_params.not_last_kernel,
//                                     (cfg.reduction_poly ==
//                                      ReductionPolynomial::X_N_minus),
//                                     mod_count);
//                                 GPUNTT_CUDA_CHECK(cudaGetLastError());
//                             }

//                             current_kernel_params = kernel_parameters
//                                 [cfg.n_power]
//                                 [kernel_parameters[cfg.n_power].size() - 1];
//                             InverseCore<<<dim3(current_kernel_params.griddim_x,
//                                                current_kernel_params.griddim_y,
//                                                batch_size),
//                                           dim3(
//                                               current_kernel_params.blockdim_x,
//                                               current_kernel_params.blockdim_y),
//                                           current_kernel_params.shared_memory,
//                                           cfg.stream>>>(
//                                 reinterpret_cast<
//                                     typename std::make_unsigned<T>::type*>(
//                                     device_out),
//                                 device_out, root_of_unity_table, modulus,
//                                 current_kernel_params.shared_index,
//                                 current_kernel_params.logm,
//                                 current_kernel_params.k,
//                                 current_kernel_params.outer_iteration_count,
//                                 cfg.n_power, cfg.mod_inverse,
//                                 current_kernel_params.not_last_kernel,
//                                 (cfg.reduction_poly ==
//                                  ReductionPolynomial::X_N_minus),
//                                 mod_count);
//                             GPUNTT_CUDA_CHECK(cudaGetLastError());
//                         }
//                         else
//                         {
//                             auto& current_kernel_params =
//                                 kernel_parameters[cfg.n_power][0];
//                             InverseCore_<<<
//                                 dim3(current_kernel_params.griddim_x,
//                                      current_kernel_params.griddim_y,
//                                      batch_size),
//                                 dim3(current_kernel_params.blockdim_x,
//                                      current_kernel_params.blockdim_y),
//                                 current_kernel_params.shared_memory,
//                                 cfg.stream>>>(
//                                 device_in, device_out, root_of_unity_table,
//                                 modulus, current_kernel_params.shared_index,
//                                 current_kernel_params.logm,
//                                 current_kernel_params.k,
//                                 current_kernel_params.outer_iteration_count,
//                                 cfg.n_power, cfg.mod_inverse,
//                                 current_kernel_params.not_last_kernel,
//                                 (cfg.reduction_poly ==
//                                  ReductionPolynomial::X_N_minus),
//                                 mod_count);
//                             GPUNTT_CUDA_CHECK(cudaGetLastError());
//                             for (int i = 1;
//                                  i < kernel_parameters[cfg.n_power].size(); i++)
//                             {
//                                 auto& current_kernel_params =
//                                     kernel_parameters[cfg.n_power][i];
//                                 InverseCore<<<
//                                     dim3(current_kernel_params.griddim_x,
//                                          current_kernel_params.griddim_y,
//                                          batch_size),
//                                     dim3(current_kernel_params.blockdim_x,
//                                          current_kernel_params.blockdim_y),
//                                     current_kernel_params.shared_memory,
//                                     cfg.stream>>>(
//                                     device_out, device_out, root_of_unity_table,
//                                     modulus, current_kernel_params.shared_index,
//                                     current_kernel_params.logm,
//                                     current_kernel_params.k,
//                                     current_kernel_params.outer_iteration_count,
//                                     cfg.n_power, cfg.mod_inverse,
//                                     current_kernel_params.not_last_kernel,
//                                     (cfg.reduction_poly ==
//                                      ReductionPolynomial::X_N_minus),
//                                     mod_count);
//                                 GPUNTT_CUDA_CHECK(cudaGetLastError());
//                             }
//                         }
//                     }
//                 }
//             }
//             break;
//             case PerCoefficient:
//             {
//                 if ((cfg.n_power <= 0 || cfg.n_power >= 10))
//                 {
//                     throw std::invalid_argument("Invalid n_power range!");
//                 }

//                 int log_batch_size = log2(batch_size);
//                 int total_size = 1 << (cfg.n_power + log_batch_size);
//                 int total_block_thread = 512;
//                 int total_block_count = total_size / (total_block_thread * 2);
//                 int blockdim_y = 1 << (cfg.n_power - 1);
//                 int blockdim_x = total_block_thread / blockdim_y;
//                 InverseCoreTranspose<<<
//                     dim3(total_block_count, 1, 1),
//                     dim3(blockdim_x, blockdim_y, 1),
//                     ((total_block_thread * 2) + (1 << cfg.n_power)) * sizeof(T),
//                     cfg.stream>>>(
//                     device_in, device_out, root_of_unity_table, modulus,
//                     cfg.mod_inverse, cfg.n_power, log_batch_size,
//                     (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
//                     mod_count);
//                 GPUNTT_CUDA_CHECK(cudaGetLastError());
//             }
//             break;
//             default:
//                 throw std::invalid_argument("Invalid ntt_layout!");
//                 break;
//         }
//     }

//     template <typename T>
//     __host__ void GPU_NTT_Inplace(T* device_inout, Root<T>* root_of_unity_table,
//                                   Modulus<T> modulus, ntt_configuration<T> cfg,
//                                   int batch_size)
//     {
//         GPU_NTT(device_inout, device_inout, root_of_unity_table, modulus, cfg,
//                 batch_size);
//     }

    template <typename T>
    void GPU_NTT_Inplace(T* device_inout, Root<T>* root_of_unity_table,
                                  Modulus<T>* modulus,
                                  ntt_rns_configuration<T> cfg, int batch_size,
                                  int mod_count)
    {
        GPU_NTT<T>(device_inout, device_inout, root_of_unity_table, modulus, cfg,
                batch_size, mod_count);
    }

//     template <typename T>
//     __host__ void GPU_INTT_Inplace(T* device_inout,
//                                    Root<T>* root_of_unity_table,
//                                    Modulus<T> modulus, ntt_configuration<T> cfg,
//                                    int batch_size)
//     {
//         GPU_INTT(device_inout, device_inout, root_of_unity_table, modulus, cfg,
//                  batch_size);
//     }

//     template <typename T>
//     __host__ void
//     GPU_INTT_Inplace(T* device_inout, Root<T>* root_of_unity_table,
//                      Modulus<T>* modulus, ntt_rns_configuration<T> cfg,
//                      int batch_size, int mod_count)
//     {
//         GPU_INTT(device_inout, device_inout, root_of_unity_table, modulus, cfg,
//                  batch_size, mod_count);
//     }

//     ////////////////////////////////////
//     // Modulus Ordered
//     ////////////////////////////////////

//     template <typename T>
//     __global__ void ForwardCoreModulusOrdered(
//         T* polynomial_in, T* polynomial_out, Root<T>* root_of_unity_table,
//         Modulus<T>* modulus, int shared_index, int logm,
//         int outer_iteration_count, int N_power, bool zero_padding,
//         bool not_last_kernel, bool reduction_poly_check, int mod_count,
//         int* order)
//     {
//         const int idx_x = threadIdx.x;
//         const int idx_y = threadIdx.y;
//         const int block_x = blockIdx.x;
//         const int block_y = blockIdx.y;
//         const int block_z = blockIdx.z;

//         const int mod_index = block_z % mod_count;
//         const int prime_index = order[mod_index];

//         // extern __shared__ T shared_memory[];
//         extern __shared__ char shared_memory_typed[];
//         T* shared_memory = reinterpret_cast<T*>(shared_memory_typed);

//         int t_2 = N_power - logm - 1;
//         location_t offset = 1 << (N_power - logm - 1);
//         int t_ = shared_index;
//         location_t m = (location_t) 1 << logm;

//         location_t global_addresss =
//             idx_x +
//             (location_t) (idx_y *
//                           (offset / (1 << (outer_iteration_count - 1)))) +
//             (location_t) (blockDim.x * block_x) +
//             (location_t) (2 * block_y * offset) +
//             (location_t) (block_z << N_power);

//         location_t omega_addresss =
//             idx_x +
//             (location_t) (idx_y *
//                           (offset / (1 << (outer_iteration_count - 1)))) +
//             (location_t) (blockDim.x * block_x) +
//             (location_t) (block_y * offset);

//         location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

//         // Load UInt64 from global & store to shared
//         shared_memory[shared_addresss] = polynomial_in[global_addresss];
//         shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
//             polynomial_in[global_addresss + offset];

//         int t = 1 << t_;
//         int in_shared_address =
//             ((shared_addresss >> t_) << t_) + shared_addresss;
//         location_t current_root_index;
//         if (not_last_kernel)
//         {
// #pragma unroll
//             for (int lp = 0; lp < outer_iteration_count; lp++)
//             {
//                 __syncthreads();
//                 if (reduction_poly_check)
//                 { // X_N_minus
//                     current_root_index = (omega_addresss >> t_2) +
//                                          (location_t) (prime_index << N_power);
//                 }
//                 else
//                 { // X_N_plus
//                     current_root_index = m + (omega_addresss >> t_2) +
//                                          (location_t) (prime_index << N_power);
//                 }
//                 CooleyTukeyUnit(shared_memory[in_shared_address],
//                                 shared_memory[in_shared_address + t],
//                                 root_of_unity_table[current_root_index],
//                                 modulus[prime_index]);

//                 t = t >> 1;
//                 t_2 -= 1;
//                 t_ -= 1;
//                 m <<= 1;

//                 in_shared_address =
//                     ((shared_addresss >> t_) << t_) + shared_addresss;
//                 //__syncthreads();
//             }
//             __syncthreads();
//         }
//         else
//         {
// #pragma unroll
//             for (int lp = 0; lp < (shared_index - 5); lp++) // 4 for 512 thread
//             {
//                 __syncthreads();
//                 if (reduction_poly_check)
//                 { // X_N_minus
//                     current_root_index = (omega_addresss >> t_2) +
//                                          (location_t) (prime_index << N_power);
//                 }
//                 else
//                 { // X_N_plus
//                     current_root_index = m + (omega_addresss >> t_2) +
//                                          (location_t) (prime_index << N_power);
//                 }

//                 CooleyTukeyUnit(shared_memory[in_shared_address],
//                                 shared_memory[in_shared_address + t],
//                                 root_of_unity_table[current_root_index],
//                                 modulus[prime_index]);

//                 t = t >> 1;
//                 t_2 -= 1;
//                 t_ -= 1;
//                 m <<= 1;

//                 in_shared_address =
//                     ((shared_addresss >> t_) << t_) + shared_addresss;
//                 //__syncthreads();
//             }
//             __syncthreads();

// #pragma unroll
//             for (int lp = 0; lp < 6; lp++)
//             {
//                 if (reduction_poly_check)
//                 { // X_N_minus
//                     current_root_index = (omega_addresss >> t_2) +
//                                          (location_t) (prime_index << N_power);
//                 }
//                 else
//                 { // X_N_plus
//                     current_root_index = m + (omega_addresss >> t_2) +
//                                          (location_t) (prime_index << N_power);
//                 }
//                 CooleyTukeyUnit(shared_memory[in_shared_address],
//                                 shared_memory[in_shared_address + t],
//                                 root_of_unity_table[current_root_index],
//                                 modulus[prime_index]);

//                 t = t >> 1;
//                 t_2 -= 1;
//                 t_ -= 1;
//                 m <<= 1;

//                 in_shared_address =
//                     ((shared_addresss >> t_) << t_) + shared_addresss;
//             }
//             __syncthreads();
//         }

//         polynomial_out[global_addresss] = shared_memory[shared_addresss];
//         polynomial_out[global_addresss + offset] =
//             shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
//     }

//     template <typename T>
//     __global__ void ForwardCoreModulusOrdered_(
//         T* polynomial_in, T* polynomial_out, Root<T>* root_of_unity_table,
//         Modulus<T>* modulus, int shared_index, int logm,
//         int outer_iteration_count, int N_power, bool zero_padding,
//         bool not_last_kernel, bool reduction_poly_check, int mod_count,
//         int* order)
//     {
//         const int idx_x = threadIdx.x;
//         const int idx_y = threadIdx.y;
//         const int block_x = blockIdx.x;
//         const int block_y = blockIdx.y;
//         const int block_z = blockIdx.z;

//         const int mod_index = block_z % mod_count;
//         const int prime_index = order[mod_index];

//         // extern __shared__ T shared_memory[];
//         extern __shared__ char shared_memory_typed[];
//         T* shared_memory = reinterpret_cast<T*>(shared_memory_typed);

//         int t_2 = N_power - logm - 1;
//         location_t offset = 1 << (N_power - logm - 1);
//         int t_ = shared_index;
//         location_t m = (location_t) 1 << logm;

//         location_t global_addresss =
//             idx_x +
//             (location_t) (idx_y *
//                           (offset / (1 << (outer_iteration_count - 1)))) +
//             (location_t) (blockDim.x * block_y) +
//             (location_t) (2 * block_x * offset) +
//             (location_t) (block_z << N_power);
//         location_t omega_addresss =
//             idx_x +
//             (location_t) (idx_y *
//                           (offset / (1 << (outer_iteration_count - 1)))) +
//             (location_t) (blockDim.x * block_y) +
//             (location_t) (block_x * offset);
//         location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

//         // Load UInt64 from global & store to shared
//         shared_memory[shared_addresss] = polynomial_in[global_addresss];
//         shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
//             polynomial_in[global_addresss + offset];

//         int t = 1 << t_;
//         int in_shared_address =
//             ((shared_addresss >> t_) << t_) + shared_addresss;
//         location_t current_root_index;
//         if (not_last_kernel)
//         {
// #pragma unroll
//             for (int lp = 0; lp < outer_iteration_count; lp++)
//             {
//                 __syncthreads();
//                 if (reduction_poly_check)
//                 { // X_N_minus
//                     current_root_index = (omega_addresss >> t_2) +
//                                          (location_t) (prime_index << N_power);
//                 }
//                 else
//                 { // X_N_plus
//                     current_root_index = m + (omega_addresss >> t_2) +
//                                          (location_t) (prime_index << N_power);
//                 }
//                 CooleyTukeyUnit(shared_memory[in_shared_address],
//                                 shared_memory[in_shared_address + t],
//                                 root_of_unity_table[current_root_index],
//                                 modulus[prime_index]);

//                 t = t >> 1;
//                 t_2 -= 1;
//                 t_ -= 1;
//                 m <<= 1;

//                 in_shared_address =
//                     ((shared_addresss >> t_) << t_) + shared_addresss;
//                 //__syncthreads();
//             }
//             __syncthreads();
//         }
//         else
//         {
// #pragma unroll
//             for (int lp = 0; lp < (shared_index - 5); lp++)
//             {
//                 __syncthreads();
//                 if (reduction_poly_check)
//                 { // X_N_minus
//                     current_root_index = (omega_addresss >> t_2) +
//                                          (location_t) (prime_index << N_power);
//                 }
//                 else
//                 { // X_N_plus
//                     current_root_index = m + (omega_addresss >> t_2) +
//                                          (location_t) (prime_index << N_power);
//                 }
//                 CooleyTukeyUnit(shared_memory[in_shared_address],
//                                 shared_memory[in_shared_address + t],
//                                 root_of_unity_table[current_root_index],
//                                 modulus[prime_index]);

//                 t = t >> 1;
//                 t_2 -= 1;
//                 t_ -= 1;
//                 m <<= 1;

//                 in_shared_address =
//                     ((shared_addresss >> t_) << t_) + shared_addresss;
//                 //__syncthreads();
//             }
//             __syncthreads();

// #pragma unroll
//             for (int lp = 0; lp < 6; lp++)
//             {
//                 if (reduction_poly_check)
//                 { // X_N_minus
//                     current_root_index = (omega_addresss >> t_2) +
//                                          (location_t) (prime_index << N_power);
//                 }
//                 else
//                 { // X_N_plus
//                     current_root_index = m + (omega_addresss >> t_2) +
//                                          (location_t) (prime_index << N_power);
//                 }
//                 CooleyTukeyUnit(shared_memory[in_shared_address],
//                                 shared_memory[in_shared_address + t],
//                                 root_of_unity_table[current_root_index],
//                                 modulus[prime_index]);

//                 t = t >> 1;
//                 t_2 -= 1;
//                 t_ -= 1;
//                 m <<= 1;

//                 in_shared_address =
//                     ((shared_addresss >> t_) << t_) + shared_addresss;
//             }
//             __syncthreads();
//         }

//         polynomial_out[global_addresss] = shared_memory[shared_addresss];
//         polynomial_out[global_addresss + offset] =
//             shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
//     }

//     template <typename T>
//     __global__ void InverseCoreModulusOrdered(
//         T* polynomial_in, T* polynomial_out,
//         Root<T>* inverse_root_of_unity_table, Modulus<T>* modulus,
//         int shared_index, int logm, int k, int outer_iteration_count,
//         int N_power, Ninverse<T>* n_inverse, bool last_kernel,
//         bool reduction_poly_check, int mod_count, int* order)
//     {
//         const int idx_x = threadIdx.x;
//         const int idx_y = threadIdx.y;
//         const int block_x = blockIdx.x;
//         const int block_y = blockIdx.y;
//         const int block_z = blockIdx.z;

//         const int mod_index = block_z % mod_count;
//         const int prime_index = order[mod_index];

//         // extern __shared__ T shared_memory[];
//         extern __shared__ char shared_memory_typed[];
//         T* shared_memory = reinterpret_cast<T*>(shared_memory_typed);

//         int t_2 = N_power - logm - 1;
//         location_t offset = 1 << (N_power - k - 1);
//         // int t_ = 9 - outer_iteration_count;
//         int t_ = (shared_index + 1) - outer_iteration_count;
//         int loops = outer_iteration_count;
//         location_t m = (location_t) 1 << logm;

//         location_t global_addresss =
//             idx_x +
//             (location_t) (idx_y *
//                           (offset / (1 << (outer_iteration_count - 1)))) +
//             (location_t) (blockDim.x * block_x) +
//             (location_t) (2 * block_y * offset) +
//             (location_t) (block_z << N_power);

//         location_t omega_addresss =
//             idx_x +
//             (location_t) (idx_y *
//                           (offset / (1 << (outer_iteration_count - 1)))) +
//             (location_t) (blockDim.x * block_x) +
//             (location_t) (block_y * offset);
//         location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

//         shared_memory[shared_addresss] = polynomial_in[global_addresss];
//         shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
//             polynomial_in[global_addresss + offset];

//         int t = 1 << t_;
//         int in_shared_address =
//             ((shared_addresss >> t_) << t_) + shared_addresss;
//         location_t current_root_index;
// #pragma unroll
//         for (int lp = 0; lp < loops; lp++)
//         {
//             __syncthreads();
//             if (reduction_poly_check)
//             { // X_N_minus
//                 current_root_index = (omega_addresss >> t_2) +
//                                      (location_t) (prime_index << N_power);
//             }
//             else
//             { // X_N_plus
//                 current_root_index = m + (omega_addresss >> t_2) +
//                                      (location_t) (prime_index << N_power);
//             }

//             GentlemanSandeUnit(shared_memory[in_shared_address],
//                                shared_memory[in_shared_address + t],
//                                inverse_root_of_unity_table[current_root_index],
//                                modulus[prime_index]);

//             t = t << 1;
//             t_2 += 1;
//             t_ += 1;
//             m >>= 1;

//             in_shared_address =
//                 ((shared_addresss >> t_) << t_) + shared_addresss;
//         }
//         __syncthreads();

//         if (last_kernel)
//         {
//             polynomial_out[global_addresss] = OPERATOR_GPU<T>::mult(
//                 shared_memory[shared_addresss], n_inverse[prime_index],
//                 modulus[prime_index]);
//             polynomial_out[global_addresss + offset] = OPERATOR_GPU<T>::mult(
//                 shared_memory[shared_addresss + (blockDim.x * blockDim.y)],
//                 n_inverse[prime_index], modulus[prime_index]);
//         }
//         else
//         {
//             polynomial_out[global_addresss] = shared_memory[shared_addresss];
//             polynomial_out[global_addresss + offset] =
//                 shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
//         }
//     }

//     template <typename T>
//     __global__ void InverseCoreModulusOrdered_(
//         T* polynomial_in, T* polynomial_out,
//         Root<T>* inverse_root_of_unity_table, Modulus<T>* modulus,
//         int shared_index, int logm, int k, int outer_iteration_count,
//         int N_power, Ninverse<T>* n_inverse, bool last_kernel,
//         bool reduction_poly_check, int mod_count, int* order)
//     {
//         const int idx_x = threadIdx.x;
//         const int idx_y = threadIdx.y;
//         const int block_x = blockIdx.x;
//         const int block_y = blockIdx.y;
//         const int block_z = blockIdx.z;

//         const int mod_index = block_z % mod_count;
//         const int prime_index = order[mod_index];

//         // extern __shared__ T shared_memory[];
//         extern __shared__ char shared_memory_typed[];
//         T* shared_memory = reinterpret_cast<T*>(shared_memory_typed);

//         int t_2 = N_power - logm - 1;
//         location_t offset = 1 << (N_power - k - 1);
//         // int t_ = 9 - outer_iteration_count;
//         int t_ = (shared_index + 1) - outer_iteration_count;
//         int loops = outer_iteration_count;
//         location_t m = (location_t) 1 << logm;

//         location_t global_addresss =
//             idx_x +
//             (location_t) (idx_y *
//                           (offset / (1 << (outer_iteration_count - 1)))) +
//             (location_t) (blockDim.x * block_y) +
//             (location_t) (2 * block_x * offset) +
//             (location_t) (block_z << N_power);

//         location_t omega_addresss =
//             idx_x +
//             (location_t) (idx_y *
//                           (offset / (1 << (outer_iteration_count - 1)))) +
//             (location_t) (blockDim.x * block_y) +
//             (location_t) (block_x * offset);
//         location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

//         shared_memory[shared_addresss] = polynomial_in[global_addresss];
//         shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
//             polynomial_in[global_addresss + offset];

//         int t = 1 << t_;
//         int in_shared_address =
//             ((shared_addresss >> t_) << t_) + shared_addresss;
//         location_t current_root_index;
// #pragma unroll
//         for (int lp = 0; lp < loops; lp++)
//         {
//             __syncthreads();
//             if (reduction_poly_check)
//             { // X_N_minus
//                 current_root_index = (omega_addresss >> t_2) +
//                                      (location_t) (prime_index << N_power);
//             }
//             else
//             { // X_N_plus
//                 current_root_index = m + (omega_addresss >> t_2) +
//                                      (location_t) (prime_index << N_power);
//             }

//             GentlemanSandeUnit(shared_memory[in_shared_address],
//                                shared_memory[in_shared_address + t],
//                                inverse_root_of_unity_table[current_root_index],
//                                modulus[prime_index]);

//             t = t << 1;
//             t_2 += 1;
//             t_ += 1;
//             m >>= 1;

//             in_shared_address =
//                 ((shared_addresss >> t_) << t_) + shared_addresss;
//         }
//         __syncthreads();

//         if (last_kernel)
//         {
//             polynomial_out[global_addresss] = OPERATOR_GPU<T>::mult(
//                 shared_memory[shared_addresss], n_inverse[prime_index],
//                 modulus[prime_index]);
//             polynomial_out[global_addresss + offset] = OPERATOR_GPU<T>::mult(
//                 shared_memory[shared_addresss + (blockDim.x * blockDim.y)],
//                 n_inverse[prime_index], modulus[prime_index]);
//         }
//         else
//         {
//             polynomial_out[global_addresss] = shared_memory[shared_addresss];
//             polynomial_out[global_addresss + offset] =
//                 shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
//         }
//     }

//     template <typename T>
//     __host__ void
//     GPU_NTT_Modulus_Ordered(T* device_in, T* device_out,
//                             Root<T>* root_of_unity_table, Modulus<T>* modulus,
//                             ntt_rns_configuration<T> cfg, int batch_size,
//                             int mod_count, int* order)
//     {
//         if ((cfg.n_power <= 9 || cfg.n_power >= 29))
//         {
//             throw std::invalid_argument("Invalid n_power range!");
//         }

//         auto kernel_parameters = (cfg.ntt_type == FORWARD)
//                                      ? CreateForwardNTTKernel<T>()
//                                      : CreateInverseNTTKernel<T>();
//         bool standart_kernel = (cfg.n_power < 25) ? true : false;
//         T* device_in_ = device_in;

//         switch (cfg.ntt_type)
//         {
//             case FORWARD:
//                 if (standart_kernel)
//                 {
//                     for (int i = 0; i < kernel_parameters[cfg.n_power].size();
//                          i++)
//                     {
//                         auto& current_kernel_params =
//                             kernel_parameters[cfg.n_power][i];
//                         ForwardCoreModulusOrdered<<<
//                             dim3(current_kernel_params.griddim_x,
//                                  current_kernel_params.griddim_y, batch_size),
//                             dim3(current_kernel_params.blockdim_x,
//                                  current_kernel_params.blockdim_y),
//                             current_kernel_params.shared_memory, cfg.stream>>>(
//                             device_in_, device_out, root_of_unity_table,
//                             modulus, current_kernel_params.shared_index,
//                             current_kernel_params.logm,
//                             current_kernel_params.outer_iteration_count,
//                             cfg.n_power, cfg.zero_padding,
//                             current_kernel_params.not_last_kernel,
//                             (cfg.reduction_poly ==
//                              ReductionPolynomial::X_N_minus),
//                             mod_count, order);
//                         GPUNTT_CUDA_CHECK(cudaGetLastError());
//                         device_in_ = device_out;
//                     }
//                 }
//                 else
//                 {
//                     for (int i = 0;
//                          i < kernel_parameters[cfg.n_power].size() - 1; i++)
//                     {
//                         auto& current_kernel_params =
//                             kernel_parameters[cfg.n_power][i];
//                         ForwardCoreModulusOrdered<<<
//                             dim3(current_kernel_params.griddim_x,
//                                  current_kernel_params.griddim_y, batch_size),
//                             dim3(current_kernel_params.blockdim_x,
//                                  current_kernel_params.blockdim_y),
//                             current_kernel_params.shared_memory, cfg.stream>>>(
//                             device_in_, device_out, root_of_unity_table,
//                             modulus, current_kernel_params.shared_index,
//                             current_kernel_params.logm,
//                             current_kernel_params.outer_iteration_count,
//                             cfg.n_power, cfg.zero_padding,
//                             current_kernel_params.not_last_kernel,
//                             (cfg.reduction_poly ==
//                              ReductionPolynomial::X_N_minus),
//                             mod_count, order);
//                         GPUNTT_CUDA_CHECK(cudaGetLastError());
//                         device_in_ = device_out;
//                     }
//                     auto& current_kernel_params = kernel_parameters
//                         [cfg.n_power]
//                         [kernel_parameters[cfg.n_power].size() - 1];
//                     ForwardCoreModulusOrdered_<<<
//                         dim3(current_kernel_params.griddim_x,
//                              current_kernel_params.griddim_y, batch_size),
//                         dim3(current_kernel_params.blockdim_x,
//                              current_kernel_params.blockdim_y),
//                         current_kernel_params.shared_memory, cfg.stream>>>(
//                         device_in_, device_out, root_of_unity_table, modulus,
//                         current_kernel_params.shared_index,
//                         current_kernel_params.logm,
//                         current_kernel_params.outer_iteration_count,
//                         cfg.n_power, cfg.zero_padding,
//                         current_kernel_params.not_last_kernel,
//                         (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
//                         mod_count, order);
//                     GPUNTT_CUDA_CHECK(cudaGetLastError());
//                 }
//                 break;
//             case INVERSE:
//                 if (standart_kernel)
//                 {
//                     for (int i = 0; i < kernel_parameters[cfg.n_power].size();
//                          i++)
//                     {
//                         auto& current_kernel_params =
//                             kernel_parameters[cfg.n_power][i];
//                         InverseCoreModulusOrdered<<<
//                             dim3(current_kernel_params.griddim_x,
//                                  current_kernel_params.griddim_y, batch_size),
//                             dim3(current_kernel_params.blockdim_x,
//                                  current_kernel_params.blockdim_y),
//                             current_kernel_params.shared_memory, cfg.stream>>>(
//                             device_in_, device_out, root_of_unity_table,
//                             modulus, current_kernel_params.shared_index,
//                             current_kernel_params.logm, current_kernel_params.k,
//                             current_kernel_params.outer_iteration_count,
//                             cfg.n_power, cfg.mod_inverse,
//                             current_kernel_params.not_last_kernel,
//                             (cfg.reduction_poly ==
//                              ReductionPolynomial::X_N_minus),
//                             mod_count, order);
//                         GPUNTT_CUDA_CHECK(cudaGetLastError());
//                         device_in_ = device_out;
//                     }
//                 }
//                 else
//                 {
//                     auto& current_kernel_params =
//                         kernel_parameters[cfg.n_power][0];
//                     InverseCoreModulusOrdered_<<<
//                         dim3(current_kernel_params.griddim_x,
//                              current_kernel_params.griddim_y, batch_size),
//                         dim3(current_kernel_params.blockdim_x,
//                              current_kernel_params.blockdim_y),
//                         current_kernel_params.shared_memory, cfg.stream>>>(
//                         device_in_, device_out, root_of_unity_table, modulus,
//                         current_kernel_params.shared_index,
//                         current_kernel_params.logm, current_kernel_params.k,
//                         current_kernel_params.outer_iteration_count,
//                         cfg.n_power, cfg.mod_inverse,
//                         current_kernel_params.not_last_kernel,
//                         (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
//                         mod_count, order);
//                     GPUNTT_CUDA_CHECK(cudaGetLastError());
//                     device_in_ = device_out;
//                     for (int i = 1; i < kernel_parameters[cfg.n_power].size();
//                          i++)
//                     {
//                         auto& current_kernel_params =
//                             kernel_parameters[cfg.n_power][i];
//                         InverseCoreModulusOrdered<<<
//                             dim3(current_kernel_params.griddim_x,
//                                  current_kernel_params.griddim_y, batch_size),
//                             dim3(current_kernel_params.blockdim_x,
//                                  current_kernel_params.blockdim_y),
//                             current_kernel_params.shared_memory, cfg.stream>>>(
//                             device_in_, device_out, root_of_unity_table,
//                             modulus, current_kernel_params.shared_index,
//                             current_kernel_params.logm, current_kernel_params.k,
//                             current_kernel_params.outer_iteration_count,
//                             cfg.n_power, cfg.mod_inverse,
//                             current_kernel_params.not_last_kernel,
//                             (cfg.reduction_poly ==
//                              ReductionPolynomial::X_N_minus),
//                             mod_count, order);
//                         GPUNTT_CUDA_CHECK(cudaGetLastError());
//                     }
//                 }
//                 break;
//             default:
//                 throw std::invalid_argument("Invalid ntt_type!");
//                 break;
//         }
//     }

//     template <typename T>
//     __host__ void GPU_NTT_Modulus_Ordered_Inplace(
//         T* device_inout, Root<T>* root_of_unity_table, Modulus<T>* modulus,
//         ntt_rns_configuration<T> cfg, int batch_size, int mod_count, int* order)
//     {
//         GPU_NTT_Modulus_Ordered(device_inout, device_inout, root_of_unity_table,
//                                 modulus, cfg, batch_size, mod_count, order);
//     }

//     ////////////////////////////////////
//     // Poly Ordered
//     ////////////////////////////////////

//     template <typename T>
//     __global__ void
//     ForwardCorePolyOrdered(T* polynomial_in, T* polynomial_out,
//                            Root<T>* root_of_unity_table, Modulus<T>* modulus,
//                            int shared_index, int logm,
//                            int outer_iteration_count, int N_power,
//                            bool zero_padding, bool not_last_kernel,
//                            bool reduction_poly_check, int mod_count, int* order)
//     {
//         const int idx_x = threadIdx.x;
//         const int idx_y = threadIdx.y;
//         const int block_x = blockIdx.x;
//         const int block_y = blockIdx.y;
//         const int block_z = blockIdx.z;

//         const int mod_index = block_z % mod_count;
//         const int input_index = order[block_z];

//         // extern __shared__ T shared_memory[];
//         extern __shared__ char shared_memory_typed[];
//         T* shared_memory = reinterpret_cast<T*>(shared_memory_typed);

//         int t_2 = N_power - logm - 1;
//         location_t offset = 1 << (N_power - logm - 1);
//         int t_ = shared_index;
//         location_t m = (location_t) 1 << logm;

//         location_t global_addresss =
//             idx_x +
//             (location_t) (idx_y *
//                           (offset / (1 << (outer_iteration_count - 1)))) +
//             (location_t) (blockDim.x * block_x) +
//             (location_t) (2 * block_y * offset) +
//             (location_t) (input_index << N_power);

//         location_t omega_addresss =
//             idx_x +
//             (location_t) (idx_y *
//                           (offset / (1 << (outer_iteration_count - 1)))) +
//             (location_t) (blockDim.x * block_x) +
//             (location_t) (block_y * offset);

//         location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

//         // Load UInt64 from global & store to shared
//         shared_memory[shared_addresss] = polynomial_in[global_addresss];
//         shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
//             polynomial_in[global_addresss + offset];

//         int t = 1 << t_;
//         int in_shared_address =
//             ((shared_addresss >> t_) << t_) + shared_addresss;
//         location_t current_root_index;
//         if (not_last_kernel)
//         {
// #pragma unroll
//             for (int lp = 0; lp < outer_iteration_count; lp++)
//             {
//                 __syncthreads();
//                 if (reduction_poly_check)
//                 { // X_N_minus
//                     current_root_index = (omega_addresss >> t_2) +
//                                          (location_t) (mod_index << N_power);
//                 }
//                 else
//                 { // X_N_plus
//                     current_root_index = m + (omega_addresss >> t_2) +
//                                          (location_t) (mod_index << N_power);
//                 }
//                 CooleyTukeyUnit(shared_memory[in_shared_address],
//                                 shared_memory[in_shared_address + t],
//                                 root_of_unity_table[current_root_index],
//                                 modulus[mod_index]);

//                 t = t >> 1;
//                 t_2 -= 1;
//                 t_ -= 1;
//                 m <<= 1;

//                 in_shared_address =
//                     ((shared_addresss >> t_) << t_) + shared_addresss;
//                 //__syncthreads();
//             }
//             __syncthreads();
//         }
//         else
//         {
// #pragma unroll
//             for (int lp = 0; lp < (shared_index - 5); lp++) // 4 for 512 thread
//             {
//                 __syncthreads();
//                 if (reduction_poly_check)
//                 { // X_N_minus
//                     current_root_index = (omega_addresss >> t_2) +
//                                          (location_t) (mod_index << N_power);
//                 }
//                 else
//                 { // X_N_plus
//                     current_root_index = m + (omega_addresss >> t_2) +
//                                          (location_t) (mod_index << N_power);
//                 }

//                 CooleyTukeyUnit(shared_memory[in_shared_address],
//                                 shared_memory[in_shared_address + t],
//                                 root_of_unity_table[current_root_index],
//                                 modulus[mod_index]);

//                 t = t >> 1;
//                 t_2 -= 1;
//                 t_ -= 1;
//                 m <<= 1;

//                 in_shared_address =
//                     ((shared_addresss >> t_) << t_) + shared_addresss;
//                 //__syncthreads();
//             }
//             __syncthreads();

// #pragma unroll
//             for (int lp = 0; lp < 6; lp++)
//             {
//                 if (reduction_poly_check)
//                 { // X_N_minus
//                     current_root_index = (omega_addresss >> t_2) +
//                                          (location_t) (mod_index << N_power);
//                 }
//                 else
//                 { // X_N_plus
//                     current_root_index = m + (omega_addresss >> t_2) +
//                                          (location_t) (mod_index << N_power);
//                 }
//                 CooleyTukeyUnit(shared_memory[in_shared_address],
//                                 shared_memory[in_shared_address + t],
//                                 root_of_unity_table[current_root_index],
//                                 modulus[mod_index]);

//                 t = t >> 1;
//                 t_2 -= 1;
//                 t_ -= 1;
//                 m <<= 1;

//                 in_shared_address =
//                     ((shared_addresss >> t_) << t_) + shared_addresss;
//             }
//             __syncthreads();
//         }

//         polynomial_out[global_addresss] = shared_memory[shared_addresss];
//         polynomial_out[global_addresss + offset] =
//             shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
//     }

//     template <typename T>
//     __global__ void ForwardCorePolyOrdered_(
//         T* polynomial_in, T* polynomial_out, Root<T>* root_of_unity_table,
//         Modulus<T>* modulus, int shared_index, int logm,
//         int outer_iteration_count, int N_power, bool zero_padding,
//         bool not_last_kernel, bool reduction_poly_check, int mod_count,
//         int* order)
//     {
//         const int idx_x = threadIdx.x;
//         const int idx_y = threadIdx.y;
//         const int block_x = blockIdx.x;
//         const int block_y = blockIdx.y;
//         const int block_z = blockIdx.z;

//         const int mod_index = block_z % mod_count;
//         const int input_index = order[block_z];

//         // extern __shared__ T shared_memory[];
//         extern __shared__ char shared_memory_typed[];
//         T* shared_memory = reinterpret_cast<T*>(shared_memory_typed);

//         int t_2 = N_power - logm - 1;
//         location_t offset = 1 << (N_power - logm - 1);
//         int t_ = shared_index;
//         location_t m = (location_t) 1 << logm;

//         location_t global_addresss =
//             idx_x +
//             (location_t) (idx_y *
//                           (offset / (1 << (outer_iteration_count - 1)))) +
//             (location_t) (blockDim.x * block_y) +
//             (location_t) (2 * block_x * offset) +
//             (location_t) (input_index << N_power);
//         location_t omega_addresss =
//             idx_x +
//             (location_t) (idx_y *
//                           (offset / (1 << (outer_iteration_count - 1)))) +
//             (location_t) (blockDim.x * block_y) +
//             (location_t) (block_x * offset);
//         location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

//         // Load UInt64 from global & store to shared
//         shared_memory[shared_addresss] = polynomial_in[global_addresss];
//         shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
//             polynomial_in[global_addresss + offset];

//         int t = 1 << t_;
//         int in_shared_address =
//             ((shared_addresss >> t_) << t_) + shared_addresss;
//         location_t current_root_index;
//         if (not_last_kernel)
//         {
// #pragma unroll
//             for (int lp = 0; lp < outer_iteration_count; lp++)
//             {
//                 __syncthreads();
//                 if (reduction_poly_check)
//                 { // X_N_minus
//                     current_root_index = (omega_addresss >> t_2) +
//                                          (location_t) (mod_index << N_power);
//                 }
//                 else
//                 { // X_N_plus
//                     current_root_index = m + (omega_addresss >> t_2) +
//                                          (location_t) (mod_index << N_power);
//                 }
//                 CooleyTukeyUnit(shared_memory[in_shared_address],
//                                 shared_memory[in_shared_address + t],
//                                 root_of_unity_table[current_root_index],
//                                 modulus[mod_index]);

//                 t = t >> 1;
//                 t_2 -= 1;
//                 t_ -= 1;
//                 m <<= 1;

//                 in_shared_address =
//                     ((shared_addresss >> t_) << t_) + shared_addresss;
//                 //__syncthreads();
//             }
//             __syncthreads();
//         }
//         else
//         {
// #pragma unroll
//             for (int lp = 0; lp < (shared_index - 5); lp++)
//             {
//                 __syncthreads();
//                 if (reduction_poly_check)
//                 { // X_N_minus
//                     current_root_index = (omega_addresss >> t_2) +
//                                          (location_t) (mod_index << N_power);
//                 }
//                 else
//                 { // X_N_plus
//                     current_root_index = m + (omega_addresss >> t_2) +
//                                          (location_t) (mod_index << N_power);
//                 }
//                 CooleyTukeyUnit(shared_memory[in_shared_address],
//                                 shared_memory[in_shared_address + t],
//                                 root_of_unity_table[current_root_index],
//                                 modulus[mod_index]);

//                 t = t >> 1;
//                 t_2 -= 1;
//                 t_ -= 1;
//                 m <<= 1;

//                 in_shared_address =
//                     ((shared_addresss >> t_) << t_) + shared_addresss;
//                 //__syncthreads();
//             }
//             __syncthreads();

// #pragma unroll
//             for (int lp = 0; lp < 6; lp++)
//             {
//                 if (reduction_poly_check)
//                 { // X_N_minus
//                     current_root_index = (omega_addresss >> t_2) +
//                                          (location_t) (mod_index << N_power);
//                 }
//                 else
//                 { // X_N_plus
//                     current_root_index = m + (omega_addresss >> t_2) +
//                                          (location_t) (mod_index << N_power);
//                 }
//                 CooleyTukeyUnit(shared_memory[in_shared_address],
//                                 shared_memory[in_shared_address + t],
//                                 root_of_unity_table[current_root_index],
//                                 modulus[mod_index]);

//                 t = t >> 1;
//                 t_2 -= 1;
//                 t_ -= 1;
//                 m <<= 1;

//                 in_shared_address =
//                     ((shared_addresss >> t_) << t_) + shared_addresss;
//             }
//             __syncthreads();
//         }

//         polynomial_out[global_addresss] = shared_memory[shared_addresss];
//         polynomial_out[global_addresss + offset] =
//             shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
//     }

//     template <typename T>
//     __global__ void
//     InverseCorePolyOrdered(T* polynomial_in, T* polynomial_out,
//                            Root<T>* inverse_root_of_unity_table,
//                            Modulus<T>* modulus, int shared_index, int logm,
//                            int k, int outer_iteration_count, int N_power,
//                            Ninverse<T>* n_inverse, bool last_kernel,
//                            bool reduction_poly_check, int mod_count, int* order)
//     {
//         const int idx_x = threadIdx.x;
//         const int idx_y = threadIdx.y;
//         const int block_x = blockIdx.x;
//         const int block_y = blockIdx.y;
//         const int block_z = blockIdx.z;

//         const int mod_index = block_z % mod_count;
//         const int input_index = order[block_z];

//         // extern __shared__ T shared_memory[];
//         extern __shared__ char shared_memory_typed[];
//         T* shared_memory = reinterpret_cast<T*>(shared_memory_typed);

//         int t_2 = N_power - logm - 1;
//         location_t offset = 1 << (N_power - k - 1);
//         // int t_ = 9 - outer_iteration_count;
//         int t_ = (shared_index + 1) - outer_iteration_count;
//         int loops = outer_iteration_count;
//         location_t m = (location_t) 1 << logm;

//         location_t global_addresss =
//             idx_x +
//             (location_t) (idx_y *
//                           (offset / (1 << (outer_iteration_count - 1)))) +
//             (location_t) (blockDim.x * block_x) +
//             (location_t) (2 * block_y * offset) +
//             (location_t) (input_index << N_power);

//         location_t omega_addresss =
//             idx_x +
//             (location_t) (idx_y *
//                           (offset / (1 << (outer_iteration_count - 1)))) +
//             (location_t) (blockDim.x * block_x) +
//             (location_t) (block_y * offset);
//         location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

//         shared_memory[shared_addresss] = polynomial_in[global_addresss];
//         shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
//             polynomial_in[global_addresss + offset];

//         int t = 1 << t_;
//         int in_shared_address =
//             ((shared_addresss >> t_) << t_) + shared_addresss;
//         location_t current_root_index;
// #pragma unroll
//         for (int lp = 0; lp < loops; lp++)
//         {
//             __syncthreads();
//             if (reduction_poly_check)
//             { // X_N_minus
//                 current_root_index = (omega_addresss >> t_2) +
//                                      (location_t) (mod_index << N_power);
//             }
//             else
//             { // X_N_plus
//                 current_root_index = m + (omega_addresss >> t_2) +
//                                      (location_t) (mod_index << N_power);
//             }

//             GentlemanSandeUnit(shared_memory[in_shared_address],
//                                shared_memory[in_shared_address + t],
//                                inverse_root_of_unity_table[current_root_index],
//                                modulus[mod_index]);

//             t = t << 1;
//             t_2 += 1;
//             t_ += 1;
//             m >>= 1;

//             in_shared_address =
//                 ((shared_addresss >> t_) << t_) + shared_addresss;
//         }
//         __syncthreads();

//         if (last_kernel)
//         {
//             polynomial_out[global_addresss] =
//                 OPERATOR_GPU<T>::mult(shared_memory[shared_addresss],
//                                       n_inverse[mod_index], modulus[mod_index]);
//             polynomial_out[global_addresss + offset] = OPERATOR_GPU<T>::mult(
//                 shared_memory[shared_addresss + (blockDim.x * blockDim.y)],
//                 n_inverse[mod_index], modulus[mod_index]);
//         }
//         else
//         {
//             polynomial_out[global_addresss] = shared_memory[shared_addresss];
//             polynomial_out[global_addresss + offset] =
//                 shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
//         }
//     }

//     template <typename T>
//     __global__ void InverseCorePolyOrdered_(
//         T* polynomial_in, T* polynomial_out,
//         Root<T>* inverse_root_of_unity_table, Modulus<T>* modulus,
//         int shared_index, int logm, int k, int outer_iteration_count,
//         int N_power, Ninverse<T>* n_inverse, bool last_kernel,
//         bool reduction_poly_check, int mod_count, int* order)
//     {
//         const int idx_x = threadIdx.x;
//         const int idx_y = threadIdx.y;
//         const int block_x = blockIdx.x;
//         const int block_y = blockIdx.y;
//         const int block_z = blockIdx.z;

//         const int mod_index = block_z % mod_count;
//         const int input_index = order[block_z];

//         // extern __shared__ T shared_memory[];
//         extern __shared__ char shared_memory_typed[];
//         T* shared_memory = reinterpret_cast<T*>(shared_memory_typed);

//         int t_2 = N_power - logm - 1;
//         location_t offset = 1 << (N_power - k - 1);
//         // int t_ = 9 - outer_iteration_count;
//         int t_ = (shared_index + 1) - outer_iteration_count;
//         int loops = outer_iteration_count;
//         location_t m = (location_t) 1 << logm;

//         location_t global_addresss =
//             idx_x +
//             (location_t) (idx_y *
//                           (offset / (1 << (outer_iteration_count - 1)))) +
//             (location_t) (blockDim.x * block_y) +
//             (location_t) (2 * block_x * offset) +
//             (location_t) (input_index << N_power);

//         location_t omega_addresss =
//             idx_x +
//             (location_t) (idx_y *
//                           (offset / (1 << (outer_iteration_count - 1)))) +
//             (location_t) (blockDim.x * block_y) +
//             (location_t) (block_x * offset);
//         location_t shared_addresss = (idx_x + (idx_y * blockDim.x));

//         shared_memory[shared_addresss] = polynomial_in[global_addresss];
//         shared_memory[shared_addresss + (blockDim.x * blockDim.y)] =
//             polynomial_in[global_addresss + offset];

//         int t = 1 << t_;
//         int in_shared_address =
//             ((shared_addresss >> t_) << t_) + shared_addresss;
//         location_t current_root_index;
// #pragma unroll
//         for (int lp = 0; lp < loops; lp++)
//         {
//             __syncthreads();
//             if (reduction_poly_check)
//             { // X_N_minus
//                 current_root_index = (omega_addresss >> t_2) +
//                                      (location_t) (mod_index << N_power);
//             }
//             else
//             { // X_N_plus
//                 current_root_index = m + (omega_addresss >> t_2) +
//                                      (location_t) (mod_index << N_power);
//             }

//             GentlemanSandeUnit(shared_memory[in_shared_address],
//                                shared_memory[in_shared_address + t],
//                                inverse_root_of_unity_table[current_root_index],
//                                modulus[mod_index]);

//             t = t << 1;
//             t_2 += 1;
//             t_ += 1;
//             m >>= 1;

//             in_shared_address =
//                 ((shared_addresss >> t_) << t_) + shared_addresss;
//         }
//         __syncthreads();

//         if (last_kernel)
//         {
//             polynomial_out[global_addresss] =
//                 OPERATOR_GPU<T>::mult(shared_memory[shared_addresss],
//                                       n_inverse[mod_index], modulus[mod_index]);
//             polynomial_out[global_addresss + offset] = OPERATOR_GPU<T>::mult(
//                 shared_memory[shared_addresss + (blockDim.x * blockDim.y)],
//                 n_inverse[mod_index], modulus[mod_index]);
//         }
//         else
//         {
//             polynomial_out[global_addresss] = shared_memory[shared_addresss];
//             polynomial_out[global_addresss + offset] =
//                 shared_memory[shared_addresss + (blockDim.x * blockDim.y)];
//         }
//     }

//     template <typename T>
//     __host__ void
//     GPU_NTT_Poly_Ordered(T* device_in, T* device_out,
//                          Root<T>* root_of_unity_table, Modulus<T>* modulus,
//                          ntt_rns_configuration<T> cfg, int batch_size,
//                          int mod_count, int* order)
//     {
//         if ((cfg.n_power <= 9 || cfg.n_power >= 29))
//         {
//             throw std::invalid_argument("Invalid n_power range!");
//         }

//         auto kernel_parameters = (cfg.ntt_type == FORWARD)
//                                      ? CreateForwardNTTKernel<T>()
//                                      : CreateInverseNTTKernel<T>();
//         bool standart_kernel = (cfg.n_power < 25) ? true : false;
//         T* device_in_ = device_in;

//         switch (cfg.ntt_type)
//         {
//             case FORWARD:
//                 if (standart_kernel)
//                 {
//                     for (int i = 0; i < kernel_parameters[cfg.n_power].size();
//                          i++)
//                     {
//                         auto& current_kernel_params =
//                             kernel_parameters[cfg.n_power][i];
//                         ForwardCorePolyOrdered<<<
//                             dim3(current_kernel_params.griddim_x,
//                                  current_kernel_params.griddim_y, batch_size),
//                             dim3(current_kernel_params.blockdim_x,
//                                  current_kernel_params.blockdim_y),
//                             current_kernel_params.shared_memory, cfg.stream>>>(
//                             device_in_, device_out, root_of_unity_table,
//                             modulus, current_kernel_params.shared_index,
//                             current_kernel_params.logm,
//                             current_kernel_params.outer_iteration_count,
//                             cfg.n_power, cfg.zero_padding,
//                             current_kernel_params.not_last_kernel,
//                             (cfg.reduction_poly ==
//                              ReductionPolynomial::X_N_minus),
//                             mod_count, order);
//                         GPUNTT_CUDA_CHECK(cudaGetLastError());
//                         device_in_ = device_out;
//                     }
//                 }
//                 else
//                 {
//                     for (int i = 0;
//                          i < kernel_parameters[cfg.n_power].size() - 1; i++)
//                     {
//                         auto& current_kernel_params =
//                             kernel_parameters[cfg.n_power][i];
//                         ForwardCorePolyOrdered<<<
//                             dim3(current_kernel_params.griddim_x,
//                                  current_kernel_params.griddim_y, batch_size),
//                             dim3(current_kernel_params.blockdim_x,
//                                  current_kernel_params.blockdim_y),
//                             current_kernel_params.shared_memory, cfg.stream>>>(
//                             device_in_, device_out, root_of_unity_table,
//                             modulus, current_kernel_params.shared_index,
//                             current_kernel_params.logm,
//                             current_kernel_params.outer_iteration_count,
//                             cfg.n_power, cfg.zero_padding,
//                             current_kernel_params.not_last_kernel,
//                             (cfg.reduction_poly ==
//                              ReductionPolynomial::X_N_minus),
//                             mod_count, order);
//                         GPUNTT_CUDA_CHECK(cudaGetLastError());
//                         device_in_ = device_out;
//                     }
//                     auto& current_kernel_params = kernel_parameters
//                         [cfg.n_power]
//                         [kernel_parameters[cfg.n_power].size() - 1];
//                     ForwardCorePolyOrdered_<<<
//                         dim3(current_kernel_params.griddim_x,
//                              current_kernel_params.griddim_y, batch_size),
//                         dim3(current_kernel_params.blockdim_x,
//                              current_kernel_params.blockdim_y),
//                         current_kernel_params.shared_memory, cfg.stream>>>(
//                         device_in_, device_out, root_of_unity_table, modulus,
//                         current_kernel_params.shared_index,
//                         current_kernel_params.logm,
//                         current_kernel_params.outer_iteration_count,
//                         cfg.n_power, cfg.zero_padding,
//                         current_kernel_params.not_last_kernel,
//                         (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
//                         mod_count, order);
//                     GPUNTT_CUDA_CHECK(cudaGetLastError());
//                 }
//                 break;
//             case INVERSE:
//                 if (standart_kernel)
//                 {
//                     for (int i = 0; i < kernel_parameters[cfg.n_power].size();
//                          i++)
//                     {
//                         auto& current_kernel_params =
//                             kernel_parameters[cfg.n_power][i];
//                         InverseCorePolyOrdered<<<
//                             dim3(current_kernel_params.griddim_x,
//                                  current_kernel_params.griddim_y, batch_size),
//                             dim3(current_kernel_params.blockdim_x,
//                                  current_kernel_params.blockdim_y),
//                             current_kernel_params.shared_memory, cfg.stream>>>(
//                             device_in_, device_out, root_of_unity_table,
//                             modulus, current_kernel_params.shared_index,
//                             current_kernel_params.logm, current_kernel_params.k,
//                             current_kernel_params.outer_iteration_count,
//                             cfg.n_power, cfg.mod_inverse,
//                             current_kernel_params.not_last_kernel,
//                             (cfg.reduction_poly ==
//                              ReductionPolynomial::X_N_minus),
//                             mod_count, order);
//                         GPUNTT_CUDA_CHECK(cudaGetLastError());
//                         device_in_ = device_out;
//                     }
//                 }
//                 else
//                 {
//                     auto& current_kernel_params =
//                         kernel_parameters[cfg.n_power][0];
//                     InverseCorePolyOrdered_<<<
//                         dim3(current_kernel_params.griddim_x,
//                              current_kernel_params.griddim_y, batch_size),
//                         dim3(current_kernel_params.blockdim_x,
//                              current_kernel_params.blockdim_y),
//                         current_kernel_params.shared_memory, cfg.stream>>>(
//                         device_in_, device_out, root_of_unity_table, modulus,
//                         current_kernel_params.shared_index,
//                         current_kernel_params.logm, current_kernel_params.k,
//                         current_kernel_params.outer_iteration_count,
//                         cfg.n_power, cfg.mod_inverse,
//                         current_kernel_params.not_last_kernel,
//                         (cfg.reduction_poly == ReductionPolynomial::X_N_minus),
//                         mod_count, order);
//                     GPUNTT_CUDA_CHECK(cudaGetLastError());
//                     device_in_ = device_out;
//                     for (int i = 1; i < kernel_parameters[cfg.n_power].size();
//                          i++)
//                     {
//                         auto& current_kernel_params =
//                             kernel_parameters[cfg.n_power][i];
//                         InverseCorePolyOrdered<<<
//                             dim3(current_kernel_params.griddim_x,
//                                  current_kernel_params.griddim_y, batch_size),
//                             dim3(current_kernel_params.blockdim_x,
//                                  current_kernel_params.blockdim_y),
//                             current_kernel_params.shared_memory, cfg.stream>>>(
//                             device_in, device_out, root_of_unity_table, modulus,
//                             current_kernel_params.shared_index,
//                             current_kernel_params.logm, current_kernel_params.k,
//                             current_kernel_params.outer_iteration_count,
//                             cfg.n_power, cfg.mod_inverse,
//                             current_kernel_params.not_last_kernel,
//                             (cfg.reduction_poly ==
//                              ReductionPolynomial::X_N_minus),
//                             mod_count, order);
//                         GPUNTT_CUDA_CHECK(cudaGetLastError());
//                     }
//                 }

//                 break;
//             default:
//                 throw std::invalid_argument("Invalid ntt_type!");
//                 break;
//         }
//     }

//     template <typename T>
//     __host__ void GPU_NTT_Poly_Ordered_Inplace(
//         T* device_inout, Root<T>* root_of_unity_table, Modulus<T>* modulus,
//         ntt_rns_configuration<T> cfg, int batch_size, int mod_count, int* order)
//     {
//         GPU_NTT_Poly_Ordered(device_inout, device_inout, root_of_unity_table,
//                              modulus, cfg, batch_size, mod_count, order);
//     }

//     ////////////////////////////////////
//     // Explicit Template Specializations
//     ////////////////////////////////////

//     template <> struct ntt_configuration<Data32>
//     {
//         int n_power;
//         type ntt_type;
//         NTTLayout ntt_layout;
//         ReductionPolynomial reduction_poly;
//         bool zero_padding;
//         Ninverse<Data32> mod_inverse;
//         cudaStream_t stream;
//     };

//     template <> struct ntt_configuration<Data64>
//     {
//         int n_power;
//         type ntt_type;
//         NTTLayout ntt_layout;
//         ReductionPolynomial reduction_poly;
//         bool zero_padding;
//         Ninverse<Data64> mod_inverse;
//         cudaStream_t stream;
//     };

//     template <> struct ntt_rns_configuration<Data32>
//     {
//         int n_power;
//         type ntt_type;
//         NTTLayout ntt_layout;
//         ReductionPolynomial reduction_poly;
//         bool zero_padding;
//         Ninverse<Data32>* mod_inverse;
//         cudaStream_t stream;
//     };

//     template <> struct ntt_rns_configuration<Data64>
//     {
//         int n_power;
//         type ntt_type;
//         NTTLayout ntt_layout;
//         ReductionPolynomial reduction_poly;
//         bool zero_padding;
//         Ninverse<Data64>* mod_inverse;
//         cudaStream_t stream;
//     };

//     template __device__ void
//     CooleyTukeyUnit<Data32>(Data32& U, Data32& V, const Root<Data32>& root,
//                             const Modulus<Data32>& modulus);
//     template __device__ void
//     CooleyTukeyUnit<Data64>(Data64& U, Data64& V, const Root<Data64>& root,
//                             const Modulus<Data64>& modulus);
//     template __device__ void
//     GentlemanSandeUnit<Data32>(Data32& U, Data32& V, const Root<Data32>& root,
//                                const Modulus<Data32>& modulus);
//     template __device__ void
//     GentlemanSandeUnit<Data64>(Data64& U, Data64& V, const Root<Data64>& root,
//                                const Modulus<Data64>& modulus);

//     template __global__ void ForwardCoreLowRing<Data32>(
//         Data32* polynomial_in, Data32* polynomial_out,
//         const Root<Data32>* __restrict__ root_of_unity_table,
//         Modulus<Data32> modulus, int shared_index, int N_power,
//         bool zero_padding, bool reduction_poly_check, int total_batch);

//     template __global__ void ForwardCoreLowRing<Data64>(
//         Data64* polynomial_in, Data64* polynomial_out,
//         const Root<Data64>* __restrict__ root_of_unity_table,
//         Modulus<Data64> modulus, int shared_index, int N_power,
//         bool zero_padding, bool reduction_poly_check, int total_batch);

//     template __global__ void ForwardCoreLowRing<Data32s>(
//         Data32s* polynomial_in, Data32* polynomial_out,
//         const Root<Data32>* __restrict__ root_of_unity_table,
//         Modulus<Data32> modulus, int shared_index, int N_power,
//         bool zero_padding, bool reduction_poly_check, int total_batch);

//     template __global__ void ForwardCoreLowRing<Data64s>(
//         Data64s* polynomial_in, Data64* polynomial_out,
//         const Root<Data64>* __restrict__ root_of_unity_table,
//         Modulus<Data64> modulus, int shared_index, int N_power,
//         bool zero_padding, bool reduction_poly_check, int total_batch);

//     template __global__ void ForwardCoreLowRing<Data32>(
//         Data32* polynomial_in, Data32* polynomial_out,
//         const Root<Data32>* __restrict__ root_of_unity_table,
//         Modulus<Data32>* modulus, int shared_index, int N_power,
//         bool zero_padding, bool reduction_poly_check, int total_batch,
//         int mod_count);

//     template __global__ void ForwardCoreLowRing<Data64>(
//         Data64* polynomial_in, Data64* polynomial_out,
//         const Root<Data64>* __restrict__ root_of_unity_table,
//         Modulus<Data64>* modulus, int shared_index, int N_power,
//         bool zero_padding, bool reduction_poly_check, int total_batch,
//         int mod_count);

//     template __global__ void ForwardCoreLowRing<Data32s>(
//         Data32s* polynomial_in, Data32* polynomial_out,
//         const Root<Data32>* __restrict__ root_of_unity_table,
//         Modulus<Data32>* modulus, int shared_index, int N_power,
//         bool zero_padding, bool reduction_poly_check, int total_batch,
//         int mod_count);

//     template __global__ void ForwardCoreLowRing<Data64s>(
//         Data64s* polynomial_in, Data64* polynomial_out,
//         const Root<Data64>* __restrict__ root_of_unity_table,
//         Modulus<Data64>* modulus, int shared_index, int N_power,
//         bool zero_padding, bool reduction_poly_check, int total_batch,
//         int mod_count);

//     template __global__ void
//     ForwardCore<Data32>(Data32* polynomial_in, Data32* polynomial_out,
//                         const Root<Data32>* __restrict__ root_of_unity_table,
//                         Modulus<Data32> modulus, int shared_index, int logm,
//                         int outer_iteration_count, int N_power,
//                         bool zero_padding, bool not_last_kernel,
//                         bool reduction_poly_check);

//     template __global__ void
//     ForwardCore<Data64>(Data64* polynomial_in, Data64* polynomial_out,
//                         const Root<Data64>* __restrict__ root_of_unity_table,
//                         Modulus<Data64> modulus, int shared_index, int logm,
//                         int outer_iteration_count, int N_power,
//                         bool zero_padding, bool not_last_kernel,
//                         bool reduction_poly_check);

//     template __global__ void
//     ForwardCore<Data32s>(Data32s* polynomial_in, Data32* polynomial_out,
//                          const Root<Data32>* __restrict__ root_of_unity_table,
//                          Modulus<Data32> modulus, int shared_index, int logm,
//                          int outer_iteration_count, int N_power,
//                          bool zero_padding, bool not_last_kernel,
//                          bool reduction_poly_check);

//     template __global__ void
//     ForwardCore<Data64s>(Data64s* polynomial_in, Data64* polynomial_out,
//                          const Root<Data64>* __restrict__ root_of_unity_table,
//                          Modulus<Data64> modulus, int shared_index, int logm,
//                          int outer_iteration_count, int N_power,
//                          bool zero_padding, bool not_last_kernel,
//                          bool reduction_poly_check);

//     template __global__ void
//     ForwardCore<Data32>(Data32* polynomial_in, Data32* polynomial_out,
//                         const Root<Data32>* __restrict__ root_of_unity_table,
//                         Modulus<Data32>* modulus, int shared_index, int logm,
//                         int outer_iteration_count, int N_power,
//                         bool zero_padding, bool not_last_kernel,
//                         bool reduction_poly_check, int mod_count);

//     template __global__ void
//     ForwardCore<Data64>(Data64* polynomial_in, Data64* polynomial_out,
//                         const Root<Data64>* __restrict__ root_of_unity_table,
//                         Modulus<Data64>* modulus, int shared_index, int logm,
//                         int outer_iteration_count, int N_power,
//                         bool zero_padding, bool not_last_kernel,
//                         bool reduction_poly_check, int mod_count);

//     template __global__ void
//     ForwardCore<Data32s>(Data32s* polynomial_in, Data32* polynomial_out,
//                          const Root<Data32>* __restrict__ root_of_unity_table,
//                          Modulus<Data32>* modulus, int shared_index, int logm,
//                          int outer_iteration_count, int N_power,
//                          bool zero_padding, bool not_last_kernel,
//                          bool reduction_poly_check, int mod_count);

//     template __global__ void
//     ForwardCore<Data64s>(Data64s* polynomial_in, Data64* polynomial_out,
//                          const Root<Data64>* __restrict__ root_of_unity_table,
//                          Modulus<Data64>* modulus, int shared_index, int logm,
//                          int outer_iteration_count, int N_power,
//                          bool zero_padding, bool not_last_kernel,
//                          bool reduction_poly_check, int mod_count);

//     template __global__ void
//     ForwardCore_<Data32>(Data32* polynomial_in, Data32* polynomial_out,
//                          const Root<Data32>* __restrict__ root_of_unity_table,
//                          Modulus<Data32> modulus, int shared_index, int logm,
//                          int outer_iteration_count, int N_power,
//                          bool zero_padding, bool not_last_kernel,
//                          bool reduction_poly_check);

//     template __global__ void
//     ForwardCore_<Data64>(Data64* polynomial_in, Data64* polynomial_out,
//                          const Root<Data64>* __restrict__ root_of_unity_table,
//                          Modulus<Data64> modulus, int shared_index, int logm,
//                          int outer_iteration_count, int N_power,
//                          bool zero_padding, bool not_last_kernel,
//                          bool reduction_poly_check);

//     template __global__ void
//     ForwardCore_<Data32s>(Data32s* polynomial_in, Data32* polynomial_out,
//                           const Root<Data32>* __restrict__ root_of_unity_table,
//                           Modulus<Data32> modulus, int shared_index, int logm,
//                           int outer_iteration_count, int N_power,
//                           bool zero_padding, bool not_last_kernel,
//                           bool reduction_poly_check);

//     template __global__ void
//     ForwardCore_<Data64s>(Data64s* polynomial_in, Data64* polynomial_out,
//                           const Root<Data64>* __restrict__ root_of_unity_table,
//                           Modulus<Data64> modulus, int shared_index, int logm,
//                           int outer_iteration_count, int N_power,
//                           bool zero_padding, bool not_last_kernel,
//                           bool reduction_poly_check);

//     template __global__ void
//     ForwardCore_<Data32>(Data32* polynomial_in, Data32* polynomial_out,
//                          const Root<Data32>* __restrict__ root_of_unity_table,
//                          Modulus<Data32>* modulus, int shared_index, int logm,
//                          int outer_iteration_count, int N_power,
//                          bool zero_padding, bool not_last_kernel,
//                          bool reduction_poly_check, int mod_count);

//     template __global__ void
//     ForwardCore_<Data64>(Data64* polynomial_in, Data64* polynomial_out,
//                          const Root<Data64>* __restrict__ root_of_unity_table,
//                          Modulus<Data64>* modulus, int shared_index, int logm,
//                          int outer_iteration_count, int N_power,
//                          bool zero_padding, bool not_last_kernel,
//                          bool reduction_poly_check, int mod_count);

//     template __global__ void
//     ForwardCore_<Data32s>(Data32s* polynomial_in, Data32* polynomial_out,
//                           const Root<Data32>* __restrict__ root_of_unity_table,
//                           Modulus<Data32>* modulus, int shared_index, int logm,
//                           int outer_iteration_count, int N_power,
//                           bool zero_padding, bool not_last_kernel,
//                           bool reduction_poly_check, int mod_count);

//     template __global__ void
//     ForwardCore_<Data64s>(Data64s* polynomial_in, Data64* polynomial_out,
//                           const Root<Data64>* __restrict__ root_of_unity_table,
//                           Modulus<Data64>* modulus, int shared_index, int logm,
//                           int outer_iteration_count, int N_power,
//                           bool zero_padding, bool not_last_kernel,
//                           bool reduction_poly_check, int mod_count);

//     template __global__ void InverseCoreLowRing<Data32>(
//         Data32* polynomial_in, Data32* polynomial_out,
//         const Root<Data32>* __restrict__ inverse_root_of_unity_table,
//         Modulus<Data32> modulus, int shared_index, int N_power,
//         Ninverse<Data32> n_inverse, bool reduction_poly_check, int total_batch);

//     template __global__ void InverseCoreLowRing<Data64>(
//         Data64* polynomial_in, Data64* polynomial_out,
//         const Root<Data64>* __restrict__ inverse_root_of_unity_table,
//         Modulus<Data64> modulus, int shared_index, int N_power,
//         Ninverse<Data64> n_inverse, bool reduction_poly_check, int total_batch);

//     template __global__ void InverseCoreLowRing<Data32s>(
//         Data32* polynomial_in, Data32s* polynomial_out,
//         const Root<Data32>* __restrict__ inverse_root_of_unity_table,
//         Modulus<Data32> modulus, int shared_index, int N_power,
//         Ninverse<Data32> n_inverse, bool reduction_poly_check, int total_batch);

//     template __global__ void InverseCoreLowRing<Data64s>(
//         Data64* polynomial_in, Data64s* polynomial_out,
//         const Root<Data64>* __restrict__ inverse_root_of_unity_table,
//         Modulus<Data64> modulus, int shared_index, int N_power,
//         Ninverse<Data64> n_inverse, bool reduction_poly_check, int total_batch);

//     template __global__ void InverseCoreLowRing<Data32>(
//         Data32* polynomial_in, Data32* polynomial_out,
//         const Root<Data32>* __restrict__ inverse_root_of_unity_table,
//         Modulus<Data32>* modulus, int shared_index, int N_power,
//         Ninverse<Data32>* n_inverse, bool reduction_poly_check, int total_batch,
//         int mod_count);

//     template __global__ void InverseCoreLowRing<Data64>(
//         Data64* polynomial_in, Data64* polynomial_out,
//         const Root<Data64>* __restrict__ inverse_root_of_unity_table,
//         Modulus<Data64>* modulus, int shared_index, int N_power,
//         Ninverse<Data64>* n_inverse, bool reduction_poly_check, int total_batch,
//         int mod_count);

//     template __global__ void InverseCoreLowRing<Data32s>(
//         Data32* polynomial_in, Data32s* polynomial_out,
//         const Root<Data32>* __restrict__ inverse_root_of_unity_table,
//         Modulus<Data32>* modulus, int shared_index, int N_power,
//         Ninverse<Data32>* n_inverse, bool reduction_poly_check, int total_batch,
//         int mod_count);

//     template __global__ void InverseCoreLowRing<Data64s>(
//         Data64* polynomial_in, Data64s* polynomial_out,
//         const Root<Data64>* __restrict__ inverse_root_of_unity_table,
//         Modulus<Data64>* modulus, int shared_index, int N_power,
//         Ninverse<Data64>* n_inverse, bool reduction_poly_check, int total_batch,
//         int mod_count);

//     template __global__ void InverseCore<Data32>(
//         Data32* polynomial_in, Data32* polynomial_out,
//         const Root<Data32>* __restrict__ inverse_root_of_unity_table,
//         Modulus<Data32> modulus, int shared_index, int logm, int k,
//         int outer_iteration_count, int N_power, Ninverse<Data32> n_inverse,
//         bool last_kernel, bool reduction_poly_check);

//     template __global__ void InverseCore<Data64>(
//         Data64* polynomial_in, Data64* polynomial_out,
//         const Root<Data64>* __restrict__ inverse_root_of_unity_table,
//         Modulus<Data64> modulus, int shared_index, int logm, int k,
//         int outer_iteration_count, int N_power, Ninverse<Data64> n_inverse,
//         bool last_kernel, bool reduction_poly_check);

//     template __global__ void InverseCore<Data32s>(
//         Data32* polynomial_in, Data32s* polynomial_out,
//         const Root<Data32>* __restrict__ inverse_root_of_unity_table,
//         Modulus<Data32> modulus, int shared_index, int logm, int k,
//         int outer_iteration_count, int N_power, Ninverse<Data32> n_inverse,
//         bool last_kernel, bool reduction_poly_check);

//     template __global__ void InverseCore<Data64s>(
//         Data64* polynomial_in, Data64s* polynomial_out,
//         const Root<Data64>* __restrict__ inverse_root_of_unity_table,
//         Modulus<Data64> modulus, int shared_index, int logm, int k,
//         int outer_iteration_count, int N_power, Ninverse<Data64> n_inverse,
//         bool last_kernel, bool reduction_poly_check);

//     template __global__ void InverseCore<Data32>(
//         Data32* polynomial_in, Data32* polynomial_out,
//         const Root<Data32>* __restrict__ inverse_root_of_unity_table,
//         Modulus<Data32>* modulus, int shared_index, int logm, int k,
//         int outer_iteration_count, int N_power, Ninverse<Data32>* n_inverse,
//         bool last_kernel, bool reduction_poly_check, int mod_count);

//     template __global__ void InverseCore<Data64>(
//         Data64* polynomial_in, Data64* polynomial_out,
//         const Root<Data64>* __restrict__ inverse_root_of_unity_table,
//         Modulus<Data64>* modulus, int shared_index, int logm, int k,
//         int outer_iteration_count, int N_power, Ninverse<Data64>* n_inverse,
//         bool last_kernel, bool reduction_poly_check, int mod_count);

//     template __global__ void InverseCore<Data32s>(
//         Data32* polynomial_in, Data32s* polynomial_out,
//         const Root<Data32>* __restrict__ inverse_root_of_unity_table,
//         Modulus<Data32>* modulus, int shared_index, int logm, int k,
//         int outer_iteration_count, int N_power, Ninverse<Data32>* n_inverse,
//         bool last_kernel, bool reduction_poly_check, int mod_count);

//     template __global__ void InverseCore<Data64s>(
//         Data64* polynomial_in, Data64s* polynomial_out,
//         const Root<Data64>* __restrict__ inverse_root_of_unity_table,
//         Modulus<Data64>* modulus, int shared_index, int logm, int k,
//         int outer_iteration_count, int N_power, Ninverse<Data64>* n_inverse,
//         bool last_kernel, bool reduction_poly_check, int mod_count);

//     template __global__ void InverseCore_<Data32>(
//         Data32* polynomial_in, Data32* polynomial_out,
//         const Root<Data32>* __restrict__ inverse_root_of_unity_table,
//         Modulus<Data32> modulus, int shared_index, int logm, int k,
//         int outer_iteration_count, int N_power, Ninverse<Data32> n_inverse,
//         bool last_kernel, bool reduction_poly_check);

//     template __global__ void InverseCore_<Data64>(
//         Data64* polynomial_in, Data64* polynomial_out,
//         const Root<Data64>* __restrict__ inverse_root_of_unity_table,
//         Modulus<Data64> modulus, int shared_index, int logm, int k,
//         int outer_iteration_count, int N_power, Ninverse<Data64> n_inverse,
//         bool last_kernel, bool reduction_poly_check);

//     template __global__ void InverseCore_<Data32s>(
//         Data32* polynomial_in, Data32s* polynomial_out,
//         const Root<Data32>* __restrict__ inverse_root_of_unity_table,
//         Modulus<Data32> modulus, int shared_index, int logm, int k,
//         int outer_iteration_count, int N_power, Ninverse<Data32> n_inverse,
//         bool last_kernel, bool reduction_poly_check);

//     template __global__ void InverseCore_<Data64s>(
//         Data64* polynomial_in, Data64s* polynomial_out,
//         const Root<Data64>* __restrict__ inverse_root_of_unity_table,
//         Modulus<Data64> modulus, int shared_index, int logm, int k,
//         int outer_iteration_count, int N_power, Ninverse<Data64> n_inverse,
//         bool last_kernel, bool reduction_poly_check);

//     template __global__ void InverseCore_<Data32>(
//         Data32* polynomial_in, Data32* polynomial_out,
//         const Root<Data32>* __restrict__ inverse_root_of_unity_table,
//         Modulus<Data32>* modulus, int shared_index, int logm, int k,
//         int outer_iteration_count, int N_power, Ninverse<Data32>* n_inverse,
//         bool last_kernel, bool reduction_poly_check, int mod_count);

//     template __global__ void InverseCore_<Data64>(
//         Data64* polynomial_in, Data64* polynomial_out,
//         const Root<Data64>* __restrict__ inverse_root_of_unity_table,
//         Modulus<Data64>* modulus, int shared_index, int logm, int k,
//         int outer_iteration_count, int N_power, Ninverse<Data64>* n_inverse,
//         bool last_kernel, bool reduction_poly_check, int mod_count);

//     template __global__ void InverseCore_<Data32s>(
//         Data32* polynomial_in, Data32s* polynomial_out,
//         const Root<Data32>* __restrict__ inverse_root_of_unity_table,
//         Modulus<Data32>* modulus, int shared_index, int logm, int k,
//         int outer_iteration_count, int N_power, Ninverse<Data32>* n_inverse,
//         bool last_kernel, bool reduction_poly_check, int mod_count);

//     template __global__ void InverseCore_<Data64s>(
//         Data64* polynomial_in, Data64s* polynomial_out,
//         const Root<Data64>* __restrict__ inverse_root_of_unity_table,
//         Modulus<Data64>* modulus, int shared_index, int logm, int k,
//         int outer_iteration_count, int N_power, Ninverse<Data64>* n_inverse,
//         bool last_kernel, bool reduction_poly_check, int mod_count);

//     template __global__ void ForwardCoreTranspose<Data32>(
//         Data32* polynomial_in, Data32* polynomial_out,
//         const Root<Data32>* __restrict__ root_of_unity_table,
//         const Modulus<Data32> modulus, int log_row, int log_column,
//         bool reduction_poly_check);

//     template __global__ void ForwardCoreTranspose<Data64>(
//         Data64* polynomial_in, Data64* polynomial_out,
//         const Root<Data64>* __restrict__ root_of_unity_table,
//         const Modulus<Data64> modulus, int log_row, int log_column,
//         bool reduction_poly_check);

//     template __global__ void ForwardCoreTranspose<Data32s>(
//         Data32s* polynomial_in, Data32* polynomial_out,
//         const Root<Data32>* __restrict__ root_of_unity_table,
//         const Modulus<Data32> modulus, int log_row, int log_column,
//         bool reduction_poly_check);

//     template __global__ void ForwardCoreTranspose<Data64s>(
//         Data64s* polynomial_in, Data64* polynomial_out,
//         const Root<Data64>* __restrict__ root_of_unity_table,
//         const Modulus<Data64> modulus, int log_row, int log_column,
//         bool reduction_poly_check);

//     template __global__ void ForwardCoreTranspose<Data32>(
//         Data32* polynomial_in, Data32* polynomial_out,
//         const Root<Data32>* __restrict__ root_of_unity_table,
//         const Modulus<Data32>* modulus, int log_row, int log_column,
//         bool reduction_poly_check, int mod_count);

    template void ForwardCoreTranspose<Data64>(
        int total_block_thread, int total_block_count,
        int blockdim_x, int blockdim_y,
        Data64* polynomial_in, Data64* polynomial_out,
        const Root<Data64>* __restrict__ root_of_unity_table,
        const Modulus<Data64>* modulus, int log_row, int log_column,
        bool reduction_poly_check, int mod_count);

//     template __global__ void ForwardCoreTranspose<Data32s>(
//         Data32s* polynomial_in, Data32* polynomial_out,
//         const Root<Data32>* __restrict__ root_of_unity_table,
//         const Modulus<Data32>* modulus, int log_row, int log_column,
//         bool reduction_poly_check, int mod_count);

//     template __global__ void ForwardCoreTranspose<Data64s>(
//         Data64s* polynomial_in, Data64* polynomial_out,
//         const Root<Data64>* __restrict__ root_of_unity_table,
//         const Modulus<Data64>* modulus, int log_row, int log_column,
//         bool reduction_poly_check, int mod_count);

//     template __global__ void InverseCoreTranspose<Data32>(
//         Data32* polynomial_in, Data32* polynomial_out,
//         const Root<Data32>* __restrict__ inverse_root_of_unity_table,
//         Modulus<Data32> modulus, Ninverse<Data32> n_inverse, int log_row,
//         int log_column, bool reduction_poly_check);

//     template __global__ void InverseCoreTranspose<Data64>(
//         Data64* polynomial_in, Data64* polynomial_out,
//         const Root<Data64>* __restrict__ inverse_root_of_unity_table,
//         Modulus<Data64> modulus, Ninverse<Data64> n_inverse, int log_row,
//         int log_column, bool reduction_poly_check);

//     template __global__ void InverseCoreTranspose<Data32s>(
//         Data32* polynomial_in, Data32s* polynomial_out,
//         const Root<Data32>* __restrict__ inverse_root_of_unity_table,
//         Modulus<Data32> modulus, Ninverse<Data32> n_inverse, int log_row,
//         int log_column, bool reduction_poly_check);

//     template __global__ void InverseCoreTranspose<Data64s>(
//         Data64* polynomial_in, Data64s* polynomial_out,
//         const Root<Data64>* __restrict__ inverse_root_of_unity_table,
//         Modulus<Data64> modulus, Ninverse<Data64> n_inverse, int log_row,
//         int log_column, bool reduction_poly_check);

//     template __global__ void InverseCoreTranspose<Data32>(
//         Data32* polynomial_in, Data32* polynomial_out,
//         const Root<Data32>* __restrict__ inverse_root_of_unity_table,
//         Modulus<Data32>* modulus, Ninverse<Data32>* n_inverse, int log_row,
//         int log_column, bool reduction_poly_check, int mod_count);

//     template __global__ void InverseCoreTranspose<Data64>(
//         Data64* polynomial_in, Data64* polynomial_out,
//         const Root<Data64>* __restrict__ inverse_root_of_unity_table,
//         Modulus<Data64>* modulus, Ninverse<Data64>* n_inverse, int log_row,
//         int log_column, bool reduction_poly_check, int mod_count);

//     template __global__ void InverseCoreTranspose<Data32s>(
//         Data32* polynomial_in, Data32s* polynomial_out,
//         const Root<Data32>* __restrict__ inverse_root_of_unity_table,
//         Modulus<Data32>* modulus, Ninverse<Data32>* n_inverse, int log_row,
//         int log_column, bool reduction_poly_check, int mod_count);

//     template __global__ void InverseCoreTranspose<Data64s>(
//         Data64* polynomial_in, Data64s* polynomial_out,
//         const Root<Data64>* __restrict__ inverse_root_of_unity_table,
//         Modulus<Data64>* modulus, Ninverse<Data64>* n_inverse, int log_row,
//         int log_column, bool reduction_poly_check, int mod_count);

//     template __host__ void
//     GPU_NTT<Data32>(Data32* device_in, Data32* device_out,
//                     Root<Data32>* root_of_unity_table, Modulus<Data32> modulus,
//                     ntt_configuration<Data32> cfg, int batch_size);

//     template __host__ void
//     GPU_NTT<Data64>(Data64* device_in, Data64* device_out,
//                     Root<Data64>* root_of_unity_table, Modulus<Data64> modulus,
//                     ntt_configuration<Data64> cfg, int batch_size);

//     template __host__ void
//     GPU_NTT<Data32s>(Data32s* device_in, Data32* device_out,
//                      Root<Data32>* root_of_unity_table, Modulus<Data32> modulus,
//                      ntt_configuration<Data32> cfg, int batch_size);

//     template __host__ void
//     GPU_NTT<Data64s>(Data64s* device_in, Data64* device_out,
//                      Root<Data64>* root_of_unity_table, Modulus<Data64> modulus,
//                      ntt_configuration<Data64> cfg, int batch_size);

//     template __host__ void
//     GPU_INTT<Data32>(Data32* device_in, Data32* device_out,
//                      Root<Data32>* root_of_unity_table, Modulus<Data32> modulus,
//                      ntt_configuration<Data32> cfg, int batch_size);

//     template __host__ void
//     GPU_INTT<Data64>(Data64* device_in, Data64* device_out,
//                      Root<Data64>* root_of_unity_table, Modulus<Data64> modulus,
//                      ntt_configuration<Data64> cfg, int batch_size);

//     template __host__ void GPU_INTT<Data32s>(Data32* device_in,
//                                              Data32s* device_out,
//                                              Root<Data32>* root_of_unity_table,
//                                              Modulus<Data32> modulus,
//                                              ntt_configuration<Data32> cfg,
//                                              int batch_size);

//     template __host__ void GPU_INTT<Data64s>(Data64* device_in,
//                                              Data64s* device_out,
//                                              Root<Data64>* root_of_unity_table,
//                                              Modulus<Data64> modulus,
//                                              ntt_configuration<Data64> cfg,
//                                              int batch_size);

//     template __host__ void GPU_NTT_Inplace<Data32>(
//         Data32* device_inout, Root<Data32>* root_of_unity_table,
//         Modulus<Data32> modulus, ntt_configuration<Data32> cfg, int batch_size);

//     template __host__ void GPU_NTT_Inplace<Data64>(
//         Data64* device_inout, Root<Data64>* root_of_unity_table,
//         Modulus<Data64> modulus, ntt_configuration<Data64> cfg, int batch_size);

//     template __host__ void GPU_INTT_Inplace<Data32>(
//         Data32* device_inout, Root<Data32>* root_of_unity_table,
//         Modulus<Data32> modulus, ntt_configuration<Data32> cfg, int batch_size);

//     template __host__ void GPU_INTT_Inplace<Data64>(
//         Data64* device_inout, Root<Data64>* root_of_unity_table,
//         Modulus<Data64> modulus, ntt_configuration<Data64> cfg, int batch_size);

//     template __host__ void GPU_NTT<Data32>(Data32* device_in,
//                                            Data32* device_out,
//                                            Root<Data32>* root_of_unity_table,
//                                            Modulus<Data32>* modulus,
//                                            ntt_rns_configuration<Data32> cfg,
//                                            int batch_size, int mod_count);

    template void GPU_NTT<Data64>(Data64* device_in,
                                    Data64* device_out,
                                    Root<Data64>* root_of_unity_table,
                                    Modulus<Data64>* modulus,
                                    ntt_rns_configuration<Data64> cfg,
                                    int batch_size, int mod_count);

//     template __host__ void GPU_NTT<Data32s>(Data32s* device_in,
//                                             Data32* device_out,
//                                             Root<Data32>* root_of_unity_table,
//                                             Modulus<Data32>* modulus,
//                                             ntt_rns_configuration<Data32> cfg,
//                                             int batch_size, int mod_count);

//     template __host__ void GPU_NTT<Data64s>(Data64s* device_in,
//                                             Data64* device_out,
//                                             Root<Data64>* root_of_unity_table,
//                                             Modulus<Data64>* modulus,
//                                             ntt_rns_configuration<Data64> cfg,
//                                             int batch_size, int mod_count);

//     template __host__ void GPU_INTT<Data32>(Data32* device_in,
//                                             Data32* device_out,
//                                             Root<Data32>* root_of_unity_table,
//                                             Modulus<Data32>* modulus,
//                                             ntt_rns_configuration<Data32> cfg,
//                                             int batch_size, int mod_count);

//     template __host__ void GPU_INTT<Data64>(Data64* device_in,
//                                             Data64* device_out,
//                                             Root<Data64>* root_of_unity_table,
//                                             Modulus<Data64>* modulus,
//                                             ntt_rns_configuration<Data64> cfg,
//                                             int batch_size, int mod_count);

//     template __host__ void GPU_INTT<Data32s>(Data32* device_in,
//                                              Data32s* device_out,
//                                              Root<Data32>* root_of_unity_table,
//                                              Modulus<Data32>* modulus,
//                                              ntt_rns_configuration<Data32> cfg,
//                                              int batch_size, int mod_count);

//     template __host__ void GPU_INTT<Data64s>(Data64* device_in,
//                                              Data64s* device_out,
//                                              Root<Data64>* root_of_unity_table,
//                                              Modulus<Data64>* modulus,
//                                              ntt_rns_configuration<Data64> cfg,
//                                              int batch_size, int mod_count);

//     template __host__ void GPU_NTT_Inplace<Data32>(
//         Data32* device_inout, Root<Data32>* root_of_unity_table,
//         Modulus<Data32>* modulus, ntt_rns_configuration<Data32> cfg,
//         int batch_size, int mod_count);

    template void GPU_NTT_Inplace<Data64>(
        Data64* device_inout, Root<Data64>* root_of_unity_table,
        Modulus<Data64>* modulus, ntt_rns_configuration<Data64> cfg,
        int batch_size, int mod_count);

//     template __host__ void GPU_INTT_Inplace<Data32>(
//         Data32* device_inout, Root<Data32>* root_of_unity_table,
//         Modulus<Data32>* modulus, ntt_rns_configuration<Data32> cfg,
//         int batch_size, int mod_count);

//     template __host__ void GPU_INTT_Inplace<Data64>(
//         Data64* device_inout, Root<Data64>* root_of_unity_table,
//         Modulus<Data64>* modulus, ntt_rns_configuration<Data64> cfg,
//         int batch_size, int mod_count);

//     template __global__ void ForwardCoreModulusOrdered<Data32>(
//         Data32* polynomial_in, Data32* polynomial_out,
//         Root<Data32>* root_of_unity_table, Modulus<Data32>* modulus,
//         int shared_index, int logm, int outer_iteration_count, int N_power,
//         bool zero_padding, bool not_last_kernel, bool reduction_poly_check,
//         int mod_count, int* order);

//     template __global__ void ForwardCoreModulusOrdered_<Data32>(
//         Data32* polynomial_in, Data32* polynomial_out,
//         Root<Data32>* root_of_unity_table, Modulus<Data32>* modulus,
//         int shared_index, int logm, int outer_iteration_count, int N_power,
//         bool zero_padding, bool not_last_kernel, bool reduction_poly_check,
//         int mod_count, int* order);

//     template __global__ void ForwardCoreModulusOrdered<Data64>(
//         Data64* polynomial_in, Data64* polynomial_out,
//         Root<Data64>* root_of_unity_table, Modulus<Data64>* modulus,
//         int shared_index, int logm, int outer_iteration_count, int N_power,
//         bool zero_padding, bool not_last_kernel, bool reduction_poly_check,
//         int mod_count, int* order);

//     template __global__ void ForwardCoreModulusOrdered_<Data64>(
//         Data64* polynomial_in, Data64* polynomial_out,
//         Root<Data64>* root_of_unity_table, Modulus<Data64>* modulus,
//         int shared_index, int logm, int outer_iteration_count, int N_power,
//         bool zero_padding, bool not_last_kernel, bool reduction_poly_check,
//         int mod_count, int* order);

//     template __global__ void InverseCoreModulusOrdered<Data32>(
//         Data32* polynomial_in, Data32* polynomial_out,
//         Root<Data32>* inverse_root_of_unity_table, Modulus<Data32>* modulus,
//         int shared_index, int logm, int k, int outer_iteration_count,
//         int N_power, Ninverse<Data32>* n_inverse, bool last_kernel,
//         bool reduction_poly_check, int mod_count, int* order);

//     template __global__ void InverseCoreModulusOrdered_<Data32>(
//         Data32* polynomial_in, Data32* polynomial_out,
//         Root<Data32>* inverse_root_of_unity_table, Modulus<Data32>* modulus,
//         int shared_index, int logm, int k, int outer_iteration_count,
//         int N_power, Ninverse<Data32>* n_inverse, bool last_kernel,
//         bool reduction_poly_check, int mod_count, int* order);

//     template __global__ void InverseCoreModulusOrdered<Data64>(
//         Data64* polynomial_in, Data64* polynomial_out,
//         Root<Data64>* inverse_root_of_unity_table, Modulus<Data64>* modulus,
//         int shared_index, int logm, int k, int outer_iteration_count,
//         int N_power, Ninverse<Data64>* n_inverse, bool last_kernel,
//         bool reduction_poly_check, int mod_count, int* order);

//     template __global__ void InverseCoreModulusOrdered_<Data64>(
//         Data64* polynomial_in, Data64* polynomial_out,
//         Root<Data64>* inverse_root_of_unity_table, Modulus<Data64>* modulus,
//         int shared_index, int logm, int k, int outer_iteration_count,
//         int N_power, Ninverse<Data64>* n_inverse, bool last_kernel,
//         bool reduction_poly_check, int mod_count, int* order);

//     template __host__ void
//     GPU_NTT_Modulus_Ordered<Data32>(Data32* device_in, Data32* device_out,
//                                     Root<Data32>* root_of_unity_table,
//                                     Modulus<Data32>* modulus,
//                                     ntt_rns_configuration<Data32> cfg,
//                                     int batch_size, int mod_count, int* order);

//     template __host__ void GPU_NTT_Modulus_Ordered_Inplace<Data32>(
//         Data32* device_inout, Root<Data32>* root_of_unity_table,
//         Modulus<Data32>* modulus, ntt_rns_configuration<Data32> cfg,
//         int batch_size, int mod_count, int* order);

//     template __host__ void
//     GPU_NTT_Modulus_Ordered<Data64>(Data64* device_in, Data64* device_out,
//                                     Root<Data64>* root_of_unity_table,
//                                     Modulus<Data64>* modulus,
//                                     ntt_rns_configuration<Data64> cfg,
//                                     int batch_size, int mod_count, int* order);

//     template __host__ void GPU_NTT_Modulus_Ordered_Inplace<Data64>(
//         Data64* device_inout, Root<Data64>* root_of_unity_table,
//         Modulus<Data64>* modulus, ntt_rns_configuration<Data64> cfg,
//         int batch_size, int mod_count, int* order);

//     template __global__ void ForwardCorePolyOrdered<Data32>(
//         Data32* polynomial_in, Data32* polynomial_out,
//         Root<Data32>* root_of_unity_table, Modulus<Data32>* modulus,
//         int shared_index, int logm, int outer_iteration_count, int N_power,
//         bool zero_padding, bool not_last_kernel, bool reduction_poly_check,
//         int mod_count, int* order);

//     template __global__ void ForwardCorePolyOrdered_<Data32>(
//         Data32* polynomial_in, Data32* polynomial_out,
//         Root<Data32>* root_of_unity_table, Modulus<Data32>* modulus,
//         int shared_index, int logm, int outer_iteration_count, int N_power,
//         bool zero_padding, bool not_last_kernel, bool reduction_poly_check,
//         int mod_count, int* order);

//     template __global__ void ForwardCorePolyOrdered<Data64>(
//         Data64* polynomial_in, Data64* polynomial_out,
//         Root<Data64>* root_of_unity_table, Modulus<Data64>* modulus,
//         int shared_index, int logm, int outer_iteration_count, int N_power,
//         bool zero_padding, bool not_last_kernel, bool reduction_poly_check,
//         int mod_count, int* order);

//     template __global__ void ForwardCorePolyOrdered_<Data64>(
//         Data64* polynomial_in, Data64* polynomial_out,
//         Root<Data64>* root_of_unity_table, Modulus<Data64>* modulus,
//         int shared_index, int logm, int outer_iteration_count, int N_power,
//         bool zero_padding, bool not_last_kernel, bool reduction_poly_check,
//         int mod_count, int* order);

//     template __global__ void InverseCorePolyOrdered<Data32>(
//         Data32* polynomial_in, Data32* polynomial_out,
//         Root<Data32>* inverse_root_of_unity_table, Modulus<Data32>* modulus,
//         int shared_index, int logm, int k, int outer_iteration_count,
//         int N_power, Ninverse<Data32>* n_inverse, bool last_kernel,
//         bool reduction_poly_check, int mod_count, int* order);

//     template __global__ void InverseCorePolyOrdered_<Data32>(
//         Data32* polynomial_in, Data32* polynomial_out,
//         Root<Data32>* inverse_root_of_unity_table, Modulus<Data32>* modulus,
//         int shared_index, int logm, int k, int outer_iteration_count,
//         int N_power, Ninverse<Data32>* n_inverse, bool last_kernel,
//         bool reduction_poly_check, int mod_count, int* order);

//     template __global__ void InverseCorePolyOrdered<Data64>(
//         Data64* polynomial_in, Data64* polynomial_out,
//         Root<Data64>* inverse_root_of_unity_table, Modulus<Data64>* modulus,
//         int shared_index, int logm, int k, int outer_iteration_count,
//         int N_power, Ninverse<Data64>* n_inverse, bool last_kernel,
//         bool reduction_poly_check, int mod_count, int* order);

//     template __global__ void InverseCorePolyOrdered_<Data64>(
//         Data64* polynomial_in, Data64* polynomial_out,
//         Root<Data64>* inverse_root_of_unity_table, Modulus<Data64>* modulus,
//         int shared_index, int logm, int k, int outer_iteration_count,
//         int N_power, Ninverse<Data64>* n_inverse, bool last_kernel,
//         bool reduction_poly_check, int mod_count, int* order);

//     template __host__ void
//     GPU_NTT_Poly_Ordered<Data32>(Data32* device_in, Data32* device_out,
//                                  Root<Data32>* root_of_unity_table,
//                                  Modulus<Data32>* modulus,
//                                  ntt_rns_configuration<Data32> cfg,
//                                  int batch_size, int mod_count, int* order);

//     template __host__ void GPU_NTT_Poly_Ordered_Inplace<Data32>(
//         Data32* device_inout, Root<Data32>* root_of_unity_table,
//         Modulus<Data32>* modulus, ntt_rns_configuration<Data32> cfg,
//         int batch_size, int mod_count, int* order);

//     template __host__ void
//     GPU_NTT_Poly_Ordered<Data64>(Data64* device_in, Data64* device_out,
//                                  Root<Data64>* root_of_unity_table,
//                                  Modulus<Data64>* modulus,
//                                  ntt_rns_configuration<Data64> cfg,
//                                  int batch_size, int mod_count, int* order);

//     template __host__ void GPU_NTT_Poly_Ordered_Inplace<Data64>(
//         Data64* device_inout, Root<Data64>* root_of_unity_table,
//         Modulus<Data64>* modulus, ntt_rns_configuration<Data64> cfg,
//         int batch_size, int mod_count, int* order);

} // namespace gpuntt