#pragma once
#include <math.h>
#include <cinttypes>
#include <string>
#include <vector>
// #include "cuda_runtime.h"
// #include "device_launch_parameters.h"

namespace heoncpu{

typedef unsigned Data32;
typedef unsigned Root32;
typedef unsigned Ninverse32;

typedef unsigned long long Data64;
typedef unsigned long long Root64;
typedef unsigned long long Ninverse64;
 

template <typename T1> struct Modulus
{
    T1 value;
    T1 bit;
    T1 mu;

    // Constructor to initialize the Modulus
    Modulus(T1 mod) : value(mod)
    {
        bit = bit_generator();
        mu = mu_generator();
    }

    Modulus() : value(0), bit(0), mu(0) {}

  private:
    T1 bit_generator() const
    {
        return static_cast<T1>(log2(value) + 1);
    }

    T1 mu_generator() const
    {
        using T2 = typename std::conditional<std::is_same<T1, Data32>::value,
                                             Data64, __uint128_t>::type;
        T2 mu_ = static_cast<T2>(1) << ((2 * bit) + 1);
        mu_ = mu_ / value;
        return static_cast<T1>(mu_);
    }
};

typedef Modulus<Data32> Modulus32;
typedef Modulus<Data64> Modulus64;

namespace modular_operation_cpu
{
    template <typename T1> class BarrettOperations
    {
        // It does not work  modulus higher than 30 bit for Data32.
        // It does not work  modulus higher than 62 bit for Data64.
      public:
        // Modular Addition
        // result = (input1 + input2) % modulus
        static  T1 add(const T1& input1, const T1& input2,
                               const Modulus<T1>& modulus)
        {
            T1 sum = input1 + input2;
            return (sum >= modulus.value) ? (sum - modulus.value) : sum;
        }

        // Modular Subtraction
        // result = (input1 - input2) % modulus
        static  T1 sub(const T1& input1, const T1& input2,
                               const Modulus<T1>& modulus)
        {
            T1 dif = input1 + modulus.value;
            dif = dif - input2;
            return (dif >= modulus.value) ? (dif - modulus.value) : dif;
        }

        // Modular Multiplication
        // result = (input1 * input2) % modulus
        static  T1 mult(const T1& input1, const T1& input2,
                                const Modulus<T1>& modulus)
        {
            using T2 =
                typename std::conditional<std::is_same<T1, Data32>::value,
                                          Data64, __uint128_t>::type;
            T2 mult = static_cast<T2>(input1) * static_cast<T2>(input2);
            T2 r = mult >> (modulus.bit - 2);
            r = r * static_cast<T2>(modulus.mu);
            r = r >> (modulus.bit + 3);
            r = r * static_cast<T2>(modulus.value);
            mult = mult - r;
            T1 result = static_cast<T1>(
                mult &
                (std::is_same<T1, Data32>::value ? UINT32_MAX : UINT64_MAX));
            return (result >= modulus.value) ? (result - modulus.value)
                                             : result;
        }
        

        // Modular Exponentiation
        // result = (base ^ exponent) % modulus
        static  T1 exp(T1 base, T1 exponent, const Modulus<T1>& modulus)
        {
            T1 result = 1;
            if (exponent == 0)
            {
                return result;
            }
            int exponent_bit = static_cast<int>(log2(exponent) + 1);
            for (int i = exponent_bit - 1; i >= 0; i--)
            {
                result = mult(result, result, modulus);
                if ((exponent >> i) & 1)
                {
                    result = mult(result, base, modulus);
                }
            }
            return result;
        }

        // Modular Inverse
        // result = (input ^ (-1)) % modulus
        static  T1 modinv(T1 input, const Modulus<T1>& modulus)
        {
            T1 index = modulus.value - 2;
            return exp(input, index, modulus);
        }

        // Modular Reduction
        static T1 reduce(const T1& input1, const Modulus<T1>& modulus)
        {
            using T2 =
                typename std::conditional<std::is_same<T1, Data32>::value,
                                          Data64, __uint128_t>::type;
            T2 mult = static_cast<T2>(input1);
            T2 r = mult >> (modulus.bit - 2);
            r = r * static_cast<T2>(modulus.mu);
            r = r >> (modulus.bit + 3);
            r = r * static_cast<T2>(modulus.value);
            mult = mult - r;
            T1 result = static_cast<T1>(
                mult &
                (std::is_same<T1, Data32>::value ? UINT32_MAX : UINT64_MAX));
            return (result >= modulus.value) ? (result - modulus.value)
                                             : result;
        }

        // Forced Reduction (Repeated Reduction Until Input < Modulus)
        // result = input1 % modulus
        static T1 reduce_forced(const T1& input, const Modulus<T1>& modulus)
        {
            T1 result = input;
            while (result >= modulus.value)
            {
                result = reduce(result, modulus);
            }
            return result;
        }
    };

} // namespace modular_operation_cpu

template <typename T>
using OPERATOR = modular_operation_cpu::BarrettOperations<T>;

typedef OPERATOR<Data32> OPERATOR32;
typedef OPERATOR<Data64> OPERATOR64;

template <typename T>
using Root = typename std::conditional<std::is_same<T, Data32>::value, Root32,
                                       Root64>::type;

template <typename T>
using Ninverse = typename std::conditional<std::is_same<T, Data32>::value,
                                           Ninverse32, Ninverse64>::type;

namespace modular_operation_gpu
{
    struct ulonglong2 {
        Data64 x;
        Data64 y;
        // ... 其他成员和方法
    };

    template <typename T1> class BarrettOperations
    {
        // It does not work  modulus higher than 30 bit for Data32.
        // It does not work  modulus higher than 62 bit for Data64.
      private:
        class uint128_t
        {
          public:
            // x -> LSB side
            // y -> MSB side
            ulonglong2 value;

            uint128_t()
            {
                value.x = 0;
                value.y = 0;
            }
 
            uint128_t(const Data64 input)
            {
                value.x = input;
                value.y = 0;
            }

            uint128_t(const Data64 low,const Data64 high)
            {
                value.x = low;
                value.y = high;
            }
 

            void operator=(const uint128_t input)
            {
                value.x = input.value.x;
                value.y = input.value.y;
            }

            void operator=(const Data64 input)
            {
                value.x = input;
                value.y = 0;
            }

            uint128_t
            operator<<(const unsigned shift)
            {
                uint128_t result;

                result.value.y = value.y << shift;
                result.value.y = (value.x >> (64 - shift)) | result.value.y;
                result.value.x = value.x << shift;

                return result;
            }

            uint128_t
            operator>>(const unsigned shift)
            {
                uint128_t result;

                result.value.x = value.x >> shift;
                result.value.x = (value.y << (64 - shift)) | result.value.x;
                result.value.y = value.y >> shift;

                return result;
            }

            // uint128_t operator-(uint128_t& other)
            // {
            //     uint128_t result;

            //     asm("{\n\t"
            //         "sub.cc.u64      %1, %3, %5;    \n\t"
            //         "subc.u64        %0, %2, %4;    \n\t"
            //         "}"
            //         : "=l"(result.value.y), "=l"(result.value.x)
            //         : "l"(value.y), "l"(value.x), "l"(other.value.y),
            //           "l"(other.value.x));

            //     return result;
            // }
            uint128_t operator-(uint128_t& other) const {
                uint128_t result;
                
                // 先计算低64位的差
                result.value.x = this->value.x - other.value.x;
                
                // 检查低64位是否发生借位
                bool borrow = this->value.x < result.value.x;
                
                // 计算高64位的差，并考虑借位
                result.value.y = this->value.y - other.value.y - (borrow?1:0);
                
                return result;
            }
            // uint128_t
            // operator-=(const uint128_t& other)
            // {
            //     uint128_t result;
            //     asm("{\n\t"
            //         "sub.cc.u64      %1, %3, %5;    \n\t"
            //         "subc.u64        %0, %2, %4;    \n\t"
            //         "}"
            //         : "=l"(result.value.y), "=l"(result.value.x)
            //         : "l"(value.y), "l"(value.x), "l"(other.value.y),
            //           "l"(other.value.x));

            //     return result;
            // }
        };

      public:
        // Modular Addition
        // result = (input1 + input2) % modulus
        static T1 add(const T1& input1,
                                                 const T1& input2,
                                                 const Modulus<T1>& modulus)
        {
            T1 sum = input1 + input2;
            return (sum >= modulus.value) ? (sum - modulus.value) : sum;
        }

        // Modular Subtraction
        // result = (input1 - input2) % modulus
        static T1 sub(const T1& input1,
                                                 const T1& input2,
                                                 const Modulus<T1>& modulus)
        {
            T1 dif = input1 + modulus.value;
            dif = dif - input2;
            return (dif >= modulus.value) ? (dif - modulus.value) : dif;
        }

        // 64-bit Multiplication for 32-bit T1
        static Data64 mult64(const Data32& a,
                                                        const Data32& b)
        {
            return static_cast<Data64>(a) * static_cast<Data64>(b);
        }

        // 128-bit Multiplication for 64-bit T1
        // static uint128_t mult128(const Data64& a,
        //                                                     const Data64& b)
        // {
        //     uint128_t result;
        //     asm("{\n\t"
        //         "mul.lo.u64      %1, %2, %3;    \n\t"
        //         "mul.hi.u64      %0, %2, %3;    \n\t"
        //         "}"
        //         : "=l"(result.value.y), "=l"(result.value.x)
        //         : "l"(a), "l"(b));
        //     return result;
        // }
        static uint128_t mult128(const Data64& a, const Data64& b)
        {
            uint128_t result;
            
            // 将64位数拆分为高32位和低32位
            uint32_t a_high = a >> 32;
            uint32_t a_low = a & 0xFFFFFFFF;
            uint32_t b_high = b >> 32;
            uint32_t b_low = b & 0xFFFFFFFF;
            
            // 计算四个32位乘积
            uint64_t p_low_low = (uint64_t)a_low * b_low;           // 低×低
            uint64_t p_low_high = (uint64_t)a_low * b_high;         // 低×高
            uint64_t p_high_low = (uint64_t)a_high * b_low;         // 高×低
            uint64_t p_high_high = (uint64_t)a_high * b_high;       // 高×高
            
            // 组合结果（128位 = 4个32位部分）
            uint64_t low_part = p_low_low;                          // 最低32位部分
            uint64_t mid_part1 = p_low_high;                        // 中间部分1
            uint64_t mid_part2 = p_high_low;                        // 中间部分2
            uint64_t high_part = p_high_high;                       // 最高32位部分
            
            // 将中间部分左移32位（对应它们在128位结果中的位置）
            uint64_t mid_combined = mid_part1 + mid_part2;
            uint64_t mid_shifted_low = mid_combined << 32;          // 中间部分的低32位影响
            uint64_t mid_shifted_high = mid_combined >> 32;         // 中间部分的高32位影响
            
            // 计算最终的低64位和高64位
            result.value.x = low_part + mid_shifted_low;            // 低64位
            result.value.y = high_part + mid_shifted_high;          // 高64位
            
            // 处理低64位加法可能产生的进位
            if (result.value.x < low_part) {  // 检查是否溢出
                result.value.y += 1;          // 向高64位进位
            }
            
            return result;
        }
        // Modular Multiplication
        // result = (input1 * input2) % modulus
        static T1 mult(const T1 input1,
                        const T1 input2,
                        const Modulus<T1> modulus)
        {
            if constexpr (std::is_same<T1, Data32>::value)
            {
                Data64 z = mult64(input1, input2);
                Data64 w = z >> (modulus.bit - 2);
                w = mult64(static_cast<Data32>(w), modulus.mu);
                w = w >> (modulus.bit + 3);
                w = mult64(static_cast<Data32>(w), modulus.value);
                z = z - w;
                return static_cast<T1>(
                    (z >= modulus.value) ? (z - modulus.value) : z);
            }
            else
            {
                uint128_t z = mult128(input1, input2);
                uint128_t w = z >> (modulus.bit - 2);
                w = mult128(w.value.x, modulus.mu);
                w = w >> (modulus.bit + 3);
                w = mult128(w.value.x, modulus.value);
                z = z - w;
                return (z.value.x >= modulus.value)
                           ? (z.value.x - modulus.value)
                           : z.value.x;
            }
        }

        // Barrett Reduction
        // result = input1 % modulus
        static T1 reduce(const T1 input,
                                                    const Modulus<T1> modulus)
        {
            if constexpr (std::is_same<T1, Data32>::value)
            {
                Data64 z = static_cast<Data64>(input);
                Data64 w = z >> (modulus.bit - 2);
                w = mult64(static_cast<Data32>(w), modulus.mu);
                w = w >> (modulus.bit + 3);
                w = mult64(static_cast<Data32>(w), modulus.value);
                z = z - w;
                return static_cast<T1>(
                    (z >= modulus.value) ? (z - modulus.value) : z);
            }
            else
            {
                uint128_t z(input);
                uint128_t w = z >> (modulus.bit - 2);
                w = mult128(w.value.x, modulus.mu);
                w = w >> (modulus.bit + 3);
                w = mult128(w.value.x, modulus.value);
                z = z - w;
                return (z.value.x >= modulus.value)
                           ? (z.value.x - modulus.value)
                           : z.value.x;
            }
        }

        // Forced Reduction
        // result = input1 % modulus
        static T1 forced_reduce(T1 input1,  const Modulus<T1>& modulus)
        {
            if constexpr (std::is_same<T1, Data32>::value)
            {
                Data64 z = (static_cast<Data64>(input1[1]) << 32) |
                           static_cast<Data32>(input1[0]);
                Data64 w = z >> (modulus.bit - 2);
                w = mult64(static_cast<Data32>(w), modulus.mu);
                w = w >> (modulus.bit + 3);
                w = mult64(static_cast<Data32>(w), modulus.value);
                z = z - w;
                return static_cast<T1>(
                    (z >= modulus.value) ? (z - modulus.value) : z);
            }
            else
            {
                uint128_t z;
                z.value.x = input1;
                z.value.y = 0;
                uint128_t w = z >> (modulus.bit - 2);
                w = mult128(w.value.x, modulus.mu);
                w = w >> (modulus.bit + 3);
                w = mult128(w.value.x, modulus.value);
                z = z - w;
                return (z.value.x >= modulus.value)
                           ? (z.value.x - modulus.value)
                           : z.value.x;
            }
        }

        // Forced Reduction (Repeated Reduction Until Input < Modulus)
        // result = input1 % modulus
        static T1
        reduce_forced(const T1 input, const Modulus<T1> modulus)
        {
            T1 result = input;
            while (result >= modulus.value)
            {
                result = forced_reduce(result, modulus);
            }
            return result;
        }

        // Forced Reduction
        // result = input1 % modulus
        // static T1 reduce(T1* input1,
        //                                             const Modulus<T1>& modulus)
        // {
        //     if constexpr (std::is_same<T1, Data32>::value)
        //     {
        //         Data64 z = (static_cast<Data64>(input1[1]) << 32) |
        //                    static_cast<Data32>(input1[0]);
        //         Data64 w = z >> (modulus.bit - 2);
        //         w = mult64(static_cast<Data32>(w), modulus.mu);
        //         w = w >> (modulus.bit + 3);
        //         w = mult64(static_cast<Data32>(w), modulus.value);
        //         z = z - w;
        //         return static_cast<T1>(
        //             (z >= modulus.value) ? (z - modulus.value) : z);
        //     }
        //     else
        //     {
        //         uint128_t z;
        //         z.value.x = input1[0];
        //         z.value.y = input1[1];
        //         uint128_t w = z >> (modulus.bit - 2);
        //         w = mult128(w.value.x, modulus.mu);
        //         w = w >> (modulus.bit + 3);
        //         w = mult128(w.value.x, modulus.value);
        //         z = z - w;
        //         return (z.value.x >= modulus.value)
        //                    ? (z.value.x - modulus.value)
        //                    : z.value.x;
        //     }
        // }
    };

} // namespace modular_operation_gpu

template <typename T>
using OPERATOR_GPU = modular_operation_gpu::BarrettOperations<T>;

typedef OPERATOR_GPU<Data32> OPERATOR_GPU_32;
typedef OPERATOR_GPU<Data64> OPERATOR_GPU_64;
}