/*
 * This file contains portions of code originally from the Microsoft SealPIR
 * project, which are licensed under the MIT License. The original code is
 * reproduced below with its original copyright:
 *
 * MIT License
 *
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * -----------------------------------------------------------------------------
 *
 * Modifications and additions by Alişah Özcan are licensed under the Apache
 * License, Version 2.0.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 *
 * -----------------------------------------------------------------------------
 *
 * Original code is under the MIT License; modifications and additions are under
 * Apache-2.0.
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * Developer: Alişah Özcan
 * Paper: https://eprint.iacr.org/2024/1543
 */

#ifndef PIR_H
#define PIR_H

#include "heongpu.cuh"

#include <string>
#include <iomanip>
#include <omp.h>

namespace pirongpu
{   

    class CudaException : public std::exception
    {
      public:
        CudaException(const std::string& file, int line, cudaError_t error)
            : file_(file), line_(line), error_(error)
        {
        }

        const char* what() const noexcept override
        {
            return m_error_string.c_str();
        }

      private:
        std::string file_;
        int line_;
        cudaError_t error_;
        std::string m_error_string = "CUDA Error in " + file_ + " at line " +
                                     std::to_string(line_) + ": " +
                                     cudaGetErrorString(error_);
    };

#define PIRONGPU_CUDA_CHECK(err)                                                \
    do                                                                         \
    {                                                                          \
        cudaError_t error = err;                                               \
        if (error != cudaSuccess)                                              \
        {                                                                      \
            throw CudaException(__FILE__, __LINE__, error);                    \
        }                                                                      \
    } while (0)

    typedef std::vector<heongpu::Plaintext> Database;
    typedef std::vector<std::vector<heongpu::Ciphertext>> PirQuery;
    typedef std::vector<heongpu::Ciphertext> PirReply;

    struct PirParams
    {
        bool enable_symmetric;
        bool enable_batching;
        bool enable_mswitching;
        std::uint64_t ele_num;
        std::uint64_t ele_size;
        std::uint64_t elements_per_plaintext;
        std::uint64_t num_of_plaintexts; // number of plaintexts in database
        std::uint32_t d; // number of dimensions for the database
        std::uint32_t expansion_ratio; // ratio of ciphertext to plaintext
        std::vector<std::uint64_t> nvec; // size of each of the d dimensions
        std::uint32_t slot_count;
    };

    void gen_pir_params(uint64_t ele_num, uint64_t ele_size, uint32_t d,
                        const heongpu::Parameters& params,
                        PirParams& pir_params, bool enable_symmetric = false,
                        bool enable_batching = true,
                        bool enable_mswitching = true);

    std::uint64_t plaintexts_per_db(std::uint32_t logt, std::uint64_t N,
                                    std::uint64_t ele_num,
                                    std::uint64_t ele_size);

    std::uint64_t elements_per_ptxt(std::uint32_t logt, std::uint64_t N,
                                    std::uint64_t ele_size);

    std::uint64_t coefficients_per_element(std::uint32_t logt,
                                           std::uint64_t ele_size);

    std::vector<std::uint64_t> bytes_to_coeffs(std::uint32_t limit,
                                               const std::uint8_t* bytes,
                                               std::uint64_t size);

    void coeffs_to_bytes(std::uint32_t limit,
                         const std::vector<std::uint64_t>& coeffs,
                         std::uint8_t* output, std::uint32_t size_out,
                         std::uint32_t ele_size);

    std::vector<std::uint64_t> compute_indices(std::uint64_t desiredIndex,
                                               std::vector<std::uint64_t> nvec);

    uint32_t compute_expansion_ratio(Modulus64& plain_mod,
                                     std::vector<Modulus64>& coeff_mod);

    void compose_to_ciphertext(int poly_modulus_degree, Modulus64& plain_mod,
                               std::vector<Modulus64>& coeff_mod,
                               std::vector<heongpu::Plaintext>& pts,
                               heongpu::Ciphertext& ct);

    __global__ void compose_to_ciphertext_piece(Data64* ct, Data64* plain,
                                                int shift);

    __global__ void decompose_to_plaintexts_piece(Data64* ct, Data64* plain,
                                                  int shift,
                                                  uint64_t pt_bitmask);

} // namespace pirongpu

#endif // PIR_H