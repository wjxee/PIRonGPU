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

#include "pir.cuh"

namespace pirongpu
{

    std::vector<std::uint64_t> get_dimensions(std::uint64_t num_of_plaintexts,
                                              std::uint32_t d)
    {
        assert(d > 0);
        assert(num_of_plaintexts > 0);

        std::uint64_t root =
            std::max(static_cast<uint32_t>(2),
                     static_cast<uint32_t>(
                         std::floor(std::pow(num_of_plaintexts, 1.0 / d))));

        std::vector<std::uint64_t> dimensions(d, root);

        for (int i = 0; i < d; i++)
        {
            if (std::accumulate(dimensions.begin(), dimensions.end(), 1,
                                std::multiplies<uint64_t>()) >
                num_of_plaintexts)
            {
                break;
            }
            dimensions[i] += 1;
        }

        std::uint32_t prod =
            std::accumulate(dimensions.begin(), dimensions.end(), 1,
                            std::multiplies<uint64_t>());
        assert(prod >= num_of_plaintexts);
        return dimensions;
    }

    void gen_pir_params(uint64_t ele_num, uint64_t ele_size, uint32_t d,
                        const heongpu::Parameters& params,
                        PirParams& pir_params, bool enable_symmetric,
                        bool enable_batching, bool enable_mswitching)
    {
        std::uint32_t N = params.poly_modulus_degree();
        Modulus64 t = params.plain_modulus();
        std::uint32_t logt = floor(log2(t.value)); // # of usable bits
        std::uint64_t elements_per_plaintext;
        std::uint64_t num_of_plaintexts;

        if (enable_batching)
        {
            elements_per_plaintext = elements_per_ptxt(logt, N, ele_size);
            num_of_plaintexts = plaintexts_per_db(logt, N, ele_num, ele_size);
        }
        else
        {
            elements_per_plaintext = 1;
            num_of_plaintexts = ele_num;
        }

        std::vector<uint64_t> nvec = get_dimensions(num_of_plaintexts, d);

        uint32_t expansion_ratio = 0;
        for (uint32_t i = 0; i < params.key_modulus_count(); ++i)
        {
            double logqi = log2(params.key_modulus()[i].value);
            expansion_ratio += ceil(logqi / logt);
        }

        pir_params.enable_symmetric = enable_symmetric;
        pir_params.enable_batching = enable_batching;
        pir_params.enable_mswitching = enable_mswitching;
        pir_params.ele_num = ele_num;
        pir_params.ele_size = ele_size;
        pir_params.elements_per_plaintext = elements_per_plaintext;
        pir_params.num_of_plaintexts = num_of_plaintexts;
        pir_params.d = d;
        pir_params.expansion_ratio = expansion_ratio << 1;
        pir_params.nvec = nvec;
        pir_params.slot_count = N;
    }

    uint64_t plaintexts_per_db(uint32_t logt, uint64_t N, uint64_t ele_num,
                               uint64_t ele_size)
    {
        uint64_t ele_per_ptxt = elements_per_ptxt(logt, N, ele_size);
        return ceil((double) ele_num / ele_per_ptxt);
    }

    uint64_t elements_per_ptxt(uint32_t logt, uint64_t N, uint64_t ele_size)
    {
        uint64_t coeff_per_ele = coefficients_per_element(logt, ele_size);
        uint64_t ele_per_ptxt = N / coeff_per_ele;
        assert(ele_per_ptxt > 0);
        return ele_per_ptxt;
    }

    uint64_t coefficients_per_element(uint32_t logt, uint64_t ele_size)
    {
        return ceil(8 * ele_size / (double) logt);
    }

    std::vector<uint64_t> bytes_to_coeffs(uint32_t limit, const uint8_t* bytes,
                                          uint64_t size)
    {
        uint64_t size_out = coefficients_per_element(limit, size);
        std::vector<uint64_t> output(size_out);

        uint32_t room = limit;
        uint64_t* target = &output[0];

        for (uint32_t i = 0; i < size; i++)
        {
            uint8_t src = bytes[i];
            uint32_t rest = 8;
            while (rest)
            {
                if (room == 0)
                {
                    target++;
                    room = limit;
                }
                uint32_t shift = rest;
                if (room < rest)
                {
                    shift = room;
                }
                *target = *target << shift;
                *target = *target | (src >> (8 - shift));
                src = src << shift;
                room -= shift;
                rest -= shift;
            }
        }

        *target = *target << room;
        return output;
    }

    void coeffs_to_bytes(uint32_t limit, const std::vector<uint64_t>& coeffs,
                         uint8_t* output, uint32_t size_out, uint32_t ele_size)
    {
        uint32_t room = 8;
        uint32_t j = 0;
        uint8_t* target = output;
        uint32_t bits_left = ele_size * 8;
        for (uint32_t i = 0; i < coeffs.size(); i++)
        {
            if (bits_left == 0)
            {
                bits_left = ele_size * 8;
            }
            uint64_t src = coeffs[i];
            uint32_t rest = min(limit, bits_left);
            while (rest && j < size_out)
            {
                uint32_t shift = rest;
                if (room < rest)
                {
                    shift = room;
                }

                target[j] = target[j] << shift;
                target[j] = target[j] | (src >> (limit - shift));
                src = src << shift;
                room -= shift;
                rest -= shift;
                bits_left -= shift;
                if (room == 0)
                {
                    j++;
                    room = 8;
                }
            }
        }
    }

    std::vector<uint64_t> compute_indices(uint64_t desiredIndex,
                                          std::vector<uint64_t> Nvec)
    {
        uint32_t num = Nvec.size();
        uint64_t product = 1;

        for (uint32_t i = 0; i < num; i++)
        {
            product *= Nvec[i];
        }

        uint64_t j = desiredIndex;
        std::vector<uint64_t> result;

        for (uint32_t i = 0; i < num; i++)
        {
            product /= Nvec[i];
            uint64_t ji = j / product;

            result.push_back(ji);
            j -= ji * product;
        }

        return result;
    }

    uint32_t compute_expansion_ratio(Modulus64& plain_mod,
                                     std::vector<Modulus64>& coeff_mod)
    {
        uint32_t expansion_ratio = 0;
        uint32_t pt_bits_per_coeff = log2(plain_mod.value);
        for (size_t i = 0; i < coeff_mod.size(); ++i)
        {
            double coeff_bit_size = log2(coeff_mod[i].value);
            expansion_ratio += ceil(coeff_bit_size / pt_bits_per_coeff);
        }
        return expansion_ratio;
    }

    // TODO: make it efficient!
    __global__ void compose_to_ciphertext_piece(Data64* ct, Data64* plain,
                                                int shift)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes

        Data64 ct_reg = 0;
        if (!(shift == 0))
        {
            ct_reg = ct[idx];
        }

        Data64 plain_reg = plain[idx];

        ct_reg = ct_reg + (plain_reg << shift);

        ct[idx] = ct_reg;
    }

    __global__ void decompose_to_plaintexts_piece(Data64* ct, Data64* plain,
                                                  int shift,
                                                  uint64_t pt_bitmask)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        Data64 ct_reg = ct[idx];
        ct_reg = (ct_reg >> shift) & pt_bitmask;

        plain[idx] = ct_reg;
    }

    void compose_to_ciphertext(int poly_modulus_degree, Modulus64& plain_mod,
                               std::vector<Modulus64>& coeff_mod,
                               std::vector<heongpu::Plaintext>& pts,
                               heongpu::Ciphertext& ct)
    {
        std::vector<Modulus64> coeff_mod_;
        coeff_mod_.assign(coeff_mod.begin(), coeff_mod.end() - 1);

        size_t ct_poly_count =
            pts.size() / compute_expansion_ratio(plain_mod, coeff_mod_);

        int pointer_inter = 0;

        const uint32_t pt_bits_per_coeff = log2(plain_mod.value);
        const auto coeff_count = poly_modulus_degree;
        const auto coeff_mod_count = coeff_mod_.size();

        for (size_t poly_index = 0; poly_index < ct_poly_count; ++poly_index)
        {
            for (size_t coeff_mod_index = 0; coeff_mod_index < coeff_mod_count;
                 ++coeff_mod_index)
            {
                const double coeff_bit_size =
                    log2(coeff_mod_[coeff_mod_index].value);
                const size_t local_expansion_ratio =
                    ceil(coeff_bit_size / pt_bits_per_coeff);

                size_t shift = 0;
                for (size_t i = 0; i < local_expansion_ratio; ++i)
                {
                    compose_to_ciphertext_piece<<<
                        dim3((coeff_count >> 8), 1, 1), 256>>>(
                        ct.data() + (coeff_mod_index * coeff_count) +
                            (coeff_mod_count * poly_index * coeff_count),
                        pts[pointer_inter].data(), shift);

                    pointer_inter++;
                    shift += pt_bits_per_coeff;
                }
            }
        }
    }

} // namespace pirongpu