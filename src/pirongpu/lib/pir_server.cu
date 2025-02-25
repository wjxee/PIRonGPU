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

#include "pir_server.cuh"

namespace pirongpu
{

    PIRServer::PIRServer(std::shared_ptr<heongpu::Parameters>& context,
                         const PirParams& pir_params)
        : context_(context), pir_params_(pir_params), is_db_preprocessed_(false)
    {
        encoder_ = std::make_shared<heongpu::HEEncoder>(*context_);
        evaluator_ = std::make_shared<heongpu::HEArithmeticOperator>(*context_,
                                                                     *encoder_);
    }

    void PIRServer::set_database(
        std::unique_ptr<std::vector<heongpu::Plaintext>>&& db)
    {
        if (!db)
        {
            throw std::invalid_argument("db cannot be null");
        }

        db_ = std::move(db);
        is_db_preprocessed_ = false;
    }

    void PIRServer::preprocess_database()
    {
        if (!is_db_preprocessed_)
        {
            for (uint32_t i = 0; i < db_->size(); i++)
            {
                evaluator_->transform_to_ntt_inplace(db_->operator[](i));
            }

            is_db_preprocessed_ = true;
        }
    }

    void PIRServer::set_database(const std::unique_ptr<const uint8_t[]>& bytes,
                                 uint64_t ele_num, uint64_t ele_size)
    {
        uint32_t logt = floor(log2(context_->plain_modulus().value));
        uint32_t N = context_->poly_modulus_degree();

        uint64_t num_of_plaintexts = pir_params_.num_of_plaintexts;

        uint64_t prod = 1;
        for (uint32_t i = 0; i < pir_params_.nvec.size(); i++)
        {
            prod *= pir_params_.nvec[i];
        }
        uint64_t matrix_plaintexts = prod;

        assert(num_of_plaintexts <= matrix_plaintexts);

        auto result = std::make_unique<std::vector<heongpu::Plaintext>>();
        result->reserve(matrix_plaintexts);

        uint64_t ele_per_ptxt = pir_params_.elements_per_plaintext;
        uint64_t bytes_per_ptxt = ele_per_ptxt * ele_size;

        uint64_t db_size = ele_num * ele_size;

        uint64_t coeff_per_ptxt =
            ele_per_ptxt * coefficients_per_element(logt, ele_size);
        assert(coeff_per_ptxt <= N);

        uint32_t offset = 0;

        for (uint64_t i = 0; i < num_of_plaintexts; i++)
        {
            uint64_t process_bytes = 0;

            if (db_size <= offset)
            {
                break;
            }
            else if (db_size < offset + bytes_per_ptxt)
            {
                process_bytes = db_size - offset;
            }
            else
            {
                process_bytes = bytes_per_ptxt;
            }
            assert(process_bytes % ele_size == 0);
            uint64_t ele_in_chunk = process_bytes / ele_size;

            // Get the coefficients of the elements that will be packed in
            // plaintext i
            std::vector<uint64_t> coefficients(coeff_per_ptxt);
            for (uint64_t ele = 0; ele < ele_in_chunk; ele++)
            {
                std::vector<uint64_t> element_coeffs = bytes_to_coeffs(
                    logt, bytes.get() + offset + (ele_size * ele), ele_size);
                std::copy(element_coeffs.begin(), element_coeffs.end(),
                          coefficients.begin() +
                              (coefficients_per_element(logt, ele_size) * ele));
            }

            offset += process_bytes;

            uint64_t used = coefficients.size();

            assert(used <= coeff_per_ptxt);

            // Pad the rest with 1s
            for (uint64_t j = 0; j < (pir_params_.slot_count - used); j++)
            {
                coefficients.push_back(1);
            }

            heongpu::Plaintext plain(*context_);

            encoder_->encode(plain, coefficients);
            result->push_back(std::move(plain));
        }

        // Add padding to make database a matrix
        uint64_t current_plaintexts = result->size();
        assert(current_plaintexts <= num_of_plaintexts);

        std::vector<Data64> padding(N, 1);

        for (uint64_t i = 0; i < (matrix_plaintexts - current_plaintexts); i++)
        {
            heongpu::Plaintext plain(padding, *context_);
            result->push_back(plain);
        }

        set_database(std::move(result));
    }

    void PIRServer::set_galois_key(std::uint32_t client_id,
                                   heongpu::Galoiskey galkey)
    {
        galoisKeys_[client_id] = heongpu::Galoiskey(galkey);
    }

    PirReply PIRServer::generate_reply(PirQuery& query, uint32_t client_id,
                                       cudaStream_t& stream)
    {
        std::vector<uint64_t> nvec = pir_params_.nvec;
        uint64_t product = 1;

        for (uint32_t i = 0; i < nvec.size(); i++)
        {
            product *= nvec[i];
        }

        auto coeff_count = context_->poly_modulus_degree();

        std::vector<heongpu::Plaintext>* cur = db_.get();
        std::vector<heongpu::Plaintext> intermediate_plain; // decompose....

        int N = context_->poly_modulus_degree();
        std::vector<Modulus64> key_modulus = context_->key_modulus();

        Modulus plainmod = context_->plain_modulus();
        int logt = std::floor(std::log2(plainmod.value));
        std::vector<Modulus64> coeff_mod_;
        coeff_mod_.assign(key_modulus.begin(), key_modulus.end() - 1);

        const uint32_t pt_bits_per_coeff = log2(plainmod.value);
        const auto coeff_mod_count = coeff_mod_.size();
        const uint64_t pt_bitmask = (1 << pt_bits_per_coeff) - 1;

        uint32_t compute_expansion_ratio_ =
            compute_expansion_ratio(plainmod, coeff_mod_);

        for (uint32_t i = 0; i < nvec.size(); i++)
        {
            std::cout << "Server: " << i + 1 << "-th recursion level started "
                      << std::endl;

            std::vector<heongpu::Ciphertext> expanded_query; //

            uint64_t n_i = nvec[i];
            std::cout << "Server: n_i = " << n_i << std::endl;
            std::cout << "Server: expanding " << query[i].size()
                      << " query ctxts" << std::endl;

            for (uint32_t j = 0; j < query[i].size(); j++)
            {
                uint64_t total = N;
                if (j == query[i].size() - 1)
                {
                    total = n_i % N;
                }
                std::cout << "-- expanding one query ctxt into " << total
                          << " ctxtss " << std::endl;

                std::vector<heongpu::Ciphertext> expanded_query_part =
                    expand_query(query[i][j], total, client_id, stream);
                expanded_query.insert(
                    expanded_query.end(),
                    std::make_move_iterator(expanded_query_part.begin()),
                    std::make_move_iterator(expanded_query_part.end()));
                expanded_query_part.clear();
            }

            std::cout << "Server: expansion done " << std::endl;
            if (expanded_query.size() != n_i)
            {
                std::cout << " size mismatch!!! " << expanded_query.size()
                          << ", " << n_i << std::endl;
            }

            // Transform expanded query to NTT, and ...
            for (uint32_t jj = 0; jj < expanded_query.size(); jj++)
            {
                evaluator_->transform_to_ntt_inplace(
                    expanded_query[jj],
                    heongpu::ExecutionOptions().set_stream(stream));
            }
            // Transform plaintext to NTT. If database is pre-processed, can
            // skip
            if ((!is_db_preprocessed_) || i > 0)
            {
                for (uint32_t jj = 0; jj < cur->size(); jj++)
                {
                    evaluator_->transform_to_ntt_inplace(
                        (*cur)[jj],
                        heongpu::ExecutionOptions().set_stream(stream));
                }
            }

            product /= n_i;

            std::vector<heongpu::Ciphertext> intermediateCtxts; //(product);
            intermediateCtxts.reserve(product);
            for (uint32_t a = 0; a < product; a++)
            {
                intermediateCtxts.emplace_back(*context_);
            }

            heongpu::Ciphertext temp(*context_);
            for (uint64_t k = 0; k < product; k++)
            {
                evaluator_->multiply_plain(
                    expanded_query[0], (*cur)[k], intermediateCtxts[k],
                    heongpu::ExecutionOptions().set_stream(stream));

                for (uint64_t j = 1; j < n_i; j++)
                {
                    evaluator_->multiply_plain(
                        expanded_query[j], (*cur)[k + j * product], temp,
                        heongpu::ExecutionOptions().set_stream(stream));
                    evaluator_->add_inplace(
                        intermediateCtxts[k], temp,
                        heongpu::ExecutionOptions().set_stream(
                            stream)); // Adds to first component.
                }
            }

            for (uint32_t jj = 0; jj < intermediateCtxts.size(); jj++)
            {
                evaluator_->transform_from_ntt_inplace(
                    intermediateCtxts[jj],
                    heongpu::ExecutionOptions().set_stream(stream));
            }

            if (i == nvec.size() - 1)
            {
                return intermediateCtxts;
            }
            else
            {
                intermediate_plain.clear();
                cur = &intermediate_plain;

                std::vector<heongpu::Plaintext> plains;
                plains.reserve(compute_expansion_ratio_ * 2 *
                               product); // ct_size = 2
                for (int inner_lp = 0;
                     inner_lp < (compute_expansion_ratio_ * 2 * product);
                     inner_lp++)
                {
                    plains.emplace_back(*context_);
                }

                int plain_counter = 0;
                int plain_size = compute_expansion_ratio_ * 2;
                for (uint64_t rr = 0; rr < product; rr++)
                {
                    /////////////////////////////////////////////////////////////////////////////////
                    // decompose_to_plaintexts
                    int counter = 0;
                    for (size_t poly_index = 0; poly_index < 2; ++poly_index)
                    {
                        for (size_t coeff_mod_index = 0;
                             coeff_mod_index < coeff_mod_count;
                             ++coeff_mod_index)
                        {
                            const double coeff_bit_size =
                                log2(coeff_mod_[coeff_mod_index].value);
                            const size_t local_expansion_ratio =
                                ceil(coeff_bit_size / pt_bits_per_coeff);
                            size_t shift = 0;
                            for (size_t i = 0; i < local_expansion_ratio; ++i)
                            {
                                decompose_to_plaintexts_piece<<<
                                    dim3((N >> 8), 1, 1), 256, 0, stream>>>(
                                    intermediateCtxts[rr].data() +
                                        (coeff_mod_index * N) +
                                        (N * poly_index * coeff_mod_count),
                                    plains[plain_counter + counter].data(),
                                    int(shift), pt_bitmask);
                                ++counter;
                                shift += pt_bits_per_coeff;
                            }
                        }
                    }
                    PIRONGPU_CUDA_CHECK(cudaGetLastError());
                    /////////////////////////////////////////////////////////////////////////////////
                    for (uint32_t jj = 0; jj < plain_size; jj++)
                    {
                        // intermediate_plain.emplace_back(plains[jj]);
                        intermediate_plain.push_back(
                            std::move(plains[plain_counter + jj]));
                    }
                    plain_counter = plain_counter + plain_size;
                }
                product =
                    intermediate_plain.size(); // multiply by expansion rate.
            }
            std::cout << "Server: " << i + 1 << "-th recursion level finished "
                      << std::endl;
            std::cout << std::endl;
        }
        std::cout << "reply generated!  " << std::endl;

        assert(0);
        std::vector<heongpu::Ciphertext> fail(1);
        return fail;
    }

    std::vector<heongpu::Ciphertext>
    PIRServer::expand_query(const heongpu::Ciphertext& encrypted, uint32_t m,
                            uint32_t client_id, cudaStream_t& stream)
    {
        heongpu::Galoiskey& galkey = galoisKeys_[client_id];

        // Assume that m is a power of 2. If not, round it to the next power
        // of 2.
        uint32_t logm = ceil(log2(m));

        std::vector<uint32_t> galois_elts;
        auto n = context_->poly_modulus_degree();
        if (logm > ceil(log2(n)))
        {
            throw std::logic_error("m > n is not allowed.");
        }
        for (int i = 0; i < ceil(log2(n)); i++)
        {
            galois_elts.push_back((n + int(pow(2, i))) / int(pow(2, i)));
        }

        std::vector<heongpu::Ciphertext> temp;
        temp.push_back(encrypted);
        heongpu::Ciphertext tempctxt(*context_);
        heongpu::Ciphertext tempctxt_rotated(*context_);
        heongpu::Ciphertext tempctxt_shifted(*context_);
        heongpu::Ciphertext tempctxt_rotatedshifted(*context_);

        for (uint32_t i = 0; i < logm - 1; i++)
        {
            std::vector<heongpu::Ciphertext> newtemp;
            newtemp.reserve(temp.size() << 1);
            for (uint32_t a = 0; a < (temp.size() << 1); a++)
            {
                newtemp.emplace_back(*context_);
            }
            // temp[a] = (j0 = a (mod 2**i) ? ) : Enc(x^{j0 - a}) else Enc(0).
            // With some scaling....
            int index_raw = (n << 1) - (1 << i);
            int index = (index_raw * galois_elts[i]) % (n << 1);

            for (uint32_t a = 0; a < temp.size(); a++)
            {
                evaluator_->apply_galois(
                    temp[a], tempctxt_rotated, galkey, galois_elts[i],
                    heongpu::ExecutionOptions().set_stream(stream));

                evaluator_->add(temp[a], tempctxt_rotated, newtemp[a],
                                heongpu::ExecutionOptions().set_stream(stream));

                evaluator_->multiply_power_of_X(
                    temp[a], tempctxt_shifted, index_raw,
                    heongpu::ExecutionOptions().set_stream(stream));

                evaluator_->multiply_power_of_X(
                    tempctxt_rotated, tempctxt_rotatedshifted, index,
                    heongpu::ExecutionOptions().set_stream(stream));

                // Enc(2^i x^j) if j = 0 (mod 2**i).
                evaluator_->add(tempctxt_shifted, tempctxt_rotatedshifted,
                                newtemp[a + temp.size()],
                                heongpu::ExecutionOptions().set_stream(stream));

                newtemp[a].switch_stream(stream);
                newtemp[a + temp.size()].switch_stream(stream);
            }

            temp = std::move(newtemp);
        }

        // Last step of the loop
        std::vector<heongpu::Ciphertext> newtemp;
        newtemp.reserve(temp.size() << 1);
        for (uint32_t a = 0; a < (temp.size() << 1); a++)
        {
            newtemp.emplace_back(*context_);
        }

        int index_raw = (n << 1) - (1 << (logm - 1));
        int index = (index_raw * galois_elts[logm - 1]) % (n << 1);

        for (uint32_t a = 0; a < temp.size(); a++)
        {
            if (a >= (m - (1 << (logm - 1))))
            { // corner case.
                evaluator_->add(temp[a], temp[a], newtemp[a],
                                heongpu::ExecutionOptions().set_stream(stream));
                newtemp[a].switch_stream(stream);
            }
            else
            {
                evaluator_->apply_galois(
                    temp[a], tempctxt_rotated, galkey, galois_elts[logm - 1],
                    heongpu::ExecutionOptions().set_stream(stream));

                evaluator_->add(temp[a], tempctxt_rotated, newtemp[a],
                                heongpu::ExecutionOptions().set_stream(stream));

                evaluator_->multiply_power_of_X(
                    temp[a], tempctxt_shifted, index_raw,
                    heongpu::ExecutionOptions().set_stream(stream));
                evaluator_->multiply_power_of_X(
                    tempctxt_rotated, tempctxt_rotatedshifted, index,
                    heongpu::ExecutionOptions().set_stream(stream));
                evaluator_->add(tempctxt_shifted, tempctxt_rotatedshifted,
                                newtemp[a + temp.size()],
                                heongpu::ExecutionOptions().set_stream(stream));

                newtemp[a].switch_stream(stream);
                newtemp[a + temp.size()].switch_stream(stream);
            }
        }

        std::vector<heongpu::Ciphertext>::iterator first = newtemp.begin();
        std::vector<heongpu::Ciphertext>::iterator last = newtemp.begin() + m;
        std::vector<heongpu::Ciphertext> newVec(std::make_move_iterator(first),
                                                std::make_move_iterator(last));

        return newVec;
    }

    void PIRServer::set_one_ct(heongpu::Ciphertext one)
    {
        one_ = one;
        evaluator_->transform_to_ntt_inplace(one_);
    }

    heongpu::Ciphertext PIRServer::simple_query(uint64_t index)
    {
        // There is no transform_from_ntt that takes a plaintext
        heongpu::Ciphertext ct(*context_);
        heongpu::Plaintext pt = db_->operator[](index);
        evaluator_->multiply_plain(one_, pt, ct);
        evaluator_->transform_from_ntt_inplace(ct);
        return ct;
    }

    heongpu::Plaintext PIRServer::simple_query2(uint64_t index)
    {
        // There is no transform_from_ntt that takes a plaintext

        heongpu::Plaintext pt = db_->operator[](index);

        return pt;
    }

} // namespace pirongpu