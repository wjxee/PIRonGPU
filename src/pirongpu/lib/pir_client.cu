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

#include "pir_client.cuh"

namespace pirongpu
{

    PIRClient::PIRClient(std::shared_ptr<heongpu::Parameters>& context,
                         const PirParams& pir_params)
        : context_(context), pir_params_(pir_params)
    {
        keygen_ = std::make_shared<heongpu::HEKeyGenerator>(*context_);

        secret_key_ = std::make_shared<heongpu::Secretkey>(*context_);
        keygen_->generate_secret_key(*secret_key_);

        public_key_ = std::make_shared<heongpu::Publickey>(*context_);
        keygen_->generate_public_key(*public_key_, *secret_key_);

        if (pir_params_.enable_symmetric)
        {
            throw std::invalid_argument(
                "Symmetric Encryption is not supported!");
        }
        else
        {
            encryptor_ =
                std::make_shared<heongpu::HEEncryptor>(*context_, *public_key_);
        }

        encoder_ = std::make_shared<heongpu::HEEncoder>(*context_);
        decryptor_ =
            std::make_shared<heongpu::HEDecryptor>(*context_, *secret_key_);
        evaluator_ = std::make_shared<heongpu::HEArithmeticOperator>(*context_,
                                                                     *encoder_);
    }

    heongpu::Galoiskey PIRClient::generate_galois_keys()
    {
        std::vector<uint32_t> galois_elts;
        int N = context_->poly_modulus_degree();
        int logN = context_->log_poly_modulus_degree();

        for (int i = 0; i < logN; i++)
        {
            galois_elts.push_back((N + int(pow(2, i))) / int(pow(2, i)));
        }
        // std::cout << "gk:";
        // for (int i = 0; i < logN; i++)
        // {
        //     std::cout << galois_elts[i] <<";";
        // }
        // std::cout << ";" << std::endl;
        heongpu::Galoiskey galois_key(*context_, galois_elts);
        keygen_->generate_galois_key(galois_key, *secret_key_);
        return galois_key;
    }

    uint64_t PIRClient::get_fv_index(uint64_t element_index)
    {
        return static_cast<uint64_t>(element_index /
                                     pir_params_.elements_per_plaintext);
    }

    uint64_t PIRClient::get_fv_offset(uint64_t element_index)
    {
        return element_index % pir_params_.elements_per_plaintext;
    }

    PirQuery PIRClient::generate_query(uint64_t desiredIndex)
    {
        indices_ = compute_indices(desiredIndex, pir_params_.nvec);
        PirQuery result(pir_params_.d);

        int N = context_->poly_modulus_degree();
        Modulus64 plain_modulus = context_->plain_modulus();
        for (uint32_t i = 0; i < indices_.size(); i++)
        {
            uint32_t num_ptxts = ceil((pir_params_.nvec[i] + 0.0) / N);
            // initialize result.
            std::cout << "Client: index " << i + 1 << "/ " << indices_.size()
                      << " = " << indices_[i] << std::endl;
            std::cout << "Client: number of ctxts needed for query = "
                      << num_ptxts << std::endl;

            for (uint32_t j = 0; j < num_ptxts; j++)
            {
                std::vector<Data64> pt_vector(N, 0ULL);
                if (indices_[i] >= N * j && indices_[i] <= N * (j + 1))
                {
                    uint64_t real_index = indices_[i] - N * j;
                    uint64_t n_i = pir_params_.nvec[i];
                    uint64_t total = N;
                    if (j == num_ptxts - 1)
                    {
                        total = n_i % N;
                    }
                    uint64_t log_total = ceil(log2(total));
                    std::cout << "Client: Inverting " << pow(2, log_total)
                              << std::endl;

                    Data64 pow_ = pow(2, log_total);
                    pt_vector[real_index] =
                        OPERATOR64::modinv(pow_, plain_modulus);
                }
                heongpu::Plaintext pt(pt_vector, *context_);
                heongpu::Ciphertext dest(*context_);

                if (pir_params_.enable_symmetric)
                {
                    throw std::invalid_argument(
                        "Symmetric Encryption is not supported!");
                }
                else
                {
                    encryptor_->encrypt(dest, pt);
                }
                result[i].push_back(dest);
            }
        }

        return result;
    }

    std::vector<uint8_t> PIRClient::extract_bytes(heongpu::Plaintext& pt,
                                                  uint64_t offset)
    {
        uint32_t N = context_->poly_modulus_degree();
        uint32_t logt = floor(log2(context_->plain_modulus().value));
        uint32_t bytes_per_ptxt =
            pir_params_.elements_per_plaintext * pir_params_.ele_size;

        // Convert from FV plaintext (polynomial) to database element at the
        // client
        std::vector<uint8_t> elems(bytes_per_ptxt);
        std::vector<uint64_t> coeffs;
        // heongpu::Message coeffs_gpu(*context_);
        encoder_->decode(coeffs, pt);
        // coeffs_gpu.device_to_host(coeffs);

        coeffs_to_bytes(logt, coeffs, elems.data(), bytes_per_ptxt,
                        pir_params_.ele_size);
        return std::vector<uint8_t>(
            elems.begin() + offset * pir_params_.ele_size,
            elems.begin() + (offset + 1) * pir_params_.ele_size);
    }

    heongpu::Plaintext PIRClient::decode_reply(PirReply& reply)
    {
        // EncryptionParameters parms;
        // parms_id_type parms_id;
        if (pir_params_.enable_mswitching)
        {
            throw std::invalid_argument("Modulus switching is not supported!");
        }
        else
        {
        }

        std::vector<Modulus64> vec_key_modulus = context_->key_modulus();

        std::vector<Modulus64> coeff_mod_;
        coeff_mod_.assign(vec_key_modulus.begin(), vec_key_modulus.end() - 1);
        Modulus64 mod_plain = context_->plain_modulus();
        uint32_t exp_ratio = compute_expansion_ratio(mod_plain, coeff_mod_);
        uint32_t recursion_level = pir_params_.d;

        std::cout << "exp_ratio: " << exp_ratio << std::endl;

        std::vector<heongpu::Ciphertext> temp = reply;
        uint32_t ciphertext_size = 2;
        std::cout << "ciphertext_size: " << ciphertext_size << std::endl;

        for (uint32_t i = 0; i < recursion_level; i++)
        {
            std::cout << "Client: " << i + 1 << "/ " << recursion_level
                      << "-th decryption layer started." << std::endl;
            std::vector<heongpu::Ciphertext> newtemp;
            std::vector<heongpu::Plaintext> tempplain;

            for (uint32_t j = 0; j < temp.size(); j++)
            {
                heongpu::Plaintext ptxt(*context_);
                decryptor_->decrypt(ptxt, temp[j]);
                tempplain.push_back(ptxt);

                if ((j + 1) % (exp_ratio * ciphertext_size) == 0 && j > 0)
                {
                    // Combine into one ciphertext.
                    heongpu::Ciphertext combined(*context_);

                    compose_to_ciphertext(context_->poly_modulus_degree(),
                                          mod_plain, vec_key_modulus, tempplain,
                                          combined);
                    newtemp.push_back(combined);
                    tempplain.clear();
                }
            }
            std::cout << "Client: done." << std::endl;
            std::cout << std::endl;
            if (i == recursion_level - 1)
            {
                assert(temp.size() == 1);
                return tempplain[0];
            }
            else
            {
                tempplain.clear();
                temp = newtemp;
            }
        }

        // This should never be called
        assert(0);
        heongpu::Plaintext fail;
        return fail;
    }

    std::vector<uint8_t> PIRClient::decode_reply(PirReply& reply,
                                                 uint64_t offset)
    {
        heongpu::Plaintext result = decode_reply(reply);
        return extract_bytes(result, offset);
    }

    heongpu::Plaintext PIRClient::decrypt(heongpu::Ciphertext ct)
    {
        heongpu::Plaintext pt(*context_);
        decryptor_->decrypt(pt, ct);
        return pt;
    }

    heongpu::Ciphertext PIRClient::get_one()
    {
        std::vector<Data64> pt_vector(context_->poly_modulus_degree(), 0ULL);
        pt_vector[0] = 1;

        heongpu::Plaintext pt(pt_vector, *context_);
        heongpu::Ciphertext dest(*context_);

        encryptor_->encrypt(dest, pt);

        return dest;
    }

} // namespace pirongpu
