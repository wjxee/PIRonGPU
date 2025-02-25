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
#include "pir_client.cuh"
#include "pir_server.cuh"

using namespace pirongpu;

int main(int argc, char* argv[])
{
    uint32_t N = 4096;

    std::vector<int> log_Q_bases_bit_sizes = {36, 36};
    std::vector<int> log_P_bases_bit_sizes = {37};
    int plain_modulus = 1179649;

    std::shared_ptr<heongpu::Parameters> context =
        std::make_shared<heongpu::Parameters>(
            heongpu::scheme_type::bfv,
            heongpu::keyswitching_type::KEYSWITCHING_METHOD_I);

    context->set_poly_modulus_degree(N);
    context->set_coeff_modulus(log_Q_bases_bit_sizes, log_P_bases_bit_sizes);
    context->set_plain_modulus(plain_modulus);
    context->generate();

    heongpu::HEKeyGenerator keygen(*context);
    heongpu::Secretkey secret_key(*context);
    keygen.generate_secret_key(secret_key);

    heongpu::Publickey public_key(*context);
    keygen.generate_public_key(public_key, secret_key);

    heongpu::HEEncoder encoder(*context);
    heongpu::HEEncryptor encryptor(*context, public_key);
    heongpu::HEDecryptor decryptor(*context, secret_key);
    heongpu::HEArithmeticOperator operators(*context, encoder);

    uint32_t logt = floor(log2(plain_modulus));

    uint32_t plain_modulusx = plain_modulus;

    size_t slot_count = N;

    std::vector<uint64_t> coefficients(slot_count, 0ULL);
    for (uint32_t i = 0; i < coefficients.size(); i++)
    {
        coefficients[i] = rand() % plain_modulusx;
    }

    heongpu::Plaintext pt(*context);
    encoder.encode(pt, coefficients);

    heongpu::Ciphertext ct(*context);
    encryptor.encrypt(ct, pt);
    std::cout << "Encrypting" << std::endl;

    const uint32_t pt_bits_per_coeff = logt;
    const auto coeff_mod_count = 2;
    const uint64_t pt_bitmask = (1 << pt_bits_per_coeff) - 1;
    std::vector<Modulus64> prime_vector = context->key_modulus();
    std::vector<Modulus64> coeff_mod_;
    coeff_mod_.assign(prime_vector.begin(), prime_vector.end() - 1);
    Modulus64 plain_mod(plain_modulus);
    uint32_t compute_expansion_ratio_ =
        compute_expansion_ratio(plain_mod, coeff_mod_);

    std::vector<heongpu::Plaintext> encoded;
    /////////////////////////////////////////////////////////////////////////////////
    // decompose_to_plaintexts
    int counter = 0;
    for (size_t poly_index = 0; poly_index < 2; ++poly_index)
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
                heongpu::Plaintext inner_encoded(*context);
                decompose_to_plaintexts_piece<<<dim3((N >> 8), 1, 1), 256>>>(
                    ct.data() + (coeff_mod_index * N) +
                        (N * poly_index * coeff_mod_count),
                    inner_encoded.data(), int(shift), pt_bitmask);
                ++counter;
                shift += pt_bits_per_coeff;
                encoded.push_back(std::move(inner_encoded));
            }
        }
    }
    PIRONGPU_CUDA_CHECK(cudaGetLastError());
    /////////////////////////////////////////////////////////////////////////////////

    // std::vector<heongpu::Plaintext> encoded = decompose_to_plaintexts(params,
    // ct);
    std::cout << "Expansion Factor: " << encoded.size() << std::endl;
    std::cout << "Decoding" << std::endl;
    heongpu::Ciphertext decoded(*context);
    compose_to_ciphertext(N, plain_mod, prime_vector, encoded, decoded);
    std::cout << "Checking" << std::endl;
    heongpu::Plaintext pt2(*context);
    decryptor.decrypt(pt2, decoded);

    // assert(pt == pt2);

    std::vector<Data64> pt_cpu(N);
    std::vector<Data64> pt2_cpu(N);

    pt.get_data(pt_cpu);
    pt2.get_data(pt2_cpu);

    for (int i = 0; i < 20; i++)
    {
        // std::cout << pt_cpu[i] << " - " << pt2_cpu[i] << std::endl;
        assert(pt_cpu[i] == pt2_cpu[i]);
    }

    std::cout << "Worked" << std::endl;

    return EXIT_SUCCESS;
}
