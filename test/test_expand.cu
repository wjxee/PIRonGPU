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
    uint64_t number_of_items = 2048;
    uint64_t size_per_item = 288; // 1024; // in bytes
    uint32_t N = 4096;

    uint32_t d = 2;
    bool use_symmetric = false; // Do not change!
    bool use_batching = true; // Do not change!
    bool use_recursive_mod_switching = false; // Do not change!

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

    std::cout << "Main: Generating PIR parameters" << std::endl;
    PirParams pir_params;
    gen_pir_params(number_of_items, size_per_item, d, *context, pir_params,
                   use_symmetric, use_batching, use_recursive_mod_switching);

    std::cout << "Main: Generating PIR client" << std::endl;
    PIRClient client(context, pir_params);

    std::cout << "Main: Generating galois keys for client" << std::endl;
    heongpu::Galoiskey galois_keys = client.generate_galois_keys();

    std::random_device rd;
    uint64_t ele_index =
        1000; // rd() % number_of_items; // element in DB at random position
    uint64_t index = client.get_fv_index(ele_index); // index of FV plaintext
    uint64_t offset = client.get_fv_offset(ele_index); // offset in FV plaintext
    std::cout << "Main: element index = " << ele_index << " from [0, "
              << number_of_items - 1 << "]" << std::endl;
    std::cout << "Main: FV index = " << index << ", FV offset = " << offset
              << std::endl;

    PirQuery query = client.generate_query(index);
    std::cout << "Main: query generated" << std::endl;

    std::cout << "Main: Initializing server" << std::endl;
    PIRServer server(context, pir_params);

    std::cout << "Main: Setting galois_keys" << std::endl;
    server.set_galois_key(0, galois_keys);

    uint64_t n_i = pir_params.nvec[0];
    cudaStream_t stream = cudaStreamDefault;
    std::vector<heongpu::Ciphertext> expanded_query =
        server.expand_query(query[0][0], n_i, 0, stream);

    std::cout << "Main: query expanded" << std::endl;

    assert(expanded_query.size() == n_i);
    std::cout << "n_i: " << n_i << std::endl;

    std::cout << "Main: checking expansion" << std::endl;
    for (size_t i = 0; i < expanded_query.size(); i++)
    {
        heongpu::Plaintext decryption = client.decrypt(expanded_query[i]);
        std::vector<Data64> decryption_cpu(N);
        decryption.get_data(decryption_cpu);

        bool all_zero =
            std::all_of(decryption_cpu.begin(), decryption_cpu.end(),
                        [](Data64 i) { return i == 0; });
        int all_sum =
            std::accumulate(decryption_cpu.begin(), decryption_cpu.end(), 0);

        if (all_zero && index != i)
        {
            continue;
        }
        else if (all_zero)
        {
            std::cout << "Found zero where index should be" << std::endl;
            return -1;
        }
        else if (all_sum != 1)
        {
            std::cout << "Query vector at index " << index
                      << " should be 1 but is instead " << all_sum << std::endl;
            return -1;
        }
        else
        {
            std::cout << "Query vector at index " << index << " is " << all_sum
                      << std::endl;
        }
    }

    return EXIT_SUCCESS;
}