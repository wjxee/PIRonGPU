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
    uint64_t number_of_items = 1 << 10;
    uint64_t size_per_item = 288; // 1024; // in bytes
    uint32_t N = 4096;

    uint32_t d = 1;
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

    // std::cout << "Main: Generating galois keys for client" << std::endl;
    // heongpu::Galoiskey galois_keys = client.generate_galois_keys();

    std::random_device rd;
    uint64_t ele_index =
        1000; // rd() % number_of_items; // element in DB at random position
    uint64_t index = client.get_fv_index(ele_index); // index of FV plaintext
    uint64_t offset = client.get_fv_offset(ele_index); // offset in FV plaintext
    std::cout << "Main: element index = " << ele_index << " from [0, "
              << number_of_items - 1 << "]" << std::endl;
    std::cout << "Main: FV index = " << index << ", FV offset = " << offset
              << std::endl;

    // PirQuery query = client.generate_query(index);
    heongpu::Ciphertext one_ct = client.get_one();
    std::cout << "Main: query generated" << std::endl;

    /////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////

    std::cout << "Main: Initializing server" << std::endl;
    PIRServer server(context, pir_params);

    // std::cout << "Main: Setting galois_keys" << std::endl;
    // server.set_galois_key(0, galois_keys);

    std::cout << "Main: Initializing database" << std::endl;
    auto db(std::make_unique<uint8_t[]>(number_of_items * size_per_item));
    auto db_copy(std::make_unique<uint8_t[]>(number_of_items * size_per_item));

// std::random_device rd;
#pragma omp parallel
    {
        // std::mt19937 gen(rd() + omp_get_thread_num());
        std::mt19937 gen(0 + omp_get_thread_num());
        std::uniform_int_distribution<uint8_t> dis(0, 255);

#pragma omp for
        for (uint64_t i = 0; i < number_of_items; i++)
        {
            for (uint64_t j = 0; j < size_per_item; j++)
            {
                uint8_t val = dis(gen);
                db.get()[(i * size_per_item) + j] = val;
                db_copy.get()[(i * size_per_item) + j] = val;
            }
        }
    }

    std::cout << "Main: Setting database" << std::endl;
    server.set_database(std::move(db), number_of_items, size_per_item);

    std::cout << "Main: Preprocessing database" << std::endl;
    server.preprocess_database();

    std::cout << "Main: Generating reply" << std::endl;
    // PirReply reply = server.generate_reply(query, 0);
    server.set_one_ct(one_ct);

    heongpu::Ciphertext reply = server.simple_query(index);

    /////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////

    // Measure response extraction
    heongpu::Plaintext res_x = client.decrypt(reply);
    std::vector<uint8_t> elems = client.extract_bytes(res_x, offset);

    assert(elems.size() == size_per_item);

    bool failed = false;
    // Check that we retrieved the correct element
    for (uint32_t i = 0; i < size_per_item; i++)
    {
        if (elems[i] != db_copy.get()[(ele_index * size_per_item) + i])
        {
            std::cout << "Main: elems " << (int) elems[i] << ", db "
                      << (int) db_copy.get()[(ele_index * size_per_item) + i]
                      << std::endl;
            std::cout << "Main: PIR result wrong at " << i << std::endl;
            failed = true;
        }
    }
    if (failed)
    {
        return -1;
    }
    std::cout << "Main: PIR result correct!" << std::endl;

    return EXIT_SUCCESS;
}