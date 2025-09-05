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
#include "pir_cpu.hpp"
#include "pir_client.cuh"
#include "pir_client_cpu.hpp"
#include "pir_server.cuh"
#include "heoncpu.hpp"
#include <omp.h>
#include <iostream>
#include <stdexcept>
#include <iterator>

using namespace pirongpu;

int main(int argc, char* argv[])
{
    heoncpu::helloworld();
    cudaSetDevice(0); // Use it for memory pool

    uint64_t number_of_items = 1 << 16;
    uint64_t size_per_item = 288; // in bytes
    uint32_t N = 1 << 12;

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
    //cpu
    std::shared_ptr<heoncpu::Parameters> context_cpu =
        std::make_shared<heoncpu::Parameters>(
            heoncpu::scheme_type::bfv,
            heoncpu::keyswitching_type::KEYSWITCHING_METHOD_I);

    context->set_poly_modulus_degree(N);
    context->set_coeff_modulus(log_Q_bases_bit_sizes, log_P_bases_bit_sizes);
    context->set_plain_modulus(plain_modulus);
    context->generate();
    //cpu
    context_cpu->set_poly_modulus_degree(N);
    context_cpu->set_coeff_modulus(log_Q_bases_bit_sizes, log_P_bases_bit_sizes);
    context_cpu->set_plain_modulus(plain_modulus);
    context_cpu->generate();

    //test consistence
    {
    int checkflag=0;
    int size = context_cpu->intt_table_->size();
    std::vector<Root64> hv(size);
    cudaMemcpy(hv.data(), context->intt_table_->data(), size * sizeof(Root64), cudaMemcpyDeviceToHost);
    for(int i=0;i<context_cpu->intt_table_->size();i++){
        if(hv[i]!=context_cpu->intt_table_->at(i)){
            checkflag+=1;
        }
        // if(*(context->intt_table_->element_ptr(i))!=context_cpu->intt_table_->at(i)){
        //     checkflag+=1;
        // }
    }
    if (checkflag!=0)
        std::cout << "context wrong " << checkflag << std::endl;
    else   
        std::cout << "context all correct " << checkflag << std::endl;

    }

    std::cout << "Main: Generating PIR parameters" << std::endl;
    PirParams pir_params;
    gen_pir_params(number_of_items, size_per_item, d, *context, pir_params,
                   use_symmetric, use_batching, use_recursive_mod_switching);
    //cpu
    heoncpu::PirParams pir_params_cpu;
    heoncpu::gen_pir_params(number_of_items, size_per_item, d, *context_cpu, pir_params_cpu,
                   use_symmetric, use_batching, use_recursive_mod_switching);

    //test consistence
    {
        int checkflag=0; 
        if(pir_params.expansion_ratio!=pir_params_cpu.expansion_ratio)
            checkflag+=1;
        if (checkflag!=0)
            std::cout << "pir_params wrong " << checkflag << std::endl;
        else   
            std::cout << "pir_params all correct " << checkflag << std::endl;

    }

    std::cout << "Main: Generating PIR client" << std::endl;
    PIRClient client(context, pir_params);
    //cpu
    heoncpu::PIRClient client_cpu(context_cpu, pir_params_cpu);

    std::cout << "Main: Generating galois keys for client" << std::endl;
    heongpu::Galoiskey galois_keys = client.generate_galois_keys();



    ////////////////////////////////
    int query_count = 4; // Total query count
    int num_thread = 4; // Each CPU thread corresponds a GPU Stream.
    assert(query_count >= num_thread);
    std::vector<PirReply> multi_reply(query_count);
    std::vector<cudaStream_t> streams;
    for (int i = 0; i < num_thread; i++)
    {
        cudaStream_t inner_stream;
        cudaStreamCreate(&inner_stream);
        streams.push_back(inner_stream);
    }
    ////////////////////////////////

    std::random_device rd;
    std::vector<uint64_t> ele_index;
    std::vector<uint64_t> index;
    std::vector<uint64_t> offset;
    std::vector<PirQuery> query;
    for (int i = 0; i < query_count; i++)
    {
        uint64_t inner_ele_index =
            rd() % number_of_items; // element in DB at random position
        uint64_t inner_index =
            client.get_fv_index(inner_ele_index); // index of FV plaintext
        uint64_t inner_offset =
            client.get_fv_offset(inner_ele_index); // offset in FV plaintext

        std::cout << "[" << (i + 1) << "/" << query_count
                  << "]: element index = " << inner_ele_index << " from [0, "
                  << number_of_items - 1 << "]" << std::endl;
        std::cout << "[" << (i + 1) << "/" << query_count
                  << "]: element index = " << inner_ele_index << " from [0, "
                  << number_of_items - 1 << "]" << std::endl;
        std::cout << "[" << (i + 1) << "/" << query_count
                  << "]: FV index = " << inner_index
                  << ", FV offset = " << inner_offset << std::endl;

        PirQuery inner_query = client.generate_query(inner_index);
        std::cout << "[" << (i + 1) << "/" << query_count
                  << "]: query generated" << std::endl;

        ele_index.push_back(inner_ele_index);
        index.push_back(inner_index);
        offset.push_back(inner_offset);
        query.push_back(inner_query);
    }

    std::cout << "Main: Initializing server" << std::endl;
    PIRServer server(context, pir_params);

    std::cout << "Main: Setting galois_keys" << std::endl;
    server.set_galois_key(0, galois_keys);

    std::cout << "Main: Initializing database" << std::endl;
    auto db(std::make_unique<uint8_t[]>(number_of_items * size_per_item));
    auto db_copy(std::make_unique<uint8_t[]>(number_of_items * size_per_item));

#pragma omp parallel
    {
        std::mt19937 gen(rd() + omp_get_thread_num());
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
#pragma omp parallel for num_threads(num_thread)
    for (auto i = 0ULL; i < query_count; i++)
    {
        int threadID = omp_get_thread_num();
        multi_reply[i] = server.generate_reply(query[i], 0, streams[threadID]);
    }

    for (int j = 0; j < query_count; j++)
    {
        std::cout << "[" << (j + 1) << "/" << query_count << "]: reply decoding"
                  << std::endl;
        std::vector<uint8_t> elems =
            client.decode_reply(multi_reply[j], offset[j]);

        assert(elems.size() == size_per_item);

        bool failed = false;
        // Check that we retrieved the correct element
        for (uint32_t i = 0; i < size_per_item; i++)
        {
            if (elems[i] != db_copy.get()[(ele_index[j] * size_per_item) + i])
            {
                std::cout
                    << "[" << (j + 1) << "/" << query_count << "]: elems "
                    << (int) elems[i] << ", db "
                    << (int) db_copy.get()[(ele_index[j] * size_per_item) + i]
                    << std::endl;
                std::cout << "[" << (j + 1) << "/" << query_count
                          << "]: PIR result wrong at " << i << std::endl;
                failed = true;
            }
        }
        if (failed)
        {
            throw std::runtime_error("Failed!");
        }
    }

    std::cout << "Main: PIR result correct!" << std::endl;

    return EXIT_SUCCESS;
}