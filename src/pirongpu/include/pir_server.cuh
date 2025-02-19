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

#ifndef PIR_SERVER_H
#define PIR_SERVER_H

#include "pir.cuh"
#include "pir_client.cuh"
#include <map>
#include <memory>
#include <vector>
#include <iomanip>

namespace pirongpu
{

    class PIRServer
    {
        friend class PIRClient;

      public:
        PIRServer(std::shared_ptr<heongpu::Parameters>& context,
                  const PirParams& pir_params);

        void set_database(const std::unique_ptr<const std::uint8_t[]>& bytes,
                          std::uint64_t ele_num, std::uint64_t ele_size);

        void
        set_database(std::unique_ptr<std::vector<heongpu::Plaintext>>&& db);

        void set_galois_key(std::uint32_t client_id, heongpu::Galoiskey galkey);

        void preprocess_database();

        PirReply generate_reply(PirQuery& query, std::uint32_t client_id,
                                cudaStream_t& stream);

        std::vector<heongpu::Ciphertext>
        expand_query(const heongpu::Ciphertext& encrypted, std::uint32_t m,
                     std::uint32_t client_id, cudaStream_t& stream);

        void set_one_ct(heongpu::Ciphertext one);

        heongpu::Ciphertext simple_query(std::uint64_t index);
        heongpu::Plaintext simple_query2(uint64_t index);

      private:
        PirParams pir_params_;
        std::unique_ptr<Database> db_;
        bool is_db_preprocessed_;

        std::map<int, heongpu::Galoiskey> galoisKeys_;
        std::shared_ptr<heongpu::HEArithmeticOperator> evaluator_;
        std::shared_ptr<heongpu::HEEncoder> encoder_;
        std::shared_ptr<heongpu::Parameters> context_;

        heongpu::Ciphertext one_;
    };

} // namespace pirongpu

#endif // PIR_SERVER_H