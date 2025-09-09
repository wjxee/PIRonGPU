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

#ifndef PIR_CLIENT_H
#define PIR_CLIENT_H

#include "pir.cuh"

namespace pirongpu
{

    class PIRClient
    {
      public:
        PIRClient(std::shared_ptr<heongpu::Parameters>& context,
                  const PirParams& pirparams);

        heongpu::Galoiskey generate_galois_keys();

        uint64_t get_fv_index(uint64_t element_index);
        uint64_t get_fv_offset(uint64_t element_index);

        PirQuery generate_query(std::uint64_t desiredIndex);

        heongpu::Plaintext decode_reply(PirReply& reply);

        std::vector<uint8_t> decode_reply(PirReply& reply, uint64_t offset);

        std::vector<uint8_t> extract_bytes(heongpu::Plaintext& pt,
                                           std::uint64_t offset);

        heongpu::Plaintext decrypt(heongpu::Ciphertext ct);

        // Only used for simple_query
        heongpu::Ciphertext get_one();

      // private:
        PirParams pir_params_;

        std::shared_ptr<heongpu::Secretkey> secret_key_;
        std::shared_ptr<heongpu::Publickey> public_key_;

        std::shared_ptr<heongpu::Parameters> context_;
        std::shared_ptr<heongpu::HEEncoder> encoder_;
        std::shared_ptr<heongpu::HEEncryptor> encryptor_;
        std::shared_ptr<heongpu::HEDecryptor> decryptor_;
        std::shared_ptr<heongpu::HEArithmeticOperator> evaluator_;
        std::shared_ptr<heongpu::HEKeyGenerator> keygen_;

        std::vector<uint64_t> indices_; // the indices for retrieval.
        std::vector<uint64_t> inverse_scales_;
    };

} // namespace pirongpu

#endif // PIR_CLIENT_H
