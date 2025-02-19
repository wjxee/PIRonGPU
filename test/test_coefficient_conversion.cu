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
#include <bitset>

using namespace pirongpu;

int main(int argc, char* argv[])
{
    const uint32_t logt = 16;
    const uint32_t ele_size = 3;
    const uint32_t num_ele = 3;
    uint8_t bytes[ele_size * num_ele] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    std::vector<uint64_t> coeffs;

    std::cout << "Coeffs: " << std::endl;
    for (int i = 0; i < num_ele; i++)
    {
        std::vector<uint64_t> ele_coeffs =
            bytes_to_coeffs(logt, bytes + (i * ele_size), ele_size);
        for (int j = 0; j < ele_coeffs.size(); j++)
        {
            std::cout << ele_coeffs[j] << std::endl;
            std::cout << std::bitset<logt>(ele_coeffs[j]) << std::endl;
            coeffs.push_back(ele_coeffs[j]);
        }
    }

    std::cout << "Num of Coeffs: " << coeffs.size() << std::endl;

    uint8_t output[ele_size * num_ele];
    coeffs_to_bytes(logt, coeffs, output, ele_size * num_ele, ele_size);

    std::cout << "Bytes: " << std::endl;
    for (int i = 0; i < ele_size * num_ele; i++)
    {
        std::cout << (int) output[i] << std::endl;
    }

    return 0;
}
