#pragma once

#include <string>
#include <iomanip>
#include <omp.h>
#include <vector>
#include <iostream>

#include "context_cpu.hpp"
#include "plaintext_cpu.hpp"
#include "ciphertext_cpu.hpp"
#include "secretkey_cpu.hpp"
#include "publickey_cpu.hpp"
#include "keygenerator_cpu.hpp"
#include "encoder_cpu.hpp"
#include "encryptor_cpu.hpp"
#include "decryptor_cpu.hpp"
#include "operator_cpu.hpp"
#include "keyswitch_cpu.hpp"



namespace heoncpu
{
    inline void helloworld(){
        std::cout << "hello world heoncpu!" << std::endl;
    }
} // namespace heongpu