#pragma once 
#include "context_cpu.hpp"

namespace heoncpu{
    void decode_kernel_bfv_cpu(
        Data64* message, Data64* message_encoded,
        Data64* location_info,
        int grid_x, int grid_y, int grid_z, int block_size) ;
    
}