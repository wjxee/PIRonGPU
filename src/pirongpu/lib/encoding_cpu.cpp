#include "encoding_cpu.hpp"

namespace heoncpu{

    void decode_kernel_bfv_cpu(
        Data64* message, Data64* message_encoded,
        Data64* location_info,
        int grid_x, int grid_y, int grid_z, int block_size) {
        
        // 计算总线程数
        int total_threads = grid_x * grid_y * grid_z * block_size;
        
        // 如果没有指定消息长度，我们需要假设一个合理的最大值
        // 在实际应用中，您应该传递消息长度作为参数
        int max_message_length = total_threads;
        
        // 模拟CUDA的线程网格结构
        for (int block_z = 0; block_z < grid_z; block_z++) {
            for (int block_y = 0; block_y < grid_y; block_y++) {
                for (int block_x = 0; block_x < grid_x; block_x++) {
                    for (int thread_id = 0; thread_id < block_size; thread_id++) {
                        // 计算全局线程索引（模拟CUDA的线程索引计算）
                        int idx = ((block_x + block_y * grid_x + block_z * grid_x * grid_y) * block_size) + thread_id;
                        
                        // 确保不超出数组范围
                        if (idx >= max_message_length) continue;
                        
                        // 获取位置信息
                        int location = location_info[idx];
                        
                        // 检查位置是否越界
                        if (location < 0 || location >= max_message_length) {
                            throw std::out_of_range("Location out of range: " + std::to_string(location) + 
                                                " at index: " + std::to_string(idx));
                        }
                        
                        // 执行解码操作
                        message[idx] = message_encoded[location];
                    }
                }
            }
        }
    }
}