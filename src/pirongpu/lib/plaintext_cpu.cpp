#include "plaintext_cpu.hpp"

namespace heoncpu
{
     Plaintext::Plaintext() {}

     Plaintext::Plaintext(Parameters& context )
    {
        scheme_ = context.scheme_;
        switch (static_cast<int>(context.scheme_))
        {
            case 1: // BFV
                plain_size_ = context.n;
                depth_ = 0;
                scale_ = 0;
                in_ntt_domain_ = false;
                break;
            default:
                break;
        }

        // if (storage_type_ == storage_type::DEVICE)
        // {
        //     device_locations_ =
        //         DeviceVector<Data64>(plain_size_, options.stream_);
        // }
        // else
        {
            host_locations_ = std::vector<Data64>(plain_size_);
        }
    }

     Plaintext::Plaintext(const std::vector<Data64>& plain,
                                  Parameters& context)
    {
        scheme_ = context.scheme_;
        switch (static_cast<int>(context.scheme_))
        {
            case 1: // BFV
                plain_size_ = context.n;
                depth_ = 0;
                scale_ = 0;
                in_ntt_domain_ = false;

                if (!(plain.size() == plain_size_))
                {
                    throw std::invalid_argument(
                        "Plaintext size should be valid!");
                }

                break;
            
            default:
                break;
        }

        // if (storage_type_ == storage_type::DEVICE)
        // {
        //     device_locations_ =
        //         DeviceVector<Data64>(plain_size_, options.stream_);

        //     cudaMemcpyAsync(device_locations_.data(), plain.data(),
        //                     plain_size_ * sizeof(Data64),
        //                     cudaMemcpyHostToDevice, options.stream_);
        //     HEONGPU_CUDA_CHECK(cudaGetLastError());
        // }
        // else
        {
            host_locations_ = std::vector<Data64>(plain_size_);
            std::memcpy(host_locations_.data(), plain.data(),
                        plain.size() * sizeof(Data64));
        }
    }

    //  Plaintext::Plaintext(const std::vector<Data64>& plain,
    //                               Parameters& context )
    // {
    //     scheme_ = context.scheme_;
    //     switch (static_cast<int>(context.scheme_))
    //     {
    //         case 1: // BFV
    //             plain_size_ = context.n;
    //             depth_ = 0;
    //             scale_ = 0;
    //             in_ntt_domain_ = false;
    //             // storage_type_ = options.storage_;

    //             if (!(plain.size() == plain_size_))
    //             {
    //                 throw std::invalid_argument(
    //                     "Plaintext size should be valid!");
    //             }

    //             break;
            // case 2: // CKKS
            //     plain_size_ = context.n * context.Q_size; // n
            //     depth_ = 0;
            //     scale_ = 0;
            //     in_ntt_domain_ = true;
            //     storage_type_ = options.storage_;

            //     if (!(plain.size() == plain_size_))
            //     {
            //         throw std::invalid_argument(
            //             "Plaintext size should be valid!");
            //     }

            //     break;
        //     default:
        //         break;
        // }

        // if (storage_type_ == storage_type::DEVICE)
        // {
        //     device_locations_ =
        //         DeviceVector<Data64>(plain_size_, options.stream_);

        //     cudaMemcpyAsync(device_locations_.data(), plain.data(),
        //                     plain_size_ * sizeof(Data64),
        //                     cudaMemcpyHostToDevice, options.stream_);
        //     HEONGPU_CUDA_CHECK(cudaGetLastError());
        // }
        // else
    //     {
    //         host_locations_ = std::vector<Data64>(plain_size_);
    //         std::memcpy(host_locations_.data(), plain.data(),
    //                     plain.size() * sizeof(Data64));
    //     }
    // }

    // void Plaintext::store_in_device(cudaStream_t stream)
    // {
    //     if (storage_type_ == storage_type::DEVICE)
    //     {
    //         // pass
    //     }
    //     else
    //     {
    //         if (memory_size() == 0)
    //         {
    //             // pass
    //         }
    //         else
    //         {
    //             device_locations_ =
    //                 DeviceVector<Data64>(host_locations_, stream);
    //             host_locations_.resize(0);
    //             host_locations_.shrink_to_fit();
    //         }

    //         storage_type_ = storage_type::DEVICE;
    //     }
    // }

    // void Plaintext::store_in_host(cudaStream_t stream)
    // {
    //     if (storage_type_ == storage_type::DEVICE)
    //     {
    //         if (memory_size() == 0)
    //         {
    //             // pass
    //         }
    //         else
    //         {
    //             host_locations_ = HostVector<Data64>(plain_size_);
    //             cudaMemcpyAsync(host_locations_.data(),
    //                             device_locations_.data(),
    //                             plain_size_ * sizeof(Data64),
    //                             cudaMemcpyDeviceToHost, stream);
    //             HEONGPU_CUDA_CHECK(cudaGetLastError());

    //             device_locations_.resize(0, stream);
    //             device_locations_.shrink_to_fit(stream);
    //         }

    //         storage_type_ = storage_type::HOST;
    //     }
    //     else
    //     {
    //         // pass
    //     }
    // }

    Data64* Plaintext::data()
    {
        // if (storage_type_ == storage_type::DEVICE)
        // {
        //     return device_locations_.data();
        // }
        // else
        {
            return host_locations_.data();
        }
    }

    void Plaintext::get_data(std::vector<Data64>& plain)
    {
        if (plain.size() < plain_size_)
        {
            plain.resize(plain_size_);
        }

        // if (storage_type_ == storage_type::DEVICE)
        // {
        //     cudaMemcpyAsync(plain.data(), device_locations_.data(),
        //                     plain_size_ * sizeof(Data64),
        //                     cudaMemcpyDeviceToHost, stream);
        //     HEONGPU_CUDA_CHECK(cudaGetLastError());
        // }
        // else
        {
            std::memcpy(plain.data(), host_locations_.data(),
                        host_locations_.size() * sizeof(Data64));
        }
    }

    void Plaintext::set_data(const std::vector<Data64>& plain )
    {
        if (!(plain.size() == plain_size_))
        {
            throw std::invalid_argument("Plaintext size should be valid!");
        }

        // if (storage_type_ == storage_type::DEVICE)
        // {
        //     cudaMemcpyAsync(device_locations_.data(), plain.data(),
        //                     plain_size_ * sizeof(Data64),
        //                     cudaMemcpyHostToDevice, options.stream_);
        //     HEONGPU_CUDA_CHECK(cudaGetLastError());
        // }
        // else
        {
            std::memcpy(host_locations_.data(), plain.data(),
                        plain.size() * sizeof(Data64));
        }
    }

    // void Plaintext::get_data(std::vector<Data64>& plain)
    // {
    //     if (plain.size() < plain_size_)
    //     {
    //         plain.resize(plain_size_);
    //     }

        // if (storage_type_ == storage_type::DEVICE)
        // {
        //     cudaMemcpyAsync(plain.data(), device_locations_.data(),
        //                     plain_size_ * sizeof(Data64),
        //                     cudaMemcpyDeviceToHost, stream);
        //     HEONGPU_CUDA_CHECK(cudaGetLastError());
        // }
        // else
    //     {
    //         std::memcpy(plain.data(), host_locations_.data(),
    //                     host_locations_.size() * sizeof(Data64));
    //     }
    // }

    // void Plaintext::set_data(const std::vector<Data64>& plain )
    // {
    //     if (!(plain.size() == plain_size_))
    //     {
    //         throw std::invalid_argument("Plaintext size should be valid!");
    //     }

        // if (storage_type_ == storage_type::DEVICE)
        // {
        //     cudaMemcpyAsync(device_locations_.data(), plain.data(),
        //                     plain_size_ * sizeof(Data64),
        //                     cudaMemcpyHostToDevice, options.stream_);
        //     HEONGPU_CUDA_CHECK(cudaGetLastError());
        // }
        // else
    //     {
    //         std::memcpy(host_locations_.data(), plain.data(),
    //                     plain.size() * sizeof(Data64));
    //     }
    // }

    int Plaintext::memory_size()
    {
        // if (storage_type_ == storage_type::DEVICE)
        // {
        //     return device_locations_.size();
        // }
        // else
        {
            return host_locations_.size();
        }
    }

    void Plaintext::memory_clear()
    {
        // if (device_locations_.size() > 0)
        // {
        //     device_locations_.resize(0, stream);
        //     device_locations_.shrink_to_fit(stream);
        // }

        if (host_locations_.size() > 0)
        {
            host_locations_.resize(0);
            host_locations_.shrink_to_fit();
        }
    }

    // void Plaintext::memory_set(DeviceVector<Data64>&& new_device_vector)
    // {
    //     storage_type_ = storage_type::DEVICE;
    //     device_locations_ = std::move(new_device_vector);

    //     if (host_locations_.size() > 0)
    //     {
    //         host_locations_.resize(0);
    //         host_locations_.shrink_to_fit();
    //     }
    // }

    // void Plaintext::copy_to_device(cudaStream_t stream)
    // {
    //     if (storage_type_ == storage_type::DEVICE)
    //     {
    //         // pass
    //     }
    //     else
    //     {
    //         if (memory_size() == 0)
    //         {
    //             // pass
    //         }
    //         else
    //         {
    //             device_locations_ =
    //                 DeviceVector<Data64>(host_locations_, stream);
    //         }

    //         storage_type_ = storage_type::DEVICE;
    //     }
    // }

    // void Plaintext::remove_from_device(cudaStream_t stream)
    // {
    //     if (storage_type_ == storage_type::DEVICE)
    //     {
    //         device_locations_.resize(0, stream);
    //         device_locations_.shrink_to_fit(stream);

    //         storage_type_ = storage_type::HOST;
    //     }
    //     else
    //     {
    //         // pass
    //     }
    // }

    void Plaintext::remove_from_host()
    {
        // if (storage_type_ == storage_type::DEVICE)
        // {
        //     // pass
        // }
        // else
        {
            host_locations_.resize(0);
            host_locations_.shrink_to_fit();

            // storage_type_ = storage_type::DEVICE;
        }
    }

} // namespace heongpu