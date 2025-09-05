#pragma once

#include "heoncpu.hpp"

#include <string>
#include <iomanip>
#include <omp.h>
#include <assert.h>

namespace heoncpu
{   

    // class CudaException : public std::exception
    // {
    //   public:
    //     CudaException(const std::string& file, int line, cudaError_t error)
    //         : file_(file), line_(line), error_(error)
    //     {
    //     }

    //     const char* what() const noexcept override
    //     {
    //         return m_error_string.c_str();
    //     }

    //   private:
    //     std::string file_;
    //     int line_;
    //     cudaError_t error_;
    //     std::string m_error_string = "CUDA Error in " + file_ + " at line " +
    //                                  std::to_string(line_) + ": " +
    //                                  cudaGetErrorString(error_);
    // };

    typedef std::vector<Plaintext> Database;
    typedef std::vector<std::vector<Ciphertext>> PirQuery;
    typedef std::vector<Ciphertext> PirReply;

    struct PirParams
    {
        bool enable_symmetric;
        bool enable_batching;
        bool enable_mswitching;
        std::uint64_t ele_num;
        std::uint64_t ele_size;
        std::uint64_t elements_per_plaintext;
        std::uint64_t num_of_plaintexts; // number of plaintexts in database
        std::uint32_t d; // number of dimensions for the database
        std::uint32_t expansion_ratio; // ratio of ciphertext to plaintext
        std::vector<std::uint64_t> nvec; // size of each of the d dimensions
        std::uint32_t slot_count;
    };

    void gen_pir_params(uint64_t ele_num, uint64_t ele_size, uint32_t d,
                        const Parameters& params,
                        PirParams& pir_params, bool enable_symmetric = false,
                        bool enable_batching = true,
                        bool enable_mswitching = true);

    std::uint64_t plaintexts_per_db(std::uint32_t logt, std::uint64_t N,
                                    std::uint64_t ele_num,
                                    std::uint64_t ele_size);

    std::uint64_t elements_per_ptxt(std::uint32_t logt, std::uint64_t N,
                                    std::uint64_t ele_size);

    std::uint64_t coefficients_per_element(std::uint32_t logt,
                                           std::uint64_t ele_size);

    std::vector<std::uint64_t> bytes_to_coeffs(std::uint32_t limit,
                                               const std::uint8_t* bytes,
                                               std::uint64_t size);

    void coeffs_to_bytes(std::uint32_t limit,
                         const std::vector<std::uint64_t>& coeffs,
                         std::uint8_t* output, std::uint32_t size_out,
                         std::uint32_t ele_size);

    std::vector<std::uint64_t> compute_indices(std::uint64_t desiredIndex,
                                               std::vector<std::uint64_t> nvec);

    uint32_t compute_expansion_ratio(Modulus64& plain_mod,
                                     std::vector<Modulus64>& coeff_mod);

    void compose_to_ciphertext(int poly_modulus_degree, Modulus64& plain_mod,
                               std::vector<Modulus64>& coeff_mod,
                               std::vector<Plaintext>& pts,
                               Ciphertext& ct);

    // __global__ void compose_to_ciphertext_piece(Data64* ct, Data64* plain,
    //                                             int shift);

    // __global__ void decompose_to_plaintexts_piece(Data64* ct, Data64* plain,
    //                                               int shift,
    //                                               uint64_t pt_bitmask);

} // namespace pirongpu
 