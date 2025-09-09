#include "pir_client_cpu.hpp"

namespace heoncpu
{

    PIRClient::PIRClient(std::shared_ptr<Parameters>& context,
                         const PirParams& pir_params)
        : context_(context), pir_params_(pir_params)
    {
        keygen_ = std::make_shared<HEKeyGenerator>(*context_);

        secret_key_ = std::make_shared<Secretkey>(*context_);
        keygen_->generate_secret_key(*secret_key_);

        public_key_ = std::make_shared<Publickey>(*context_);
        keygen_->generate_public_key(*public_key_, *secret_key_);

        if (pir_params_.enable_symmetric)
        {
            throw std::invalid_argument(
                "Symmetric Encryption is not supported!");
        }
        else
        {
            encryptor_ =
                std::make_shared<HEEncryptor>(*context_, *public_key_);
        }

        encoder_ = std::make_shared<HEEncoder>(*context_);
        decryptor_ =
            std::make_shared<HEDecryptor>(*context_, *secret_key_);
        evaluator_ = std::make_shared<HEArithmeticOperator>(*context_,
                                                                     *encoder_);
    }

    Galoiskey PIRClient::generate_galois_keys()
    {
        std::vector<uint32_t> galois_elts;
        int N = context_->poly_modulus_degree();
        int logN = context_->log_poly_modulus_degree();

        for (int i = 0; i < logN; i++)
        {
            galois_elts.push_back((N + int(pow(2, i))) / int(pow(2, i)));
        }
        // std::cout << "gk:";
        // for (int i = 0; i < logN; i++)
        // {
        //     std::cout << galois_elts[i] <<";";
        // }
        // std::cout << ";" << std::endl;
        Galoiskey galois_key(*context_, galois_elts);
        keygen_->generate_galois_key(galois_key, *secret_key_);
        return galois_key;
    }

    uint64_t PIRClient::get_fv_index(uint64_t element_index)
    {
        return static_cast<uint64_t>(element_index /
                                     pir_params_.elements_per_plaintext);
    }

    uint64_t PIRClient::get_fv_offset(uint64_t element_index)
    {
        return element_index % pir_params_.elements_per_plaintext;
    }

    PirQuery PIRClient::generate_query(uint64_t desiredIndex)
    {
        indices_ = compute_indices(desiredIndex, pir_params_.nvec);
        PirQuery result(pir_params_.d);

        int N = context_->poly_modulus_degree();
        Modulus64 plain_modulus = context_->plain_modulus();
        for (uint32_t i = 0; i < indices_.size(); i++)
        {
            uint32_t num_ptxts = ceil((pir_params_.nvec[i] + 0.0) / N);
            // initialize result.
            std::cout << "Client: index " << i + 1 << "/ " << indices_.size()
                      << " = " << indices_[i] << std::endl;
            std::cout << "Client: number of ctxts needed for query = "
                      << num_ptxts << std::endl;

            for (uint32_t j = 0; j < num_ptxts; j++)
            {
                std::vector<Data64> pt_vector(N, 0ULL);
                if (indices_[i] >= N * j && indices_[i] <= N * (j + 1))
                {
                    uint64_t real_index = indices_[i] - N * j;
                    uint64_t n_i = pir_params_.nvec[i];
                    uint64_t total = N;
                    if (j == num_ptxts - 1)
                    {
                        total = n_i % N;
                    }
                    uint64_t log_total = ceil(log2(total));
                    std::cout << "Client: Inverting " << pow(2, log_total)
                              << std::endl;

                    Data64 pow_ = pow(2, log_total);
                    pt_vector[real_index] =
                        OPERATOR64::modinv(pow_, plain_modulus);
                }
                Plaintext pt(pt_vector, *context_);
                Ciphertext dest(*context_);

                if (pir_params_.enable_symmetric)
                {
                    throw std::invalid_argument(
                        "Symmetric Encryption is not supported!");
                }
                else
                {
                    encryptor_->encrypt(dest, pt);
                }
                result[i].push_back(dest);
            }
        }

        return result;
    }

    // std::vector<uint8_t> PIRClient::extract_bytes(Plaintext& pt,
    //                                               uint64_t offset)
    // {
    //     uint32_t N = context_->poly_modulus_degree();
    //     uint32_t logt = floor(log2(context_->plain_modulus().value));
    //     uint32_t bytes_per_ptxt =
    //         pir_params_.elements_per_plaintext * pir_params_.ele_size;

    //     // Convert from FV plaintext (polynomial) to database element at the
    //     // client
    //     std::vector<uint8_t> elems(bytes_per_ptxt);
    //     std::vector<uint64_t> coeffs;
    //     // Message coeffs_gpu(*context_);
    //     encoder_->decode(coeffs, pt);
    //     // coeffs_gpu.device_to_host(coeffs);

    //     coeffs_to_bytes(logt, coeffs, elems.data(), bytes_per_ptxt,
    //                     pir_params_.ele_size);
    //     return std::vector<uint8_t>(
    //         elems.begin() + offset * pir_params_.ele_size,
    //         elems.begin() + (offset + 1) * pir_params_.ele_size);
    // }

    // Plaintext PIRClient::decode_reply(PirReply& reply)
    // {
    //     // EncryptionParameters parms;
    //     // parms_id_type parms_id;
    //     if (pir_params_.enable_mswitching)
    //     {
    //         throw std::invalid_argument("Modulus switching is not supported!");
    //     }
    //     else
    //     {
    //     }

    //     std::vector<Modulus64> vec_key_modulus = context_->key_modulus();

    //     std::vector<Modulus64> coeff_mod_;
    //     coeff_mod_.assign(vec_key_modulus.begin(), vec_key_modulus.end() - 1);
    //     Modulus64 mod_plain = context_->plain_modulus();
    //     uint32_t exp_ratio = compute_expansion_ratio(mod_plain, coeff_mod_);
    //     uint32_t recursion_level = pir_params_.d;

    //     std::cout << "exp_ratio: " << exp_ratio << std::endl;

    //     std::vector<Ciphertext> temp = reply;
    //     uint32_t ciphertext_size = 2;
    //     std::cout << "ciphertext_size: " << ciphertext_size << std::endl;

    //     for (uint32_t i = 0; i < recursion_level; i++)
    //     {
    //         std::cout << "Client: " << i + 1 << "/ " << recursion_level
    //                   << "-th decryption layer started." << std::endl;
    //         std::vector<Ciphertext> newtemp;
    //         std::vector<Plaintext> tempplain;

    //         for (uint32_t j = 0; j < temp.size(); j++)
    //         {
    //             Plaintext ptxt(*context_);
    //             decryptor_->decrypt(ptxt, temp[j]);
    //             tempplain.push_back(ptxt);

    //             if ((j + 1) % (exp_ratio * ciphertext_size) == 0 && j > 0)
    //             {
    //                 // Combine into one ciphertext.
    //                 Ciphertext combined(*context_);

    //                 compose_to_ciphertext(context_->poly_modulus_degree(),
    //                                       mod_plain, vec_key_modulus, tempplain,
    //                                       combined);
    //                 newtemp.push_back(combined);
    //                 tempplain.clear();
    //             }
    //         }
    //         std::cout << "Client: done." << std::endl;
    //         std::cout << std::endl;
    //         if (i == recursion_level - 1)
    //         {
    //             assert(temp.size() == 1);
    //             return tempplain[0];
    //         }
    //         else
    //         {
    //             tempplain.clear();
    //             temp = newtemp;
    //         }
    //     }

    //     // This should never be called
    //     assert(0);
    //     Plaintext fail;
    //     return fail;
    // }

    // std::vector<uint8_t> PIRClient::decode_reply(PirReply& reply,
    //                                              uint64_t offset)
    // {
    //     Plaintext result = decode_reply(reply);
    //     return extract_bytes(result, offset);
    // }

    // Plaintext PIRClient::decrypt(Ciphertext ct)
    // {
    //     Plaintext pt(*context_);
    //     decryptor_->decrypt(pt, ct);
    //     return pt;
    // }

    // Ciphertext PIRClient::get_one()
    // {
    //     std::vector<Data64> pt_vector(context_->poly_modulus_degree(), 0ULL);
    //     pt_vector[0] = 1;

    //     Plaintext pt(pt_vector, *context_);
    //     Ciphertext dest(*context_);

    //     encryptor_->encrypt(dest, pt);

    //     return dest;
    // }

} // namespace pirongpu