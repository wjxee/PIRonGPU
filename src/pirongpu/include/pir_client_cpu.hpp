#pragma once

#include "pir_cpu.hpp"

namespace pironcpu
{

    class PIRClient
    {
      public:
        PIRClient(std::shared_ptr<Parameters>& context,
                  const PirParams& pirparams);

        // Galoiskey generate_galois_keys();

        uint64_t get_fv_index(uint64_t element_index);
        uint64_t get_fv_offset(uint64_t element_index);

        PirQuery generate_query(std::uint64_t desiredIndex);

        Plaintext decode_reply(PirReply& reply);

        std::vector<uint8_t> decode_reply(PirReply& reply, uint64_t offset);

        std::vector<uint8_t> extract_bytes(Plaintext& pt,
                                           std::uint64_t offset);

        Plaintext decrypt(Ciphertext ct);

        // Only used for simple_query
        Ciphertext get_one();

      private:
        PirParams pir_params_;

        std::shared_ptr<Secretkey> secret_key_;
        std::shared_ptr<Publickey> public_key_;

        std::shared_ptr<Parameters> context_;
        std::shared_ptr<HEEncoder> encoder_;
        std::shared_ptr<HEEncryptor> encryptor_;
        std::shared_ptr<HEDecryptor> decryptor_;
        std::shared_ptr<HEArithmeticOperator> evaluator_;
        std::shared_ptr<HEKeyGenerator> keygen_;

        std::vector<uint64_t> indices_; // the indices for retrieval.
        std::vector<uint64_t> inverse_scales_;
    };

} // namespace pirongpu
 