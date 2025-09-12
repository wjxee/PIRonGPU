#include "pir.cuh"
#include "pir_cpu.hpp"
#include "pir_client.cuh"
#include "pir_client_cpu.hpp"
#include "pir_server.cuh"
#include "heoncpu.hpp"
#include <omp.h>
#include <iostream>
#include <stdexcept>
#include <iterator>

using namespace pirongpu;

int main(int argc, char* argv[])
{
    heoncpu::helloworld();
    // Data64 U=68719230976;
    // Data64 V=0;
    // const heoncpu::Root<Data64> root(56404117254);
    // const heoncpu::Modulus<Data64> modulus(68719230977);
    // // modulus.value= 68719230977;
    // // modulus.bit=36;
    // // modulus.mu=137439444991;
    // // const Modulus<Data64> modu = modulus;
    // heoncpu::CooleyTukeyUnit<Data64>(U,V,root,modulus);

    cudaSetDevice(0); // Use it for memory pool

    uint64_t number_of_items = 1 << 16;
    uint64_t size_per_item = 288; // in bytes
    uint32_t N = 1 << 12;

    uint32_t d = 2;
    bool use_symmetric = false; // Do not change!
    bool use_batching = true; // Do not change!
    bool use_recursive_mod_switching = false; // Do not change!

    std::vector<int> log_Q_bases_bit_sizes = {36, 36};
    std::vector<int> log_P_bases_bit_sizes = {37};
    int plain_modulus = 1179649;

    std::shared_ptr<heongpu::Parameters> context =
        std::make_shared<heongpu::Parameters>(
            heongpu::scheme_type::bfv,
            heongpu::keyswitching_type::KEYSWITCHING_METHOD_I);
    //cpu
    std::shared_ptr<heoncpu::Parameters> context_cpu =
        std::make_shared<heoncpu::Parameters>(
            heoncpu::scheme_type::bfv,
            heoncpu::keyswitching_type::KEYSWITCHING_METHOD_I);

    context->set_poly_modulus_degree(N);
    context->set_coeff_modulus(log_Q_bases_bit_sizes, log_P_bases_bit_sizes);
    context->set_plain_modulus(plain_modulus);
    context->generate();
    //cpu
    context_cpu->set_poly_modulus_degree(N);
    context_cpu->set_coeff_modulus(log_Q_bases_bit_sizes, log_P_bases_bit_sizes);
    context_cpu->set_plain_modulus(plain_modulus);
    context_cpu->generate();

    //test consistence
    {
    int checkflag=0;
    int size = context_cpu->intt_table_->size();
    std::vector<Root64> hv(size);
    cudaMemcpy(hv.data(), context->intt_table_->data(), size * sizeof(Root64), cudaMemcpyDeviceToHost);
    for(int i=0;i<context_cpu->intt_table_->size();i++){
        if(hv[i]!=context_cpu->intt_table_->at(i)){
            checkflag+=1;
        }
        // if(*(context->intt_table_->element_ptr(i))!=context_cpu->intt_table_->at(i)){
        //     checkflag+=1;
        // }
    }
    if (checkflag!=0)
        std::cout << "context wrong " << checkflag << std::endl;
    else   
        std::cout << "context all correct " << checkflag << std::endl;

    }

    std::cout << "Main: Generating PIR parameters" << std::endl;
    PirParams pir_params;
    gen_pir_params(number_of_items, size_per_item, d, *context, pir_params,
                   use_symmetric, use_batching, use_recursive_mod_switching);
    //cpu
    heoncpu::PirParams pir_params_cpu;
    heoncpu::gen_pir_params(number_of_items, size_per_item, d, *context_cpu, pir_params_cpu,
                   use_symmetric, use_batching, use_recursive_mod_switching);

    //test consistence
    {
        int checkflag=0; 
        if(pir_params.expansion_ratio!=pir_params_cpu.expansion_ratio)
            checkflag+=1;
        if (checkflag!=0)
            std::cout << "pir_params wrong " << checkflag << std::endl;
        else   
            std::cout << "pir_params all correct " << checkflag << std::endl;

    }

    std::cout << "Main: Generating PIR client" << std::endl;
    PIRClient client(context, pir_params);
    //cpu
    heoncpu::PIRClient client_cpu(context_cpu, pir_params_cpu);
    //test consistence
    {
        int checkflag=0;
        int sk_size = client_cpu.secret_key_->secretkey_.size();
        std::vector<int> sk_gpu(sk_size);
        cudaMemcpy(sk_gpu.data(), client.secret_key_->secretkey_.data(), sk_size * sizeof(int), cudaMemcpyDeviceToHost);
        for(int i=0;i<sk_size;i++){
            if(sk_gpu[i]!=client_cpu.secret_key_->secretkey_[i]){
                checkflag+=1;
            } 
        }
        if (checkflag!=0)
            std::cout << "sk wrong " << checkflag << std::endl;
        else   
            std::cout << "sk all correct " << checkflag << std::endl;

    }
    { //test pk
        int checkflag=0;
        int pk_size = client_cpu.public_key_->locations_.size();
        std::vector<Data64> pk_gpu(pk_size);
        cudaMemcpy(pk_gpu.data(), client.public_key_->locations_.data(), pk_size * sizeof(Data64), cudaMemcpyDeviceToHost);
        for(int i=0;i<pk_size;i++){
            if(pk_gpu[i]!=client_cpu.public_key_->locations_[i]){
                checkflag+=1;
            } 
        }
        if (checkflag!=0)
            std::cout << "pk wrong " << checkflag << std::endl;
        else   
            std::cout << "pk all correct " << checkflag << std::endl;

    }
    std::cout << "Main: Generating galois keys for client" << std::endl;
    heongpu::Galoiskey galois_keys = client.generate_galois_keys();
    //cpu
    heoncpu::Galoiskey galois_keys_cpu = client_cpu.generate_galois_keys();
    //test consistence
    {
        // int checkflag=0;
        // int gsize = galois_keys_cpu.custom_galois_elt.size();
        // std::vector<u_int32_t> gk_gpu(gsize);
        //     cudaMemcpy(gk_gpu.data(), galois_keys.custom_galois_elt.data(),
        //      gsize * sizeof(u_int32_t), cudaMemcpyDeviceToHost);
        // for (int i=0;i<gsize;i++){
        //     // int first_ele = pair.first;
        //     // std::vector<Data64> second_ele = pair.second;
        //     // int gk_size = second_ele.size();
        //     // std::vector<Data64> gk_gpu(gk_size);
        //     // cudaMemcpy(gk_gpu.data(), galois_keys.device_location_.at(first_ele).data(), gk_size * sizeof(Data64), cudaMemcpyDeviceToHost);
        //     // for(int i=0;i<gk_size;i++){
        //         if(gk_gpu[i]!=galois_keys_cpu.custom_galois_elt[i]){
                    
        //             checkflag+=1;
        //         } 
        //         std::cout << i << ":" << gk_gpu[i] << "==" <<  galois_keys_cpu.custom_galois_elt[i] <<";";
        //     // }
        // } 
        // if (checkflag!=0)
        //     std::cout << "gk wrong " << checkflag << std::endl;
        // else   
        //     std::cout << "gk all correct " << checkflag << std::endl;
            
        int checkflag=0;
        for (auto& pair: galois_keys_cpu.host_location_){
            int first_ele = pair.first;
            std::vector<Data64> second_ele = pair.second;
            int gk_size = second_ele.size();
            std::vector<Data64> gk_gpu(gk_size);
            cudaMemcpy(gk_gpu.data(), galois_keys.device_location_.at(first_ele).data(), gk_size * sizeof(Data64), cudaMemcpyDeviceToHost);
            for(int i=0;i<gk_size;i++){
                if(gk_gpu[i]!=second_ele[i]){
                    // std::cout << i << ":" << gk_gpu[i] << "==" <<  second_ele[i] <<";" <<std::endl;
                    checkflag+=1;
                } 
            }
        } 
        if (checkflag!=0)
            std::cout << "gk wrong " << checkflag << std::endl;
        else   
            std::cout << "gk all correct " << checkflag << std::endl;

    }
    std::cout << "Initialization Phase Done." << std::endl;

    ////////////////////////////////
    int query_count = 4; // Total query count
    int num_thread = 4; // Each CPU thread corresponds a GPU Stream.
    assert(query_count >= num_thread);
    std::vector<PirReply> multi_reply(query_count);
    //cpu
    std::vector<heoncpu::PirReply> multi_reply_cpu(query_count);
    std::vector<cudaStream_t> streams;
    for (int i = 0; i < num_thread; i++)
    {
        cudaStream_t inner_stream;
        cudaStreamCreate(&inner_stream);
        streams.push_back(inner_stream);
    }
    ////////////////////////////////

    std::random_device rd;
    std::vector<uint64_t> ele_index;
    std::vector<uint64_t> index;
    std::vector<uint64_t> offset;
    std::vector<PirQuery> query;
    //cpu
    std::vector<uint64_t> ele_index_cpu;
    std::vector<uint64_t> index_cpu;
    std::vector<uint64_t> offset_cpu;
    std::vector<heoncpu::PirQuery> query_cpu;
    for (int i = 0; i < query_count; i++)
    {
        // uint64_t inner_ele_index =
        //     rd() % number_of_items; // element in DB at random position
        uint64_t inner_ele_index = i;
        uint64_t inner_index =
            client.get_fv_index(inner_ele_index); // index of FV plaintext
        uint64_t inner_offset =
            client.get_fv_offset(inner_ele_index); // offset in FV plaintext
        //cpu
        uint64_t inner_index_cpu =
            client_cpu.get_fv_index(inner_ele_index); // index of FV plaintext
        uint64_t inner_offset_cpu =
            client_cpu.get_fv_offset(inner_ele_index); // offset in FV plaintext

        std::cout << "[" << (i + 1) << "/" << query_count
                  << "]: element index = " << inner_ele_index << " from [0, "
                  << number_of_items - 1 << "]" << std::endl;
        std::cout << "[" << (i + 1) << "/" << query_count
                  << "]: element index = " << inner_ele_index << " from [0, "
                  << number_of_items - 1 << "]" << std::endl;
        std::cout << "[" << (i + 1) << "/" << query_count
                  << "]: FV index = " << inner_index
                  << ", FV offset = " << inner_offset << std::endl;

        PirQuery inner_query = client.generate_query(inner_index);
        //cpu
        // heoncpu::PirQuery inner_query_cpu = client_cpu.generate_query(inner_index_cpu);
        std::cout << "[" << (i + 1) << "/" << query_count
                  << "]: query generated" << std::endl;

        ele_index.push_back(inner_ele_index);
        index.push_back(inner_index);
        offset.push_back(inner_offset);
        query.push_back(inner_query);
        //cpu
        ele_index_cpu.push_back(inner_ele_index);
        index_cpu.push_back(inner_index_cpu);
        offset_cpu.push_back(inner_offset_cpu);
        // query_cpu.push_back(inner_query_cpu);
    }

    std::cout << "Main: Initializing server" << std::endl;
    PIRServer server(context, pir_params);

    std::cout << "Main: Setting galois_keys" << std::endl;
    server.set_galois_key(0, galois_keys);

    std::cout << "Main: Initializing database" << std::endl;
    auto db(std::make_unique<uint8_t[]>(number_of_items * size_per_item));
    auto db_copy(std::make_unique<uint8_t[]>(number_of_items * size_per_item));

#pragma omp parallel
    {
        std::mt19937 gen(rd() + omp_get_thread_num());
        std::uniform_int_distribution<uint8_t> dis(0, 255);

#pragma omp for
        for (uint64_t i = 0; i < number_of_items; i++)
        {
            for (uint64_t j = 0; j < size_per_item; j++)
            {
                uint8_t val = dis(gen);
                db.get()[(i * size_per_item) + j] = val;
                db_copy.get()[(i * size_per_item) + j] = val;
            }
        }
    }

    std::cout << "Main: Setting database" << std::endl;
    server.set_database(std::move(db), number_of_items, size_per_item);

    std::cout << "Main: Preprocessing database" << std::endl;
    server.preprocess_database();

    std::cout << "Main: Generating reply" << std::endl;
#pragma omp parallel for num_threads(num_thread)
    for (auto i = 0ULL; i < query_count; i++)
    {
        int threadID = omp_get_thread_num();
        multi_reply[i] = server.generate_reply(query[i], 0, streams[threadID]);
        
    }
    //cpu copy
    for (auto i = 0ULL; i < query_count; i++)
    {
        multi_reply_cpu[i].reserve(multi_reply[i].size());
        for (uint32_t a = 0; a < multi_reply[i].size(); a++)
        {
            multi_reply_cpu[i].emplace_back(*context_cpu);
            multi_reply_cpu[i][a].ring_size_ = multi_reply[i][a].ring_size_;
            multi_reply_cpu[i][a].coeff_modulus_count_ = multi_reply[i][a].coeff_modulus_count_;
            multi_reply_cpu[i][a].cipher_size_ = multi_reply[i][a].cipher_size_;
            multi_reply_cpu[i][a].depth_ = multi_reply[i][a].depth_;
            multi_reply_cpu[i][a].scheme_ = heoncpu::scheme_type::bfv;
            multi_reply_cpu[i][a].in_ntt_domain_ = multi_reply[i][a].in_ntt_domain_; 
 
            multi_reply_cpu[i][a].scale_ = multi_reply[i][a].scale_;
            multi_reply_cpu[i][a].rescale_required_ = multi_reply[i][a].rescale_required_;
            multi_reply_cpu[i][a].relinearization_required_ = multi_reply[i][a].relinearization_required_;

            if (multi_reply[i][a].storage_type_ == heongpu::storage_type::DEVICE)
            {
                std::cout << "reply in device" << std::endl;
                // std::vector<Data64> gk_gpu(gk_size);
                // cudaMemcpy(gk_gpu.data(), galois_keys.device_location_.at(first_ele).data(), gk_size * sizeof(Data64), cudaMemcpyDeviceToHost);
                multi_reply_cpu[i][a].host_locations_.resize(multi_reply[i][a].device_locations_.size());
                cudaMemcpyAsync(
                    multi_reply_cpu[i][a].host_locations_.data(), multi_reply[i][a].device_locations_.data(),
                    multi_reply[i][a].device_locations_.size() * sizeof(Data64),
                    cudaMemcpyDeviceToHost );  
                // std::cout << "devive len:"<< multi_reply[i][a].device_locations_.size() << std::endl; 
                // std::cout << "host len:"<< multi_reply_cpu[i][a].host_locations_.size() << std::endl; 
                // assert(multi_reply_cpu[i][a].host_locations_.size() >0);
                // assert(multi_reply_cpu[i][a].host_locations_.size() == multi_reply_cpu[i][a].cipher_size_);
            }
            else
            {
                std::cout << "reply in host" << std::endl;
                // std::memcpy(host_locations_.data(),
                //             copy.host_locations_.data(),
                //             copy.host_locations_.size() * sizeof(Data64));
            }
        }
        //copy client
        client_cpu.context_->prime_vector.clear();
        for(int i=0;i<client.context_->prime_vector.size();i++){
            client_cpu.context_->prime_vector.emplace_back(client.context_->prime_vector[i]);
        }
        client_cpu.context_->plain_modulus_.value=client.context_->plain_modulus_.value;
        client_cpu.context_->plain_modulus_.bit=client.context_->plain_modulus_.bit;
        client_cpu.context_->plain_modulus_.mu=client.context_->plain_modulus_.mu;
        client_cpu.context_->n = client.context_->n;
        // client_cpu.pir_params_ = client.pir_params_;
        client_cpu.pir_params_.enable_symmetric = client.pir_params_.enable_symmetric;
        client_cpu.pir_params_.enable_batching = client.pir_params_.enable_batching;
        client_cpu.pir_params_.enable_mswitching = client.pir_params_.enable_mswitching;
        client_cpu.pir_params_.ele_num = client.pir_params_.ele_num;
        client_cpu.pir_params_.ele_size = client.pir_params_.ele_size;
        client_cpu.pir_params_.elements_per_plaintext = client.pir_params_.elements_per_plaintext;
        client_cpu.pir_params_.num_of_plaintexts = client.pir_params_.num_of_plaintexts;
        client_cpu.pir_params_.d = client.pir_params_.d;
        client_cpu.pir_params_.expansion_ratio = client.pir_params_.expansion_ratio;
        client_cpu.pir_params_.nvec = client.pir_params_.nvec;  
        client_cpu.pir_params_.slot_count = client.pir_params_.slot_count;
        // client_cpu.decryptor_ = client.decryptor_; 
        client_cpu.decryptor_->seed_=client.decryptor_->seed_;
        client_cpu.decryptor_->offset_=client.decryptor_->offset_; 
        client_cpu.decryptor_->secret_key_.resize(client.decryptor_->secret_key_.size());
        cudaMemcpyAsync(
            client_cpu.decryptor_->secret_key_.data(), client.decryptor_->secret_key_.data(),
            client.decryptor_->secret_key_.size() * sizeof(Data64),
            cudaMemcpyDeviceToHost );  
        client_cpu.decryptor_->n=client.decryptor_->n;
        client_cpu.decryptor_->n_power=client.decryptor_->n_power;
        client_cpu.decryptor_->Q_size_=client.decryptor_->Q_size_; 
        client_cpu.decryptor_->modulus_->resize(client.decryptor_->modulus_->size());
        cudaMemcpyAsync(
            client_cpu.decryptor_->modulus_->data(), client.decryptor_->modulus_->data(),
            client.decryptor_->modulus_->size() * sizeof(Data64),
            cudaMemcpyDeviceToHost );   
        client_cpu.decryptor_->ntt_table_->resize(client.decryptor_->ntt_table_->size());
        cudaMemcpyAsync(
            client_cpu.decryptor_->ntt_table_->data(), client.decryptor_->ntt_table_->data(),
            client.decryptor_->ntt_table_->size() * sizeof(Data64),
            cudaMemcpyDeviceToHost );   
        client_cpu.decryptor_->intt_table_->resize(client.decryptor_->intt_table_->size());
        cudaMemcpyAsync(
            client_cpu.decryptor_->intt_table_->data(), client.decryptor_->intt_table_->data(),
            client.decryptor_->intt_table_->size() * sizeof(Data64),
            cudaMemcpyDeviceToHost );   
        client_cpu.decryptor_->n_inverse_->resize(client.decryptor_->n_inverse_->size());
        cudaMemcpyAsync(
            client_cpu.decryptor_->n_inverse_->data(), client.decryptor_->n_inverse_->data(),
            client.decryptor_->n_inverse_->size() * sizeof(Data64),
            cudaMemcpyDeviceToHost );  
        client_cpu.decryptor_->plain_modulus_.value=client.decryptor_->plain_modulus_.value;
        client_cpu.decryptor_->plain_modulus_.bit=client.decryptor_->plain_modulus_.bit;
        client_cpu.decryptor_->plain_modulus_.mu=client.decryptor_->plain_modulus_.mu;
        client_cpu.decryptor_->gamma_.value=client.decryptor_->gamma_.value;
        client_cpu.decryptor_->gamma_.bit=client.decryptor_->gamma_.bit;
        client_cpu.decryptor_->gamma_.mu=client.decryptor_->gamma_.mu; 
        client_cpu.decryptor_->Qi_t_->resize(client.decryptor_->Qi_t_->size());
        cudaMemcpyAsync(
            client_cpu.decryptor_->Qi_t_->data(), client.decryptor_->Qi_t_->data(),
            client.decryptor_->Qi_t_->size() * sizeof(Data64),
            cudaMemcpyDeviceToHost );   
        client_cpu.decryptor_->Qi_gamma_->resize(client.decryptor_->Qi_gamma_->size());
        cudaMemcpyAsync(
            client_cpu.decryptor_->Qi_gamma_->data(), client.decryptor_->Qi_gamma_->data(),
            client.decryptor_->Qi_gamma_->size() * sizeof(Data64),
            cudaMemcpyDeviceToHost );   
        client_cpu.decryptor_->Qi_inverse_->resize(client.decryptor_->Qi_inverse_->size());
        cudaMemcpyAsync(
            client_cpu.decryptor_->Qi_inverse_->data(), client.decryptor_->Qi_inverse_->data(),
            client.decryptor_->Qi_inverse_->size() * sizeof(Data64),
            cudaMemcpyDeviceToHost );  
        client_cpu.decryptor_->mulq_inv_t_=client.decryptor_->mulq_inv_t_;
        client_cpu.decryptor_->mulq_inv_gamma_=client.decryptor_->mulq_inv_gamma_;
        client_cpu.decryptor_->inv_gamma_=client.decryptor_->inv_gamma_; 
        client_cpu.decryptor_->Mi_->resize(client.decryptor_->Mi_->size());
        cudaMemcpyAsync(
            client_cpu.decryptor_->Mi_->data(), client.decryptor_->Mi_->data(),
            client.decryptor_->Mi_->size() * sizeof(Data64),
            cudaMemcpyDeviceToHost );   
        client_cpu.decryptor_->Mi_inv_->resize(client.decryptor_->Mi_inv_->size());
        cudaMemcpyAsync(
            client_cpu.decryptor_->Mi_inv_->data(), client.decryptor_->Mi_inv_->data(),
            client.decryptor_->Mi_inv_->size() * sizeof(Data64),
            cudaMemcpyDeviceToHost );   
        client_cpu.decryptor_->upper_half_threshold_->resize(client.decryptor_->upper_half_threshold_->size());
        cudaMemcpyAsync(
            client_cpu.decryptor_->upper_half_threshold_->data(), client.decryptor_->upper_half_threshold_->data(),
            client.decryptor_->upper_half_threshold_->size() * sizeof(Data64),
            cudaMemcpyDeviceToHost );   
        client_cpu.decryptor_->decryption_modulus_->resize(client.decryptor_->decryption_modulus_->size());
        cudaMemcpyAsync(
            client_cpu.decryptor_->decryption_modulus_->data(), client.decryptor_->decryption_modulus_->data(),
            client.decryptor_->decryption_modulus_->size() * sizeof(Data64),
            cudaMemcpyDeviceToHost );  
        client_cpu.decryptor_->total_bit_count_=client.decryptor_->total_bit_count_;
        
        int n;
        int n_power;
        int slot_count_;

        // BFV
        std::shared_ptr<DeviceVector<Modulus64>> plain_modulus_;
        std::shared_ptr<DeviceVector<Ninverse64>> n_plain_inverse_;
        std::shared_ptr<DeviceVector<Root64>> plain_ntt_tables_;
        std::shared_ptr<DeviceVector<Root64>> plain_intt_tables_;
        std::shared_ptr<DeviceVector<Data64>> encoding_location_;

  

        int Q_size_;
        int total_coeff_bit_count_;
        std::shared_ptr<DeviceVector<Modulus64>> modulus_;
        std::shared_ptr<DeviceVector<Root64>> ntt_table_;
        std::shared_ptr<DeviceVector<Root64>> intt_table_;
        std::shared_ptr<DeviceVector<Ninverse64>> n_inverse_;

        std::shared_ptr<DeviceVector<Data64>> Mi_;
        std::shared_ptr<DeviceVector<Data64>> Mi_inv_;
        std::shared_ptr<DeviceVector<Data64>> upper_half_threshold_;
        std::shared_ptr<DeviceVector<Data64>> decryption_modulus_;

    }

    for (int j = 0; j < query_count; j++)
    {
        std::cout << "[" << (j + 1) << "/" << query_count << "]: reply decoding"
                  << std::endl;
        std::vector<uint8_t> elems =
            client.decode_reply(multi_reply[j], offset[j]);
        std::vector<uint8_t> elems_cpu =
            client_cpu.decode_reply(multi_reply_cpu[j], offset[j]);

        assert(elems.size() == size_per_item);
        assert(elems_cpu.size() == size_per_item);

        bool failed = false;
        // Check that we retrieved the correct element
        for (uint32_t i = 0; i < size_per_item; i++)
        {
            if (elems[i] != db_copy.get()[(ele_index[j] * size_per_item) + i])
            {
                std::cout
                    << "[" << (j + 1) << "/" << query_count << "]: elems "
                    << (int) elems[i] << ", db "
                    << (int) db_copy.get()[(ele_index[j] * size_per_item) + i]
                    << std::endl;
                std::cout << "[" << (j + 1) << "/" << query_count
                          << "]: PIR result wrong at " << i << std::endl;
                failed = true;
            }
        }
        //cpu test
        for (uint32_t i = 0; i < size_per_item; i++)
        {
            if (elems_cpu[i] != db_copy.get()[(ele_index[j] * size_per_item) + i])
            {
                std::cout
                    << "[" << (j + 1) << "/" << query_count << "]: elems_cpu "
                    << (int) elems_cpu[i] << ", db "
                    << (int) db_copy.get()[(ele_index[j] * size_per_item) + i]
                    << std::endl;
                std::cout << "[" << (j + 1) << "/" << query_count
                          << "]: PIR_cpu result wrong at " << i << std::endl;
                failed = true;
            }
        }

        if (failed)
        {
            throw std::runtime_error("Failed!");
        }
    }

    std::cout << "Main: PIR result correct!" << std::endl;

    return EXIT_SUCCESS;
}