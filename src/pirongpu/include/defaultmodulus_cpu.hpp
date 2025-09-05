#pragma once

// #include "common.cuh"
#include "nttparameters_cpu.hpp"
#include <unordered_map>
#include <vector>

namespace heoncpu
{
    /*
    The default modulus for different poly_modulus_degree values was determined
    based on security estimates from the lattice-estimator tool, with parameters
    selected to align with the desired security level.
    lattice-estimator: https://github.com/malb/lattice-estimator
    Reference: https://eprint.iacr.org/2015/046
    */
    namespace defaultparams
    {

        const std::unordered_map<std::size_t, std::vector<Modulus64>>&
        get_128bit_sec_modulus();

        const std::unordered_map<std::size_t, std::vector<Modulus64>>&
        get_192bit_sec_modulus();

        const std::unordered_map<std::size_t, std::vector<Modulus64>>&
        get_256bit_sec_modulus();

    } // namespace defaultparams
} // namespace heongpu 