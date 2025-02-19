# PIRonGPU 
PIRonGPU is an innovative library that accelerates data retrieval while preserving user privacy through the implementation of Private Information Retrieval (PIR) technology. By modifying the existing [SealPIR](https://github.com/microsoft/SealPIR) protocol with [HEonGPU](https://github.com/Alisah-Ozcan/HEonGPU/tree/main), PIRonGPU leverages the parallel processing capabilities of GPUs to perform computation-intensive tasks at significantly higher speeds. This approach not only ensures that the client’s data request remains confidential but also enhances performance in high-demand applications. With its modular and flexible design, PIRonGPU offers a secure and efficient solution for modern data querying needs, making it an ideal tool for applications requiring both privacy and high-performance computing.

- Original Paper: [PIR with compressed queries and amortized query processing](https://eprint.iacr.org/2017/1142)
- HEonGPU version paper: [HEonGPU: a GPU-based Fully Homomorphic Encryption Library 1.0](https://eprint.iacr.org/2024/1543)

## Installation

### Requirements

- [CMake](https://cmake.org/download/) >=3.26.4
- [GCC](https://gcc.gnu.org/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) >=11.4
- [HEonGPU](https://github.com/Alisah-Ozcan/HEonGPU/tree/main) >=1.1.0

### Build & Install

To build and install PIRonGPU, follow the steps below. This includes configuring the project using CMake, compiling the source code, and installing the library on your system.

<div align="center">

| GPU Architecture | Compute Capability (CMAKE_CUDA_ARCHITECTURES Value) |
|:----------------:|:---------------------------------------------------:|
| Volta  | 70, 72 |
| Turing | 75 |
| Ampere | 80, 86 |
| Ada	 | 89, 90 |

</div>

```bash
$ cmake -S . -D CMAKE_CUDA_ARCHITECTURES=89 -B build
$ cmake --build ./build/
```

## Examples

To run examples:

```bash
$ cmake -S . -D PIRonGPU_BUILD_EXAMPLES=ON -D CMAKE_CUDA_ARCHITECTURES=89 -B build
$ cmake --build ./build/

$ ./build/bin/examples/<...>
$ Example: ./build/bin/examples/pir_example
```

## Tests

To run examples:

```bash
$ cmake -S . -D PIRonGPU_BUILD_TESTS=ON -D CMAKE_CUDA_ARCHITECTURES=89 -B build
$ cmake --build ./build/

$ ./build/bin/tests/<...>
```

## License
This project is licensed under the [Apache License](LICENSE). For more details, please refer to the License file.

## Contributing
Contributions are welcome! Please check the [CONTRIBUTING](CONTRIBUTING.md) file for guidelines on how to contribute to the project.

## Contact
If you have any questions or feedback, feel free to contact me: 
- Email: alisah@sabanciuniv.edu
- LinkedIn: [Profile](https://www.linkedin.com/in/ali%C5%9Fah-%C3%B6zcan-472382305/)