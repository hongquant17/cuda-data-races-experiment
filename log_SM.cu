#include <iostream>
#include <cuda_runtime.h>

int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // Get properties of device 0
    std::cout << "Number of SMs: " << prop.multiProcessorCount << std::endl;
    return 0;
}