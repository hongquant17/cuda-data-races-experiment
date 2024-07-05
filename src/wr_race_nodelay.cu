#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>

#define THREADS_PER_BLOCK 64

__device__ int getSMID()
{
    int smid;
    asm("mov.u32 %0, %smid;" : "=r"(smid));
    return smid;
}

template <typename T>
__global__ void dataRaceKernelSameWarp(T *data, T *buffer, int thread1, int thread2)
{   
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int smid = getSMID();
    // if (gridDim.x > 1 and (blockIdx.x == 0 or blockIdx.x == gridDim.x - 1)) {
    //     printf("Block [%d] is on SM %d\n", blockIdx.x, smid);
    // }

    if (idx == thread1)
    {
        data[0] = idx;
    }
    else if (idx == thread2)
    {
        buffer[0] = data[0];
    }
}

template <typename T>
void experiment1()
{
    T *d_data, *d_buffer;
    T h_data;

    cudaMalloc((void **)&d_data, sizeof(T));
    cudaMalloc((void **)&d_buffer, sizeof(T));

    h_data = static_cast<T>(-1);
    cudaMemcpy(d_data, &h_data, sizeof(T), cudaMemcpyHostToDevice);

    printf("Experiment 1 (Same Warp): ");
    dataRaceKernelSameWarp<<<1, THREADS_PER_BLOCK>>>(d_data, d_buffer, 2, 3);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_data, d_buffer, sizeof(T), cudaMemcpyDeviceToHost);

    if constexpr (std::is_same<T, float>::value)
    {
        printf("float %f\n", h_data);
    }
    else if constexpr (std::is_same<T, double>::value)
    {
        printf("double %f\n", h_data);
    }
    else if constexpr (std::is_same<T, int8_t>::value)
    {
        printf("int8 %d\n", h_data);
    }
    else if constexpr (std::is_same<T, int16_t>::value)
    {
        printf("int16 %d\n", h_data);
    }
    else if constexpr (std::is_same<T, int32_t>::value)
    {
        printf("int32 %d\n", h_data);
    }
    else if constexpr (std::is_same<T, int64_t>::value)
    {
        printf("int64 %ld\n", h_data);
    }
    else if constexpr (std::is_same<T, u_int8_t>::value)
    {
        printf("u_int8 %d\n", h_data);
    }
    else if constexpr (std::is_same<T, u_int16_t>::value)
    {
        printf("u_int16 %d\n", h_data);
    }
    else if constexpr (std::is_same<T, u_int32_t>::value)
    {
        printf("u_int32 %d\n", h_data);
    }
    else if constexpr (std::is_same<T, u_int64_t>::value)
    {
        printf("u_int64 %ld\n", h_data);
    }
    else if constexpr (std::is_same<T, __half>::value)
    {
        printf("__half %f\n\n", __half2float(h_data));
    }
    else
    {
        printf("Unknown type\n");
    }

    cudaFree(d_data);
    cudaFree(d_buffer);
}

template <typename T>
void experiment2()
{
    T *d_data, *d_buffer;
    T h_data;

    cudaMalloc((void **)&d_data, sizeof(T));
    cudaMalloc((void **)&d_buffer, sizeof(T));

    h_data = static_cast<T>(-1);
    cudaMemcpy(d_data, &h_data, sizeof(T), cudaMemcpyHostToDevice);

    printf("Experiment 2 (Different Warp Same Block): ");
    dataRaceKernelSameWarp<<<1, THREADS_PER_BLOCK>>>(d_data, d_buffer, 2, 40);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_data, d_buffer, sizeof(T), cudaMemcpyDeviceToHost);
    if constexpr (std::is_same<T, float>::value)
    {
        printf("float %f\n", h_data);
    }
    else if constexpr (std::is_same<T, double>::value)
    {
        printf("double %f\n", h_data);
    }
    else if constexpr (std::is_same<T, int8_t>::value)
    {
        printf("int8 %d\n", h_data);
    }
    else if constexpr (std::is_same<T, int16_t>::value)
    {
        printf("int16 %d\n", h_data);
    }
    else if constexpr (std::is_same<T, int32_t>::value)
    {
        printf("int32 %d\n", h_data);
    }
    else if constexpr (std::is_same<T, int64_t>::value)
    {
        printf("int64 %ld\n", h_data);
    }
    else if constexpr (std::is_same<T, u_int8_t>::value)
    {
        printf("u_int8 %d\n", h_data);
    }
    else if constexpr (std::is_same<T, u_int16_t>::value)
    {
        printf("u_int16 %d\n", h_data);
    }
    else if constexpr (std::is_same<T, u_int32_t>::value)
    {
        printf("u_int32 %d\n", h_data);
    }
    else if constexpr (std::is_same<T, u_int64_t>::value)
    {
        printf("u_int64 %ld\n", h_data);
    }
    else if constexpr (std::is_same<T, __half>::value)
    {
        printf("__half %f\n\n", __half2float(h_data));
    }
    else
    {
        printf("Unknown type\n");
    }

    cudaFree(d_data);
    cudaFree(d_buffer);
}

template <typename T>
void experiment3()
{
    T *d_data, *d_buffer;
    T h_data;

    cudaMalloc((void **)&d_data, sizeof(T));
    cudaMalloc((void **)&d_buffer, sizeof(T));

    h_data = static_cast<T>(-1);
    cudaMemcpy(d_data, &h_data, sizeof(T), cudaMemcpyHostToDevice);

    printf("Experiment 3 (Different Block Same SM): ");
    dataRaceKernelSameWarp<<<21, THREADS_PER_BLOCK>>>(d_data, d_buffer, 2, 1288);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_data, d_buffer, sizeof(T), cudaMemcpyDeviceToHost);
    if constexpr (std::is_same<T, float>::value)
    {
        printf("float %f\n", h_data);
    }
    else if constexpr (std::is_same<T, double>::value)
    {
        printf("double %f\n", h_data);
    }
    else if constexpr (std::is_same<T, int8_t>::value)
    {
        printf("int8 %d\n", h_data);
    }
    else if constexpr (std::is_same<T, int16_t>::value)
    {
        printf("int16 %d\n", h_data);
    }
    else if constexpr (std::is_same<T, int32_t>::value)
    {
        printf("int32 %d\n", h_data);
    }
    else if constexpr (std::is_same<T, int64_t>::value)
    {
        printf("int64 %ld\n", h_data);
    }
    else if constexpr (std::is_same<T, u_int8_t>::value)
    {
        printf("u_int8 %d\n", h_data);
    }
    else if constexpr (std::is_same<T, u_int16_t>::value)
    {
        printf("u_int16 %d\n", h_data);
    }
    else if constexpr (std::is_same<T, u_int32_t>::value)
    {
        printf("u_int32 %d\n", h_data);
    }
    else if constexpr (std::is_same<T, u_int64_t>::value)
    {
        printf("u_int64 %ld\n", h_data);
    }
    else if constexpr (std::is_same<T, __half>::value)
    {
        printf("__half %f\n\n", __half2float(h_data));
    }
    else
    {
        printf("Unknown type\n");
    }

    cudaFree(d_data);
    cudaFree(d_buffer);
}

template <typename T>
void experiment4()
{
    T *d_data, *d_buffer;
    T h_data;

    cudaMalloc((void **)&d_data, sizeof(T));
    cudaMalloc((void **)&d_buffer, sizeof(T));

    h_data = static_cast<T>(-1);
    cudaMemcpy(d_data, &h_data, sizeof(T), cudaMemcpyHostToDevice);

    printf("Experiment 4 (Different SM): ");
    int numBlocks = 16;

    dataRaceKernelSameWarp<<<numBlocks, THREADS_PER_BLOCK>>>(d_data, d_buffer, 2, 65);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_data, d_buffer, sizeof(T), cudaMemcpyDeviceToHost);

    if constexpr (std::is_same<T, float>::value)
    {
        printf("float %f\n", h_data);
    }
    else if constexpr (std::is_same<T, double>::value)
    {
        printf("double %f\n", h_data);
    }
    else if constexpr (std::is_same<T, int8_t>::value)
    {
        printf("int8 %d\n", h_data);
    }
    else if constexpr (std::is_same<T, int16_t>::value)
    {
        printf("int16 %d\n", h_data);
    }
    else if constexpr (std::is_same<T, int32_t>::value)
    {
        printf("int32 %d\n", h_data);
    }
    else if constexpr (std::is_same<T, int64_t>::value)
    {
        printf("int64 %ld\n", h_data);
    }
    else if constexpr (std::is_same<T, u_int8_t>::value)
    {
        printf("u_int8 %d\n", h_data);
    }
    else if constexpr (std::is_same<T, u_int16_t>::value)
    {
        printf("u_int16 %d\n", h_data);
    }
    else if constexpr (std::is_same<T, u_int32_t>::value)
    {
        printf("u_int32 %d\n", h_data);
    }
    else if constexpr (std::is_same<T, u_int64_t>::value)
    {
        printf("u_int64 %ld\n", h_data);
    }
    else if constexpr (std::is_same<T, __half>::value)
    {
        printf("__half %f\n\n", __half2float(h_data));
    }
    else
    {
        printf("Unknown type\n");
    }

    cudaFree(d_data);
    cudaFree(d_buffer);
}

int main(int argc, char **argv)
{
    printf("Running experiments to observe data races:\n\n");

    {
        experiment1<float>();

        experiment1<double>();

        experiment1<int8_t>();

        experiment1<int16_t>();

        experiment1<int32_t>();

        experiment1<int64_t>();

        experiment1<u_int8_t>();

        experiment1<u_int16_t>();

        experiment1<u_int32_t>();

        experiment1<u_int64_t>();

        experiment1<__half>();
    }

    {
        experiment2<float>();

        experiment2<double>();

        experiment2<int8_t>();

        experiment2<int16_t>();

        experiment2<int32_t>();

        experiment2<int64_t>();

        experiment2<u_int8_t>();

        experiment2<u_int16_t>();

        experiment2<u_int32_t>();

        experiment2<u_int64_t>();

        experiment2<__half>();
    }

    {
        // experiment3<float>();

        // experiment3<double>();

        // experiment3<int8_t>();

        // experiment3<int16_t>();

        // experiment3<int32_t>();

        // experiment3<int64_t>();

        // experiment3<u_int8_t>();

        // experiment3<u_int16_t>();

        // experiment3<u_int32_t>();

        // experiment3<u_int64_t>();

        // experiment3<__half>();
    }

    {
        experiment4<float>();

        experiment4<double>();

        experiment4<int8_t>();

        experiment4<int16_t>();

        experiment4<int32_t>();

        experiment4<int64_t>();

        experiment4<u_int8_t>();

        experiment4<u_int16_t>();

        experiment4<u_int32_t>();

        experiment4<u_int64_t>();

        experiment4<__half>();
    }

    return 0;
}
