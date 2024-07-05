#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>

#define THREADS_PER_BLOCK 1024

__global__ void initCurandStates(curandState *states, unsigned long seed)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &states[idx]);
}

__device__ __inline__ void artificialDelay(curandState *state, int minCycles, int maxCycles)
{
    int cycles = minCycles + (curand(state) % (maxCycles - minCycles));
    clock_t start_clock = clock();
    clock_t clock_offset = cycles * 1000;
    while (clock() - start_clock < clock_offset)
        ;
}

__device__ int getSMID()
{
    int smid;
    asm("mov.u32 %0, %smid;" : "=r"(smid));
    return smid;
}

template <typename T>
__global__ void dataRaceKernelSameWarp(T *data, T *buffer, curandState *randStates, int thread1, int thread2)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // int smid = getSMID();
    // if (gridDim.x > 1 and (blockIdx.x == 0 or blockIdx.x == gridDim.x - 1)) {
    //     printf("Block [%d] is on SM %d\n", blockIdx.x, smid);
    // }

    if (idx == thread1)
    {
        // artificialDelay(&randStates[idx], 100, 200);
        data[0] = idx;
    }
    else if (idx == thread2)
    {
        // artificialDelay(&randStates[idx], 100, 200);
        buffer[0] = data[0];
    }
}

template <typename T>
void experiment1()
{
    T *d_data, *d_buffer;
    T h_data;
    curandState *d_randStates;

    cudaMalloc((void **)&d_data, sizeof(T));
    cudaMalloc((void **)&d_buffer, sizeof(T));
    cudaMalloc((void **)&d_randStates, THREADS_PER_BLOCK * sizeof(curandState));

    h_data = static_cast<T>(-1);
    cudaMemcpy(d_data, &h_data, sizeof(T), cudaMemcpyHostToDevice);

    initCurandStates<<<1, THREADS_PER_BLOCK>>>(d_randStates, time(0));
    cudaDeviceSynchronize();

    printf("Experiment 1 (Same Warp): ");
    dataRaceKernelSameWarp<<<1, THREADS_PER_BLOCK>>>(d_data, d_buffer, d_randStates, 2, 3);
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
    cudaFree(d_randStates);
    cudaFree(d_buffer);
}

template <typename T>
void experiment2()
{
    T *d_data, *d_buffer;
    T h_data;
    curandState *d_randStates;

    cudaMalloc((void **)&d_data, sizeof(T));
    cudaMalloc((void **)&d_buffer, sizeof(T));
    cudaMalloc((void **)&d_randStates, THREADS_PER_BLOCK * sizeof(curandState));

    h_data = static_cast<T>(-1);
    cudaMemcpy(d_data, &h_data, sizeof(T), cudaMemcpyHostToDevice);

    initCurandStates<<<1, THREADS_PER_BLOCK>>>(d_randStates, time(0));
    cudaDeviceSynchronize();

    printf("Experiment 2 (Different Warp Same Block): ");
    dataRaceKernelSameWarp<<<1, THREADS_PER_BLOCK>>>(d_data, d_buffer, d_randStates, 2, 128);
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
    cudaFree(d_randStates);
}

template <typename T>
void experiment3()
{
    T *d_data, *d_buffer;
    T h_data;
    curandState *d_randStates;

    cudaMalloc((void **)&d_data, sizeof(T));
    cudaMalloc((void **)&d_buffer, sizeof(T));
    cudaMalloc((void **)&d_randStates, THREADS_PER_BLOCK * sizeof(curandState));

    h_data = static_cast<T>(-1);
    cudaMemcpy(d_data, &h_data, sizeof(T), cudaMemcpyHostToDevice);

    initCurandStates<<<21, THREADS_PER_BLOCK>>>(d_randStates, time(0));
    cudaDeviceSynchronize();

    printf("Experiment 3 (Different Block Same SM): ");
    dataRaceKernelSameWarp<<<21, THREADS_PER_BLOCK>>>(d_data, d_buffer, d_randStates, 2, 20488);
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
    cudaFree(d_randStates);
}

template <typename T>
void experiment4()
{
    T *d_data, *d_buffer;
    T h_data;
    curandState *d_randStates;

    cudaMalloc((void **)&d_data, sizeof(T));
    cudaMalloc((void **)&d_buffer, sizeof(T));
    cudaMalloc((void **)&d_randStates, THREADS_PER_BLOCK * sizeof(curandState));

    h_data = static_cast<T>(-1);
    cudaMemcpy(d_data, &h_data, sizeof(T), cudaMemcpyHostToDevice);

    initCurandStates<<<20, THREADS_PER_BLOCK>>>(d_randStates, time(0));
    cudaDeviceSynchronize();

    printf("Experiment 4 (Different SM): ");
    int numBlocks = 16;

    dataRaceKernelSameWarp<<<numBlocks, THREADS_PER_BLOCK>>>(d_data, d_buffer, d_randStates, 2, 1030);
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
    cudaFree(d_randStates);
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
        experiment3<float>();

        experiment3<double>();

        experiment3<int8_t>();

        experiment3<int16_t>();

        experiment3<int32_t>();

        experiment3<int64_t>();

        experiment3<u_int8_t>();

        experiment3<u_int16_t>();

        experiment3<u_int32_t>();

        experiment3<u_int64_t>();

        experiment3<__half>();
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
