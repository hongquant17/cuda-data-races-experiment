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

template <typename T>
void printVal(T h_data) {
    if constexpr (std::is_same<T, float>::value)
    {
        printf("float %f\n", h_data);
    }
    else if constexpr(std::is_same<T, double>::value) {
        printf("double %f\n", h_data);
    }
    else if constexpr(std::is_same<T, int8_t>::value) {
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
    else if constexpr(std::is_same<T, u_int8_t>::value) {
        printf("u_int8 %d\n", h_data);
    }
    else if constexpr(std::is_same<T, u_int16_t>::value) {
        printf("u_int16 %d\n", h_data);
    }
    else if constexpr(std::is_same<T, u_int32_t>::value) {
        printf("u_int32 %d\n", h_data);
    }
    else if constexpr(std::is_same<T, u_int64_t>::value) {
        printf("u_int64 %ld\n", h_data);
    }
    else if constexpr(std::is_same<T, __half>::value) {
        printf("__half %f\n\n", __half2float(h_data));
    }
}

template <typename T>
__global__ void dataRaceKernelSameWarp(T *data, curandState *randStates)
{
    int idx = threadIdx.x;
    if (idx < 3 && idx > 0)
    {
        // artificialDelay(&randStates[idx], 100, 200);
        data[0] = idx;
    }
}

template <typename T>
void experiment1()
{
    T *d_data;
    T h_data;

    cudaMalloc((void **)&d_data, sizeof(T));
    h_data = static_cast<T>(-1);
    cudaMemcpy(d_data, &h_data, sizeof(T), cudaMemcpyHostToDevice);

    curandState *d_randStates;
    cudaMalloc((void **)&d_randStates, THREADS_PER_BLOCK * sizeof(curandState));
    // initCurandStates<<<1, THREADS_PER_BLOCK>>>(d_randStates, time(0));
    // cudaDeviceSynchronize();

    dataRaceKernelSameWarp<<<1, THREADS_PER_BLOCK>>>(d_data, d_randStates);
    cudaMemcpy(&h_data, d_data, sizeof(T), cudaMemcpyDeviceToHost);


    printf("Experiment 1 (Same Warp): ");
    printVal(h_data);

    cudaFree(d_data);
    cudaFree(d_randStates);
}

template <typename T>
__global__ void dataRaceKernelDifferentWarpSameBlock(T *data, curandState *randStates)
{
    int idx = threadIdx.x;

    if (idx == 2 || idx == 60)
    {
        // artificialDelay(&randStates[idx], 100, 200);
        data[0] = idx;
    }
}

template <typename T>
void experiment2()
{
    T *d_data;
    T h_data;

    cudaMalloc((void **)&d_data, sizeof(T));
    h_data = static_cast<T>(-1);
    cudaMemcpy(d_data, &h_data, sizeof(T), cudaMemcpyHostToDevice);

    curandState *d_randStates;
    cudaMalloc((void **)&d_randStates, THREADS_PER_BLOCK * sizeof(curandState));
    // initCurandStates<<<1, THREADS_PER_BLOCK>>>(d_randStates, time(0));
    // cudaDeviceSynchronize();

    dataRaceKernelDifferentWarpSameBlock<<<1, THREADS_PER_BLOCK>>>(d_data, d_randStates);
    cudaMemcpy(&h_data, d_data, sizeof(T), cudaMemcpyDeviceToHost);
    printf("Experiment 2 (Different Warp Same Block): ");
    printVal(h_data);

    cudaFree(d_data);
    cudaFree(d_randStates);
}

template <typename T>
__global__ void dataRaceKernelDifferentBlocksSameSM(T *data, curandState *randStates)
{
    int idx = threadIdx.x;

    if (idx == 0)
    {
        artificialDelay(&randStates[idx], 100, 200);
        data[0] = blockIdx.x;
    }
}

template <typename T>
void experiment3()
{
    T *d_data;
    T h_data;

    cudaMalloc((void **)&d_data, sizeof(T));
    h_data = static_cast<T>(-1);
    cudaMemcpy(d_data, &h_data, sizeof(T), cudaMemcpyHostToDevice);

    curandState *d_randStates;
    cudaMalloc((void **)&d_randStates, THREADS_PER_BLOCK * sizeof(curandState));
    initCurandStates<<<2, THREADS_PER_BLOCK>>>(d_randStates, time(0));
    cudaDeviceSynchronize();

    dataRaceKernelDifferentBlocksSameSM<<<2, THREADS_PER_BLOCK>>>(d_data, d_randStates);
    cudaMemcpy(&h_data, d_data, sizeof(T), cudaMemcpyDeviceToHost);
    printf("Experiment 3 (Different Blocks Same SM): ");
    printVal(h_data);

    cudaFree(d_data);
    cudaFree(d_randStates);
}

__device__ int getSMID()
{
    int smid;
    asm("mov.u32 %0, %smid;" : "=r"(smid));
    return smid;
}

template <typename T>
__global__ void dataRaceKernelDifferentSMs(T *data, curandState *randStates)
{
    int idx = threadIdx.x;
    // int smid = getSMID();
    if (idx == 0 && (blockIdx.x == 0 || blockIdx.x == (gridDim.x - 1)))
    {
        // printf("Block [%d] is on SM %d\n", blockIdx.x, smid);
        // artificialDelay(&randStates[idx], 100, 200);
        data[0] = blockIdx.x;
    }
}

template <typename T>
void experiment4()
{
    T *d_data;
    T h_data;

    cudaMalloc((void **)&d_data, sizeof(T));
    h_data = static_cast<T>(-1);
    cudaMemcpy(d_data, &h_data, sizeof(T), cudaMemcpyHostToDevice);

    int numBlocks = 12;

    curandState *d_randStates;
    cudaMalloc((void **)&d_randStates, THREADS_PER_BLOCK * sizeof(curandState));
    // initCurandStates<<<numBlocks, THREADS_PER_BLOCK>>>(d_randStates, time(0));
    // cudaDeviceSynchronize();


    dataRaceKernelDifferentSMs<<<numBlocks, THREADS_PER_BLOCK>>>(d_data, d_randStates);
    cudaMemcpy(&h_data, d_data, sizeof(T), cudaMemcpyDeviceToHost);
    printf("Experiment 4 (Different SMs): ");
    printVal(h_data);

    cudaFree(d_data);
    cudaFree(d_randStates);
}

int main(int argc, char **argv)
{
    printf("Running experiments to observe data races:\n\n");


    {
        experiment1<float>();
        cudaDeviceSynchronize();

        experiment1<double>();
        cudaDeviceSynchronize();

        experiment1<int8_t>();
        cudaDeviceSynchronize();

        experiment1<int16_t>();
        cudaDeviceSynchronize();

        experiment1<int32_t>();
        cudaDeviceSynchronize();

        experiment1<int64_t>();
        cudaDeviceSynchronize();

        experiment1<u_int8_t>();
        cudaDeviceSynchronize();

        experiment1<u_int16_t>();
        cudaDeviceSynchronize();

        experiment1<u_int32_t>();
        cudaDeviceSynchronize();

        experiment1<u_int64_t>();
        cudaDeviceSynchronize();

        experiment1<__half>();
        cudaDeviceSynchronize();
    }

    {
        experiment2<float>();
        cudaDeviceSynchronize();

        experiment2<double>();
        cudaDeviceSynchronize();

        experiment2<int8_t>();
        cudaDeviceSynchronize();

        experiment2<int16_t>();
        cudaDeviceSynchronize();

        experiment2<int32_t>();
        cudaDeviceSynchronize();

        experiment2<int64_t>();
        cudaDeviceSynchronize();

        experiment2<u_int8_t>();
        cudaDeviceSynchronize();

        experiment2<u_int16_t>();
        cudaDeviceSynchronize();

        experiment2<u_int32_t>();
        cudaDeviceSynchronize();

        experiment2<u_int64_t>();
        cudaDeviceSynchronize();

        experiment2<__half>();
        cudaDeviceSynchronize();
    }

    {
        // experiment3<float>();
        // cudaDeviceSynchronize();

        // experiment3<double>();
        // cudaDeviceSynchronize();

        // experiment3<int8_t>();
        // cudaDeviceSynchronize();

        // experiment3<int16_t>();
        // cudaDeviceSynchronize();

        // experiment3<int32_t>();
        // cudaDeviceSynchronize();

        // experiment3<int64_t>();
        // cudaDeviceSynchronize();

        // experiment3<u_int8_t>();
        // cudaDeviceSynchronize();

        // experiment3<u_int16_t>();
        // cudaDeviceSynchronize();

        // experiment3<u_int32_t>();
        // cudaDeviceSynchronize();

        // experiment3<u_int64_t>();
        // cudaDeviceSynchronize();

        // experiment3<__half>();
        // cudaDeviceSynchronize();
    }

    {
        experiment4<float>();
        cudaDeviceSynchronize();

        experiment4<double>();
        cudaDeviceSynchronize();

        experiment4<int8_t>();
        cudaDeviceSynchronize();

        experiment4<int16_t>();
        cudaDeviceSynchronize();

        experiment4<int32_t>();
        cudaDeviceSynchronize();

        experiment4<int64_t>();
        cudaDeviceSynchronize();

        experiment4<u_int8_t>();
        cudaDeviceSynchronize();

        experiment4<u_int16_t>();
        cudaDeviceSynchronize();

        experiment4<u_int32_t>();
        cudaDeviceSynchronize();

        experiment4<u_int64_t>();
        cudaDeviceSynchronize();

        experiment4<__half>();
        cudaDeviceSynchronize();
    }

    return 0;
}
