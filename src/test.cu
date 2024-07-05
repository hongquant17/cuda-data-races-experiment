#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>

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
    while (clock() - start_clock < clock_offset) ;
}

__global__ void dataRaceKernelSameWarp(int *data, curandState *randStates)
{
    int idx = threadIdx.x;
    if (idx < 2)
    {
        artificialDelay(&randStates[idx], 100, 200);
        data[0] = idx;
    }
}

void experiment1()
{
    int *d_data;
    int h_data;

    cudaMalloc((void **)&d_data, sizeof(int));
    h_data = -1;
    cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice);

    curandState *d_randStates;
    cudaMalloc((void **)&d_randStates, THREADS_PER_BLOCK * sizeof(curandState));
    initCurandStates<<<1, THREADS_PER_BLOCK>>>(d_randStates, time(0));
    cudaDeviceSynchronize();

    dataRaceKernelSameWarp<<<1, THREADS_PER_BLOCK>>>(d_data, d_randStates);
    cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Experiment 1 (Same Warp): %d\n", h_data);

    cudaFree(d_data);
    cudaFree(d_randStates);
}

__global__ void dataRaceKernelDifferentWarpSameBlock(int *data, curandState *randStates)
{
    int idx = threadIdx.x;

    if (idx == 1 || idx == 1000)
    {
        artificialDelay(&randStates[idx], 100, 200);
        data[0] = idx;
    }
}

void experiment2()
{
    int *d_data;
    int h_data;

    cudaMalloc((void **)&d_data, sizeof(int));
    h_data = -1;
    cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice);

    curandState *d_randStates;
    cudaMalloc((void **)&d_randStates, THREADS_PER_BLOCK * sizeof(curandState));
    initCurandStates<<<1, THREADS_PER_BLOCK>>>(d_randStates, time(0));
    cudaDeviceSynchronize();

    dataRaceKernelDifferentWarpSameBlock<<<1, THREADS_PER_BLOCK>>>(d_data, d_randStates);
    cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Experiment 2 (Different Warp Same Block): %d\n", h_data);

    cudaFree(d_data);
    cudaFree(d_randStates);
}

__global__ void dataRaceKernelDifferentBlocksSameSM(int *data, curandState *randStates)
{
    int idx = threadIdx.x;

    if (idx == 0)
    {
        artificialDelay(&randStates[idx], 100, 200);
        data[0] = blockIdx.x;
    }
}

void experiment3()
{
    int *d_data;
    int h_data;

    cudaMalloc((void **)&d_data, sizeof(int));
    h_data = -1;
    cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice);

    curandState *d_randStates;
    cudaMalloc((void **)&d_randStates, THREADS_PER_BLOCK * sizeof(curandState));
    initCurandStates<<<2, THREADS_PER_BLOCK>>>(d_randStates, time(0));
    cudaDeviceSynchronize();

    dataRaceKernelDifferentBlocksSameSM<<<2, THREADS_PER_BLOCK>>>(d_data, d_randStates);
    cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Experiment 3 (Different Blocks Same SM): %d\n", h_data);

    cudaFree(d_data);
    cudaFree(d_randStates);
}

__global__ void dataRaceKernelDifferentSMs(int *data, curandState *randStates)
{
    int idx = threadIdx.x;

    if (idx == 0 && (blockIdx.x == 0 || blockIdx.x == (gridDim.x - 1)))
    {
        artificialDelay(&randStates[idx], 100, 200);
        data[0] = blockIdx.x;
    }
}

void experiment4()
{
    int *d_data;
    int h_data;

    cudaMalloc((void **)&d_data, sizeof(int));
    h_data = -1;
    cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice);

    curandState *d_randStates;
    cudaMalloc((void **)&d_randStates, THREADS_PER_BLOCK * sizeof(curandState));
    initCurandStates<<<12, THREADS_PER_BLOCK>>>(d_randStates, time(0));
    cudaDeviceSynchronize();

    int numBlocks = 12;

    dataRaceKernelDifferentSMs<<<numBlocks, THREADS_PER_BLOCK>>>(d_data, d_randStates);
    cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Experiment 4 (Different SMs): %d\n", h_data);

    cudaFree(d_data);
    cudaFree(d_randStates);
}

int main(int argc, char **argv)
{
    printf("Running experiments to observe data races:\n\n");

    // experiment1();
    // cudaDeviceSynchronize();

    // experiment2();
    // cudaDeviceSynchronize();

    // experiment3();
    // cudaDeviceSynchronize();

    experiment4();
    cudaDeviceSynchronize();

    return 0;
}
