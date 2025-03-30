/*
    Run the following command to get debug info:
    nvcc -g -G reduction.cu -o bin/reduction
*/
#include <iostream>
#include <stdio.h>
#include <time.h>

#define BLOCK_SZ 4
#define TILE (2 * BLOCK_SZ)
#define DEBUG true

using namespace std;

const float *getRandomMatrix(unsigned int X1)
{
    unsigned int numElements = X1;
    float *matrix = new float[numElements];
    for (unsigned int i = 0; i < numElements; i++)
    {
        matrix[i] = rand() / (float)RAND_MAX;
        if (DEBUG)
            matrix[i] = 1.0;
    }
    return (const float *)matrix;
}

void printMatrix(const float *matrix, unsigned int X1)
{
    if (matrix == nullptr)
    {
        cout << "Cannot print null matrix";
        return;
    }
    for (unsigned int i = 0; i < X1; i++)
    {
        cout << int(matrix[i]) << " ";
    }
}

float *cpu_reduction(const float *vec, unsigned int X1)
{
    float *ans = new float[1];
    for (int i = 0; i < X1; i++)
    {
        ans[0] += vec[i];
    }
    return ans;
}

__global__ void gpu_reduction(float *vec, unsigned int X1, float *output)
{
    for (int level = 0; level < (X1 + TILE - 1) / TILE; level++)
    {
        // TODO: Use level inside to prevent useless operations
        int i = 2 * blockIdx.x * BLOCK_SZ + threadIdx.x;
        if (i < X1)
        {
            float s1 = vec[i];

            float s2 = 0.0;
            int j = i + BLOCK_SZ;
            if (j < X1)
                s2 = vec[j];
            __syncthreads();
            vec[i] = s1 + s2;
        }
        __syncthreads();
    }
    if (blockIdx.x == 0)  // TODO: cannot sync across blocks. Use atomicAdd
    {
        float finalSum = 0.0;
        for (int i = 0; i < min(X1, BLOCK_SZ); i++)
        {
            finalSum += vec[i];
        }
        output[0] = finalSum;
    }
}

int main()
{
    unsigned int X1 = DEBUG ? 32 : 128;
    printf("X1=%d \n", X1);
    const float *vec = getRandomMatrix(X1);

    if (DEBUG)
    {
        cout << "Matrix 1:" << endl;
        printMatrix(vec, X1);
        cout << endl;
    }
    clock_t t0 = clock();
    float *cpu_ans = cpu_reduction(vec, X1);
    clock_t t1 = clock();
    float cpu_time = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
    printf("CPU stencil took %fs\n", cpu_time);
    if (DEBUG)
    {
        cout << "CPU Ans: " << *cpu_ans << endl;
    }

    clock_t t_gpu_overall_st = clock();
    // allocate memory on device
    float *vec_d;
    float *ans_d;
    cudaMalloc(&vec_d, X1 * sizeof(float));
    cudaMalloc(&ans_d, 1 * sizeof(float));
    cudaDeviceSynchronize();

    // copy to device
    cudaMemcpy(vec_d, vec, X1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // compute
    clock_t t2 = clock();
    dim3 numThreadsPerBlock(BLOCK_SZ); // 1024 is the max allowed value for this
    int nBlocks = (X1 + TILE - 1) / TILE;
    if (DEBUG)
    {
        printf("numBlocks=%d\n", nBlocks);
    }
    dim3 numBlocks(nBlocks);
    gpu_reduction<<<numBlocks, numThreadsPerBlock>>>(vec_d, X1, ans_d);
    cudaDeviceSynchronize();

    // copy result to host
    float *gpu_ans = new float[1];
    cudaMemcpy(gpu_ans, ans_d, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    clock_t t3 = clock();
    float gpu_time = ((double)(t3 - t2)) / CLOCKS_PER_SEC;
    printf("GPU stencil took %fs\n", gpu_time);

    // free memory on device
    cudaFree(vec_d);
    cudaFree(ans_d);
    cudaDeviceSynchronize();
    clock_t t_gpu_overall_en = clock();
    float gpu_overall_time = ((double)(t_gpu_overall_en - t_gpu_overall_st)) / CLOCKS_PER_SEC;
    printf("Overall GPU stencil took %fs\n", gpu_overall_time);

    // Check correctness
    float eps = 1e-2;
    if (fabs(*cpu_ans - *gpu_ans) >= eps)
    {
        printf("cpu_ans has %f but gpu_ans has %f\n", *cpu_ans, *gpu_ans);
        return 0;
    }

    return 0;
}
