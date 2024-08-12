/*
    Run the following command to get debug info:
    nvcc -g -G stencil_7pt.cu -o bin/stencil_7pt
*/
#include <iostream>
#include <stdio.h>
#include <time.h>

#define BLOCK_SZ 32
#define DEBUG true

using namespace std;

const float* getRandomMatrix(unsigned int numRows, unsigned int numCols) {
    unsigned int numElements = numRows * numCols;
    float* matrix = new float[numRows * numCols];
    for(unsigned int i = 0; i < numElements; i++) {
        matrix[i] = rand() / (float)RAND_MAX;
    }
    return (const float*)matrix;
}

void printMatrix(const float* matrix, unsigned int numRows, unsigned int numCols) {
    if(matrix == nullptr) {
        cout << "Cannot print null matrix";
        return;
    }
    for (unsigned int i = 0; i < numRows; i++) {
        for (unsigned int j = 0; j < numCols; j++)
        {
            cout << int(matrix[i * numCols + j]) << " ";
        }
        cout << endl;
    }
}

float* cpu_stencil_7pt(const float* matrix_1, unsigned int numRows_1, unsigned int numCols_1) {
}

__global__ void gpu_stencil_7pt(float* matrix_1_d, unsigned int numRows_1, unsigned int numCols_1,
                                float* matrix_ans_d) {
}

int main() {
    unsigned int numRows_1 = DEBUG ? 5 : 1024;
    unsigned int numCols_1 = DEBUG ? 5: 1024;
    const float* matrix_1 = getRandomMatrix(numRows_1, numCols_1);

    if(DEBUG) {
        cout << "Matrix 1:" << endl;
        printMatrix(matrix_1, numRows_1, numCols_1);
    }
    clock_t t0 = clock();
    float* matrix_ans = cpu_stencil_7pt(matrix_1, numRows_1, numCols_1);
    clock_t t1 = clock();
    float cpu_time = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
    printf("CPU stencil took %fs\n", cpu_time);
    if(DEBUG) {
        cout << "Matrix Ans:" << endl;
        printMatrix(matrix_ans, numRows_1, numCols_1);
    }

    clock_t t_gpu_overall_st = clock();
    // allocate memory on device
    float* matrix_1_d;
    float* matrix_ans_d;
    cudaMalloc(&matrix_1_d, numRows_1 * numCols_1 * sizeof(float));
    cudaMalloc(&matrix_ans_d, numRows_1 * numCols_1 * sizeof(float));
    cudaDeviceSynchronize();
    
    // copy to device
    cudaMemcpy(matrix_1_d, matrix_1, numRows_1 * numCols_1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // compute
    clock_t t2 = clock();
    dim3 numThreadsPerBlock(BLOCK_SZ, BLOCK_SZ);  // 1024 is the max allowed value for this
    int numBlocksPerThread_x = (numCols_1 + BLOCK_SZ - 1) / BLOCK_SZ;
    int numBlocksPerThread_y = (numRows_1 + BLOCK_SZ - 1) / BLOCK_SZ;
    if(DEBUG) {
        printf("numBlocksPerThread_x=%d, numBlocksPerThread_y=%d\n", numBlocksPerThread_x, numBlocksPerThread_y);
    }
    dim3 numBlocksPerThread(numBlocksPerThread_x, numBlocksPerThread_y);
    gpu_stencil_7pt <<< numBlocksPerThread, numThreadsPerBlock >>> (matrix_1_d, numRows_1, numCols_1, matrix_ans_d);
    cudaDeviceSynchronize();

    // copy result to host
    float* gpu_matrix_ans = new float[numRows_1 * numCols_1];
    cudaMemcpy(gpu_matrix_ans, matrix_ans_d, numRows_1 * numCols_1 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    clock_t t3 = clock();
    float gpu_time = ((double)(t3 - t2)) / CLOCKS_PER_SEC;
    printf("GPU stencil took %fs\n", gpu_time);

    // free memory on device
    cudaFree(matrix_1_d);
    cudaFree(matrix_ans_d);
    cudaDeviceSynchronize();
    clock_t t_gpu_overall_en = clock();
    float gpu_overall_time = ((double)(t_gpu_overall_en - t_gpu_overall_st)) / CLOCKS_PER_SEC;
    printf("Overall GPU stencil took %fs\n", gpu_overall_time);

    // Check correctness
    float eps = 1e-2;
    for(int i = 0; i < numRows_1; i++) {
        for(int j = 0; j < numCols_1; j++) {
            float cpu_ans = matrix_ans[i * numCols_1 + j];
            float gpu_ans = gpu_matrix_ans[i * numCols_1 + j];
            if (fabs(cpu_ans - gpu_ans) >= eps) {
                printf("At position (%d, %d) matrix_ans has %f but gpu_matrix_ans has %f\n", i, j, cpu_ans, gpu_ans);
                printf("Fatal error. Returning early.\n");
                return 0;
            }
        }
    }
    
    return 0;
}


