/*
    Run the following command to get debug info:
    nvcc -g -G stencil_7pt.cu -o bin/stencil_7pt
*/
#include <iostream>
#include <stdio.h>
#include <time.h>

#define BLOCK_SZ 16
#define IN_TILE_DIM BLOCK_SZ
#define OUT_TILE_DIM (IN_TILE_DIM - 2)
#define DEBUG false
#define C0 1
#define C1 1

using namespace std;

const float* getRandomMatrix(unsigned int X1, unsigned int X2, unsigned int X3) {
    unsigned int numElements = X1 * X2 * X3;
    float* matrix = new float[numElements];
    for(unsigned int i = 0; i < numElements; i++) {
        matrix[i] = rand() / (float)RAND_MAX;
        // matrix[i] = 1.0;
    }
    return (const float*)matrix;
}

void printMatrix(const float* matrix, unsigned int X1, unsigned int X2, unsigned int X3) {
    if(matrix == nullptr) {
        cout << "Cannot print null matrix";
        return;
    }
    for (unsigned int i = 0; i < X3; i++) {
        for (unsigned int j = 0; j < X2; j++) {
            for (unsigned int k = 0; k < X1; k++) {
                cout << int(matrix[i * X1 * X2 + j * X2 + k]) << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
}

float* cpu_inclusive_scan(const float* input, unsigned int X1, float* output) {
}

__global__ void gpu_inclusive_scan(const float* input, unsigned int X1, float* output) {
}

int main() {
    unsigned int X1 = DEBUG ? 5 : 512;
    unsigned int X2 = DEBUG ? 5: 512;
    unsigned int X3 = DEBUG ? 5: 512;
    printf("X1=%d X2=%d X3=%d\n", X1, X2, X3);
    const float* matrix_1 = getRandomMatrix(X1, X2, X3);

    if(DEBUG) {
        cout << "Matrix 1:" << endl;
        printMatrix(matrix_1, X1, X2, X3);
    }
    clock_t t0 = clock();
    float* matrix_ans = cpu_stencil_7pt(matrix_1, X1, X2, X3);
    clock_t t1 = clock();
    float cpu_time = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
    printf("CPU stencil took %fs\n", cpu_time);
    if(DEBUG) {
        cout << "Matrix Ans:" << endl;
        printMatrix(matrix_ans, X1, X2, X3);
    }

    clock_t t_gpu_overall_st = clock();
    // allocate memory on device
    float* matrix_1_d;
    float* matrix_ans_d;
    cudaMalloc(&matrix_1_d, X1 * X2 * X3 * sizeof(float));
    cudaMalloc(&matrix_ans_d, X1 * X2 * X3 * sizeof(float));
    cudaDeviceSynchronize();
    
    // copy to device
    cudaMemcpy(matrix_1_d, matrix_1, X1 * X2 * X3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // compute
    clock_t t2 = clock();
    dim3 numThreadsPerBlock(BLOCK_SZ, BLOCK_SZ);  // 1024 is the max allowed value for this
    int numBlocks_x = (max(max(X1, X2), X3) + OUT_TILE_DIM - 1) / OUT_TILE_DIM;
    int numBlocks_y = (max(max(X1, X2), X3) + OUT_TILE_DIM - 1) / OUT_TILE_DIM;
    if(DEBUG) {
        printf("numBlocks_x=%d, numBlocks_y=%d\n", numBlocks_x, numBlocks_y);
    }
    dim3 numBlocks(numBlocks_x, numBlocks_y);
    gpu_stencil_7pt <<< numBlocks, numThreadsPerBlock >>> (matrix_1_d, X1, X2, X3, matrix_ans_d);
    cudaDeviceSynchronize();

    // copy result to host
    float* gpu_matrix_ans = new float[X1 * X2 * X3];
    cudaMemcpy(gpu_matrix_ans, matrix_ans_d, X1 * X2 * X3 * sizeof(float), cudaMemcpyDeviceToHost);
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

    if(DEBUG) {
        cout << "Matrix GPU:" << endl;
        printMatrix(gpu_matrix_ans, X1, X2, X3);
    }

    // Check correctness
    float eps = 1e-2;
    for(unsigned int i = 0; i < X3; i++) {
        for(unsigned j = 0; j < X2; j++) {
            for(unsigned int k = 0; k < X1; k++) {
                float cpu_ans = matrix_ans[i * X1 * X2 + j * X2 + k];
                float gpu_ans = gpu_matrix_ans[i * X1 * X2 + j * X2 + k];
                if (fabs(cpu_ans - gpu_ans) >= eps) {
                    printf("At position (%d, %d, %d) matrix_ans has %f but gpu_matrix_ans has %f\n", i, j, k, cpu_ans, gpu_ans);
                    printf("Fatal error. Returning early.\n");
                    return 0;
                }
            }
        }
    }
    
    return 0;
}


