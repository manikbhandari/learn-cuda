#include <iostream>
#include <stdio.h>
#include <time.h>

using namespace std;

const float* getRandomMatrix(unsigned int numRows, unsigned int numCols) {
    unsigned int numElements = numRows * numCols;
    float* matrix = new float[numRows * numCols];
    for (unsigned int i = 0; i < numElements; i++)
    {
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
            cout << matrix[i * numCols + j] << " ";
        }
        cout << endl;
    }
}

float* cpu_multiply(const float* matrix_1, unsigned int numRows_1, unsigned int numCols_1,
                    const float* matrix_2, unsigned int numRows_2, unsigned int numCols_2) {
    if(numCols_1 != numRows_2) {
        cout << "Cannot multiple matrices of shape (" << numRows_1 << ", " << numCols_1 << ")";
        cout << "and (" << numRows_2 << ", " << numCols_2 << ")" << endl;
        return nullptr;
    }
    float* matrix_ans = new float[numRows_1 * numCols_2];
    for (unsigned int i = 0; i < numRows_1; i++) {
        for (unsigned int j = 0; j < numCols_2; j++) {
            float sum = 0;
            for (unsigned int k = 0; k < numCols_1; k++) {
                sum += matrix_1[i * numCols_1 + k] * matrix_2[k * numCols_2 + j];
            }
            matrix_ans[i * numCols_2 + j] = sum;
        }
    }
    return matrix_ans;
}

__global__ void gpu_multiply(float* matrix_1_d, float* matrix_2_d, float* matrix_ans_d, 
                             int numRows_1, int numCols_1, int numRows_2, int numCols_2) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < numRows_1 && col < numCols_2) {  // boundary condition
        float sum = 0;
        for(int k = 0; k < numCols_1; k++) {
            float matrix_1_el = matrix_1_d[row * numCols_1 + k];
            float matrix_2_el = matrix_2_d[k * numCols_2 + col];
            sum += matrix_1_el * matrix_2_el;
        }
        matrix_ans_d[row * numCols_2 + col] = sum;
    }
}

int main() {
    unsigned int numRows_1 = 1000;
    unsigned int numCols_1 = 1000;
    unsigned int numRows_2 = 1000;
    unsigned int numCols_2 = 1000;
    const float* matrix_1 = getRandomMatrix(numRows_1, numCols_1);
    const float* matrix_2 = getRandomMatrix(numRows_2, numCols_2);

    // cout << "Matrix 1:" << endl;
    // printMatrix(matrix_1, numRows_1, numCols_1);
    // cout << "Matrix 2:" << endl;
    // printMatrix(matrix_2, numRows_2, numCols_2);
    clock_t t0 = clock();
    float* matrix_ans = cpu_multiply(matrix_1, numRows_1, numCols_1, matrix_2, numRows_2, numCols_2);
    clock_t t1 = clock();
    float cpu_time = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
    printf("CPU multiply took %fs\n", cpu_time);
    // cout << "Matrix Ans:" << endl;
    // printMatrix(matrix_ans, numRows_1, numCols_2);

    clock_t t_gpu_overall_st = clock();
    // allocate memory on device
    float* matrix_1_d;
    float* matrix_2_d;
    float* matrix_ans_d;
    cudaMalloc(&matrix_1_d, numRows_1 * numCols_1 * sizeof(float));
    cudaMalloc(&matrix_2_d, numRows_2 * numCols_2 * sizeof(float));
    cudaMalloc(&matrix_ans_d, numRows_1 * numCols_2 * sizeof(float));
    
    // copy to device
    cudaMemcpy(matrix_1_d, matrix_1, numRows_1 * numCols_1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_2_d, matrix_2, numRows_2 * numCols_2 * sizeof(float), cudaMemcpyHostToDevice);

    // compute
    clock_t t2 = clock();
    dim3 numThreadsPerBlock(32, 32);  // 1024 is the max allowed value for this
    int numBlocksPerThread_x = (numCols_2 + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x;
    int numBlocksPerThread_y = (numRows_1 + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y;
    dim3 numBlocksPerThread(numBlocksPerThread_x, numBlocksPerThread_y);
    gpu_multiply <<< numBlocksPerThread, numThreadsPerBlock >>> (matrix_1_d, matrix_2_d, matrix_ans_d, 
                                                                 numRows_1, numCols_1, numRows_2, numCols_2);

    // copy result to host
    float* gpu_matrix_ans = new float[numRows_1 * numCols_2];
    cudaMemcpy(gpu_matrix_ans, matrix_ans_d, numRows_1 * numCols_2 * sizeof(float), cudaMemcpyDeviceToHost);
    clock_t t3 = clock();
    float gpu_time = ((double)(t3 - t2)) / CLOCKS_PER_SEC;
    printf("GPU multiply took %fs\n", gpu_time);

    // free memory on device
    cudaFree(matrix_1_d);
    cudaFree(matrix_2_d);
    cudaFree(matrix_ans_d);
    clock_t t_gpu_overall_en = clock();
    float gpu_overall_time = ((double)(t_gpu_overall_en - t_gpu_overall_st)) / CLOCKS_PER_SEC;
    printf("Overall GPU multiply took %fs\n", gpu_overall_time);

    // Check correctness
    float eps = 1e-2;
    for(int i = 0; i < numRows_1; i++) {
        for(int j = 0; j < numCols_2; j++) {
            float cpu_ans = matrix_ans[i * numCols_2 + j];
            float gpu_ans = gpu_matrix_ans[i * numCols_2 + j];
            if (fabs(cpu_ans - gpu_ans) >= eps) {
                printf("At position (%d, %d) matrix_ans has %f but gpu_matrix_ans has %f\n", i, j, cpu_ans, gpu_ans);
            }
        }
    }
    
    return 0;
}