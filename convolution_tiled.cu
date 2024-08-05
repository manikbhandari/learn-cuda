#include <iostream>
#include <stdio.h>
#include <time.h>

// TODO: this will only work for odd mask dim
#define MASK_DIM 5
#define BLOCK_SZ 32
#define TILE_DIM (BLOCK_SZ - MASK_DIM)

using namespace std;

__constant__ float mask_c[MASK_DIM][MASK_DIM];

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

float* cpu_convolution(const float* matrix_1, unsigned int numRows_1, unsigned int numCols_1, 
                       const float* matrix_2, unsigned int numRows_2, unsigned int numCols_2) {
    float* matrix_ans = new float[numRows_1 * numCols_1];
    int radius_h = numRows_2 / 2;
    int radius_w = numCols_2 / 2;
    if(radius_h < 1 || radius_w < 1) {
        printf("radius_h=%d, radius_w=%d but neither can be < 1", radius_h, radius_w);
    }
    // printf("radius_h=%d, radius_w=%d\n", radius_h, radius_w);
    for(int i_1 = 0; i_1 < numRows_1; i_1++){
        for(int j_1 = 0; j_1 < numCols_1; j_1++) {
            float sum = 0;
            for(int i_2 = i_1 - radius_h; i_2 <= i_1 + radius_h; i_2++){
                for(int j_2 = j_1 - radius_w; j_2 <= j_1 + radius_w; j_2++) {
                    float mat1_el = 0;
                    if(i_2 >= 0 && i_2 < numRows_1 && j_2 >= 0 and j_2 < numCols_1) {
                        mat1_el = matrix_1[i_2 * numCols_1 + j_2];
                    }
                    float mat2_el = matrix_2[(i_2 - i_1 + radius_h) * numCols_2 + (j_2 - j_1 + radius_w)];
                    // printf("mat1_el=%f, mat2_el=%f\n", mat1_el, mat2_el);
                    sum += mat1_el * mat2_el;
                }
            }
            matrix_ans[i_1 * numCols_1 + j_1] = sum;
        }
    }
    return matrix_ans;
}

__global__ void gpu_convolution_tiled(const float* matrix_1_d, unsigned int numRows_1, unsigned int numCols_1, 
                                unsigned int numRows_2, unsigned int numCols_2, 
                                float* matrix_ans_d) {

    int radius_h = numRows_2 / 2;  // same as mask_dim / 2
    int radius_w = numCols_2 / 2;

    int i_1 = blockIdx.y * blockDim.y + threadIdx.y;  
    int j_1 = blockIdx.x * blockDim.x + threadIdx.x;

    // load shared memory. Ghost elements load 0
    __shared__ float A[BLOCK_SZ][BLOCK_SZ];
    float el = 0;
    int i_el = i_1 - blockIdx.y * radius_h * 2 - radius_h;
    int j_el = j_1 - blockIdx.x * radius_w * 2 - radius_w;
    if(i_el >= 0 && i_el < numRows_1 && j_el >= 0 && j_el < numCols_1) {
        el = matrix_1_d[i_el * numCols_1 + j_el];
    }
    A[threadIdx.y][threadIdx.x] = el;
    __syncthreads();

    if(i_el >= 0 && i_el < numRows_1 && j_el >= 0 && j_el < numCols_1
       && threadIdx.y >= radius_h && threadIdx.y + radius_h < BLOCK_SZ
       && threadIdx.x >= radius_w && threadIdx.x + radius_w < BLOCK_SZ) {
        float sum = 0.0;
        for(int i_2 = threadIdx.y - radius_h; i_2 <= threadIdx.y + radius_h; i_2++){
            for(int j_2 = threadIdx.x - radius_w; j_2 <= threadIdx.x + radius_w; j_2++) {
                float mat1_el = A[i_2][j_2];
                float mat2_el = mask_c[(i_2 - threadIdx.y + radius_h)][(j_2 - threadIdx.x + radius_w)];
                sum += mat1_el * mat2_el;
            }
        }
        matrix_ans_d[i_el * numCols_1 + j_el] = sum;
    }
    __syncthreads();
}

int main() {
    unsigned int numRows_1 = 1024;
    unsigned int numCols_1 = 1024;
    // TODO: Use 1-d array mask to use constant memory dynamically
    unsigned int numRows_2 = MASK_DIM;
    unsigned int numCols_2 = MASK_DIM;
    const float* matrix_1 = getRandomMatrix(numRows_1, numCols_1);
    const float* matrix_2 = getRandomMatrix(numRows_2, numCols_2);

    // cout << "Matrix 1:" << endl;
    // printMatrix(matrix_1, numRows_1, numCols_1);
    // cout << "Matrix 2:" << endl;
    // printMatrix(matrix_2, numRows_2, numCols_2);
    clock_t t0 = clock();
    float* matrix_ans = cpu_convolution(matrix_1, numRows_1, numCols_1, matrix_2, numRows_2, numCols_2);
    clock_t t1 = clock();
    float cpu_time = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
    printf("CPU multiply took %fs\n", cpu_time);
    // cout << "Matrix Ans:" << endl;
    // printMatrix(matrix_ans, numRows_1, numCols_1);

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
    // copy to constant memory
    cudaMemcpyToSymbol(mask_c, matrix_2, numRows_2 * numCols_2 * sizeof(float));
    cudaDeviceSynchronize();

    // compute
    clock_t t2 = clock();
    dim3 numThreadsPerBlock(BLOCK_SZ, BLOCK_SZ);  // 1024 is the max allowed value for this
    int numBlocksPerThread_x = (numCols_1 + TILE_DIM - 1) / TILE_DIM;
    int numBlocksPerThread_y = (numRows_1 + TILE_DIM - 1) / TILE_DIM;
    // printf("numBlocksPerThread_x=%d, numBlocksPerThread_y=%d\n", numBlocksPerThread_x, numBlocksPerThread_y);
    dim3 numBlocksPerThread(numBlocksPerThread_x, numBlocksPerThread_y);
    gpu_convolution_tiled <<< numBlocksPerThread, numThreadsPerBlock >>> (matrix_1_d, numRows_1, numCols_1, 
                                                                    numRows_2, numCols_2, matrix_ans_d);
    cudaDeviceSynchronize();

    // copy result to host
    float* gpu_matrix_ans = new float[numRows_1 * numCols_1];
    cudaMemcpy(gpu_matrix_ans, matrix_ans_d, numRows_1 * numCols_1 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    clock_t t3 = clock();
    float gpu_time = ((double)(t3 - t2)) / CLOCKS_PER_SEC;
    printf("GPU multiply took %fs\n", gpu_time);

    // free memory on device
    cudaFree(matrix_1_d);
    cudaFree(matrix_ans_d);
    cudaDeviceSynchronize();
    clock_t t_gpu_overall_en = clock();
    float gpu_overall_time = ((double)(t_gpu_overall_en - t_gpu_overall_st)) / CLOCKS_PER_SEC;
    printf("Overall GPU multiply took %fs\n", gpu_overall_time);

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

