/*
    Run the following command to get debug info:
    nvcc -g -G stencil_7pt.cu -o bin/stencil_7pt
*/
#include <iostream>
#include <stdio.h>
#include <time.h>

#define BLOCK_SZ 32
#define IN_TILE_DIM BLOCK_SZ
#define OUT_TILE_DIM (IN_TILE_DIM - 2)
#define DEBUG true
#define C0 1
#define C1 1

using namespace std;

const float* getRandomMatrix(unsigned int X1, unsigned int X2, unsigned int X3) {
    unsigned int numElements = X1 * X2 * X3;
    float* matrix = new float[numElements];
    for(unsigned int i = 0; i < numElements; i++) {
        // matrix[i] = rand() / (float)RAND_MAX;
        matrix[i] = 1.0;
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

float* cpu_stencil_7pt(const float* matrix_1, unsigned int X1, unsigned int X2, unsigned int X3) {
    float* ans = new float[X1 * X2 * X3];
    for (unsigned int i = 0; i < X3; i++) {
        for (unsigned int j = 0; j < X2; j++) {
            for (unsigned int k = 0; k < X1; k++) {
                ans[i * X1 * X2 + j * X2 + k] = C0 * matrix_1[i * X1 * X2 + j * X2 + k] 
                                           + C1 * (
                                                (k == X1 - 1 ? 0 : matrix_1[i * X1 * X2 + j * X2 + k + 1])
                                                + (k == 0 ? 0 : matrix_1[i * X1 * X2 + j * X2 + k - 1])
                                                + (j == X2 - 1 ? 0 : matrix_1[i * X1 * X2 + (j + 1) * X2 + k])
                                                + (j == 0 ? 0 : matrix_1[i * X1 * X2 + (j - 1) * X2 + k])
                                                + (i == X3 - 1 ? 0 : matrix_1[(i + 1) * X1 * X2 + j * X2 + k])
                                                + (i == 0 ? 0 : matrix_1[(i - 1) * X1 * X2 + j * X2 + k])
                                            );
            }
        }
    }
    return ans;
}

__global__ void gpu_stencil_7pt(float* matrix_1_d, unsigned int X1, unsigned int X2, unsigned int X3,
                                float* matrix_ans_d) {
    __shared__ float A[IN_TILE_DIM][IN_TILE_DIM];
    // Each thread block will cover the whole X3 plane. This is thread coarsening - each thread is doing
    // more work than just computing the stencil for 1 element.
    int j = blockIdx.y * blockDim.y + threadIdx.y - 1;
    int k = blockIdx.x * blockDim.x + threadIdx.x - 1;

    float next = 0;
    float prev = 0;
    // threads at the boundary of shared memory load 0
    float el = 0;  // load ghost element for boundary elements
    if(j >= 0 && j < X2 
       && k >= 0 && k < X3
       // interior threads load from global memory
       && threadIdx.y >= 1 && threadIdx.y <= OUT_TILE_DIM
       && threadIdx.x >= 1 && threadIdx.x <= OUT_TILE_DIM
    ) {
        el = matrix_1_d[j * X2 + k];  // element of the first plane
    }
    A[j][k] = el;
    __syncthreads();

    for(unsigned int i = 0; i < X3; i++) {
        // next must be loaded from global memory each time
        next = i < X3 - 1 ? matrix_1_d[(i + 1) * X1 * X2 + j * X2 + k] : 0;
        // only interior threads perofrm computation
        if(threadIdx.y >= 1 && threadIdx.y <= OUT_TILE_DIM && threadIdx.x >= 1  && threadIdx.x <= OUT_TILE_DIM) {
            matrix_ans_d[i * X1 * X2 + j * X2 + k] = C0 * (A[threadIdx.y][threadIdx.x]);
                                                    //  + C1 * (
                                                    //     // 4 elements in the plane
                                                    //     A[threadIdx.y][threadIdx.x + 1]
                                                    //     + A[threadIdx.y][threadIdx.x - 1]
                                                    //     + A[threadIdx.y + 1][threadIdx.x]
                                                    //     + A[threadIdx.y - 1][threadIdx.x]
                                                    //     // next and prev plane on the register. 
                                                    //     // This is register tiling.
                                                    //     + next
                                                    //     + prev
                                                    //  );
        }
        prev = A[threadIdx.y][threadIdx.x];  // no other thread will need this value
        A[threadIdx.y][threadIdx.x] = next;  // other threads need this value
        __syncthreads();
    }
}

int main() {
    unsigned int X1 = DEBUG ? 3 : 1024;
    unsigned int X2 = DEBUG ? 3: 1024;
    unsigned int X3 = DEBUG ? 3: 1024;
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

    // Check correctness
    float eps = 1e-2;
    for(int i = 0; i < X3; i++) {
        for(int j = 0; j < X2; j++) {
            for(int k = 0; k < X1; k++) {
                float cpu_ans = matrix_ans[i * X3 + j*X2 + k];
                float gpu_ans = gpu_matrix_ans[i * X3 + j*X2 + k];
                if (fabs(cpu_ans - gpu_ans) >= eps) {
                    printf("At position (%d, %d) matrix_ans has %f but gpu_matrix_ans has %f\n", i, j, cpu_ans, gpu_ans);
                    // printf("Fatal error. Returning early.\n");
                    // return 0;
                }
            }
        }
    }
    
    return 0;
}


