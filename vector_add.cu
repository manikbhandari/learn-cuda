#include <stdio.h>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


const int DSIZE = 1 << 28;
const int numThreadsPerBlock = 256;  // CUDA maximum is 1024
// vector add kernel: C = A + B
__global__ void vadd(const float *A, const float *B, float *C, int ds){
  int idx = blockIdx.x * blockDim.x + threadIdx.x; // create typical 1D thread index from built-in variables
  if (idx < ds) {
        C[idx] = A[idx] + B[idx];
  }
}

int main(){
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    h_A = new float[DSIZE];  // allocate space for vectors in host memory
    h_B = new float[DSIZE];
    h_C = new float[DSIZE];

    for (int i = 0; i < DSIZE; i++){  // initialize vectors in host memory
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
        h_C[i] = 0;
    }
    cudaMalloc(&d_A, DSIZE * sizeof(float));  // allocate device space for vector A
    cudaMalloc(&d_B, DSIZE * sizeof(float));  // allocate device space for vector B
    cudaMalloc(&d_C, DSIZE * sizeof(float));  // allocate device space for vector C
    cudaCheckErrors("cudaMalloc failure"); // error checking

    // copy vector A to device:
    cudaMemcpy(d_A, h_A, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    // copy vector B to device:
    cudaMemcpy(d_B, h_B, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    // Take ceiling number of blocks => launch some extra threads
    const int numBlocks = (DSIZE + numThreadsPerBlock - 1) / numThreadsPerBlock;
    vadd<<<numBlocks, numThreadsPerBlock>>>(d_A, d_B, d_C, DSIZE);
    cudaCheckErrors("kernel launch failure");

    // copy vector C from device to host:
    cudaMemcpy(h_C, d_C, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");

    // Why not synchronize here?
    printf("A[0] = %f\n", h_A[0]);
    printf("B[0] = %f\n", h_B[0]);
    printf("C[0] = %f\n", h_C[0]);
    return 0;
}
  