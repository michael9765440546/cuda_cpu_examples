#include <iostream>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#define N 3

int A[N][N] = {
    {5, 7, 0},
    {7, 4, 8},
    {2, 7, 4}
};

int B[N][N] = {
    {0, 9, 8},
    {6, 4, 1},
    {1, 4, 9}
};

int C[N][N] = {0};

#ifdef __CUDACC__
__global__ void matMulKernel(int *A, int *B, int *C, int N) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    int sum = 0;
    for(int k=0; k<N; k++) {
        sum += A[row*N + k] * B[k*N + col];
    }
    C[row*N + col] = sum;
}
#endif

void matMulCPU() {
    for(int i=0;i<N;i++) {
        for(int j=0;j<N;j++) {
            int sum = 0;
            for(int k=0;k<N;k++) sum += A[i][k]*B[k][j];
            C[i][j] = sum;
        }
    }
}

int main() {
#ifdef __CUDACC__
    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N*N*sizeof(int));
    cudaMalloc((void**)&d_B, N*N*sizeof(int));
    cudaMalloc((void**)&d_C, N*N*sizeof(int));

    cudaMemcpy(d_A, A, N*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N*N*sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(N, N);
    matMulKernel<<<1, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, N*N*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
#else
    matMulCPU();
#endif

    std::cout << "Matrix A:\n";
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++) std::cout << A[i][j] << " ";
        std::cout << "\n";
    }
    std::cout << "\nMatrix B:\n";
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++) std::cout << B[i][j] << " ";
        std::cout << "\n";
    }
    std::cout << "\nMatrix C = A * B:\n";
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++) std::cout << C[i][j] << " ";
        std::cout << "\n";
    }
    return 0;
}
