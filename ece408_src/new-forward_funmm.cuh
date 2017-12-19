
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

// #define TILE_WIDTH 25
#define TILE_WIDTH 25
#define MAX_NUM_THREADS 1024
// __constant__ float K_const [50*1*25];

__global__ void forward_kernel(float *A, float *B, float *C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
// __global__ void forward_kernel(float *B, float *C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
        
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int Row = by*TILE_WIDTH+ty;
    int Col = bx*TILE_WIDTH+tx;
    int b = blockIdx.z;
    __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];
    float Cvalue = 0;
    for (int ph=0;ph<ceil(numAColumns/(float)TILE_WIDTH);++ph){
    if ((Row<numARows)&& ((ph*TILE_WIDTH+tx)<numAColumns))
    Ads[ty][tx] = A[Row*numAColumns+ph*TILE_WIDTH+tx];
    else Ads[ty][tx]=0;
    if ((ph*TILE_WIDTH+ty)<numBRows && (Col<numBColumns))
    Bds[ty][tx] = B[b*numBColumns*numBRows +(ph*TILE_WIDTH+ty)*numBColumns+Col];
    else Bds[ty][tx]=0;
    __syncthreads();
    for(int j=0;j<TILE_WIDTH;++j)
    Cvalue += Ads[ty][j]*Bds[j][tx];
    // Cvalue += K_const[Row*numAColumns+ph*TILE_WIDTH+j]*Bds[j][tx];
    __syncthreads();
    }
    if(Row<numCRows && Col<numCColumns)
    C[b*numCColumns*numCRows +Row*numCColumns+Col]=Cvalue;

}

__global__ void unroll_kernel(int C, int H, int W, int K, float *x, float *x_unroll){
    #define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]
    int c, s, h_out, w_out, h_unroll,w_unroll, h_base, p, q;
    int t = blockIdx.x * MAX_NUM_THREADS + threadIdx.x;
    int b = blockIdx.y;
    int H_out = H - K +1;
    int W_out = W - K + 1;
    int W_unroll = W_out * H_out;
    if (t< C * W_unroll){
        c = t/W_unroll;
        s = t%W_unroll;
        h_out = s/W_out;
        w_out = s%W_out;
        w_unroll = h_out * W_out + w_out;
        h_base = c * K * K;
        for(p=0;p<K;p++)
            for(q=0;q<K;q++){
                h_unroll = h_base+p*K+q;
                x_unroll[b*C*K*K*H_out*W_out+h_unroll*H_out*W_out+w_unroll] = x4d(b,c,h_out+p,w_out+q);
            }
   }
    #undef x4d
}






/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template<>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w) {
    

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    //CHECK_EQ(0, 1) << "Missing an ECE408 GPU implementation!";

    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    // cudaStream_t s = y.stream_->stream_;

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];

    // Set the kernel dimensions
    int W_out = W - K + 1;
    int H_out = H - K + 1;

    float *devicey;
    float *devicex;
    float *devicek;
    float *devicexUnroll;

    int sizex = B * C * H * W * sizeof(float);
    int sizey = B * M * H_out * W_out * sizeof(float);
    int sizek = C * M * K * K * sizeof(float);
    int sizexUnroll = B * C * K * K * H_out * W_out * sizeof(float);
    // size_t shmemsize = ((TILE_WIDTH+K-1)*(TILE_WIDTH+K-1)+K*K)*sizeof(float);
    int x_unroll_Row = C * K * K;
    int x_unroll_Col = H_out * W_out;
    // printf ("Bsize:%d\n", B);
    // printf ("Msize:%d\n", M);
    // printf ("Csize:%d\n", C);
    // printf ("Ksize:%d\n", K);
    // printf ("Hsize:%d,Width:%d\n", H,W);
    MSHADOW_CUDA_CALL(cudaMalloc ((void **) &devicex,sizex));
    MSHADOW_CUDA_CALL(cudaMalloc ((void **) &devicey,sizey));
    MSHADOW_CUDA_CALL(cudaMalloc ((void **) &devicek,sizek));
    MSHADOW_CUDA_CALL(cudaMalloc ((void **) &devicexUnroll,sizexUnroll));
    
    MSHADOW_CUDA_CALL(cudaMemcpy (devicex,x.dptr_,sizex,cudaMemcpyHostToDevice));
    MSHADOW_CUDA_CALL(cudaMemcpy (devicek,w.dptr_,sizek,cudaMemcpyHostToDevice));
    // cudaMemcpyToSymbol(K_const, w.dptr_, 50*25*sizeof(float));

    dim3 blockDim_unroll(MAX_NUM_THREADS,1,1);
    dim3 gridDim_unroll(ceil((C*H_out*W_out)/float(MAX_NUM_THREADS)),B,1);
    unroll_kernel<<<gridDim_unroll,blockDim_unroll>>>(C,H,W,K,devicex,devicexUnroll);

    // printf ("unrollkernelfinished\n");

    dim3 gridDim(ceil(H_out*W_out/(TILE_WIDTH*1.0)),ceil(M/(TILE_WIDTH*1.0)),B);
    dim3 blockDim(TILE_WIDTH,TILE_WIDTH,1);
    forward_kernel<<<gridDim,blockDim>>>(devicek,devicexUnroll,devicey,M,x_unroll_Row,x_unroll_Row,x_unroll_Col,M,x_unroll_Col);
    // forward_kernel<<<gridDim,blockDim>>>(devicexUnroll,devicey,M,x_unroll_Row,x_unroll_Row,x_unroll_Col,M,x_unroll_Col);
    
    // printf ("forwardkernelfinished\n");
    // int W_grid = ceil(W_out/float(TILE_WIDTH);
    // int H_grid = ceil(H_out/float(TILE_WIDTH));
    // int Y = W_grid * H_grid;
    // dim3 gridDim(B, M, Y);
    // dim3 blockDim(TILE_WIDTH,TILE_WIDTH,1);
    // // Set the kernel dimensions
    // // dim3 gridDim(0);
    // // dim3 blockDim(0);
    // // Call the kernel
    // forward_kernel<<<gridDim, blockDim, shmemsize,s>>>(devicey,devicex,devicek, B,M,C,H,W,K,W_grid);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

    //
    MSHADOW_CUDA_CALL(cudaMemcpy (y.dptr_,devicey,sizey,cudaMemcpyDeviceToHost));
    // cudaFree(devicek);
    // cudaFree(devicex);
    // cudaFree(devicey);
    // cudaFree(devicexUnroll);

}


/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
    assert( 0 && "No forward implementation for other datatypes needed for ECE408");
}

}
}

#endif
