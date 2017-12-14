
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

#define TILE_WIDTH 24


__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K , int W_grid) {

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    
    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a
    #define y4d(i3,i2,i1,i0) y[(i3) * (M * H_out * W_out) + (i2)*(H_out * W_out) + (i1)*(W_out) + i0]
    #define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]
    #define k4d(i3,i2,i1,i0) k[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]

    /*
        Your code here!
    */
    int x_tile_width = TILE_WIDTH + K -1;
    int b = blockIdx.x;
    int m = blockIdx.y;
    int h0 = threadIdx.y; 
    int h_base = (blockIdx.z/W_grid)*TILE_WIDTH;
    int w0 = threadIdx.x; 
    int w_base = (blockIdx.z%W_grid)*TILE_WIDTH;
    int h = h_base + h0;
    int w = w_base + w0;

    extern __shared__ float shmem[];
    float * x_shared = &shmem[0];
    float * k_shared = &shmem[x_tile_width*x_tile_width];

    float acc = 0;

    if (b<B & m<M & h<H_out & w<W_out){
        for (int c = 0;c<C;c++){
            //load kernel into shared memory
            if ((h0<K) && (w0<K)){
                k_shared[h0*K+w0] = k4d(m,c,h0,w0);
                // k_shared[h0][w0] = k4d(m,c,h0,w0);
            }
            __syncthreads();
            //load input x into shared memory x_shared
            for(int i=h;i<h_base+x_tile_width;i+=TILE_WIDTH){
                for (int j = w;j<w_base+x_tile_width;j+=TILE_WIDTH){
                    x_shared[(i-h_base)*x_tile_width+(j-w_base)] = x4d(b,c,i,j);
                    // x_shared[i-h_base)][j-w_base] = x4d(b,c,i,j);
                }
            }
            __syncthreads();
            //convolution
            for (int p=0;p<K;p++){
                for (int q=0;q<K;q++){
                    acc += x_shared[(h0+p)*x_tile_width+(w0+q)] * k_shared[p*K+q];
                }
            }
            __syncthreads();
        }
        y4d(b,m,h,w) = acc;
    }
    // if (b<B & m<M & h<H_out & w<W_out){
    //     for (int c=0; c<C;c++){
    //         for (int p=0;p<K;p++){
    //             for (int q=0;q<K;q++){
    //                 acc += x4d(b,c,h+p,w+q) * k4d(m,c,p,q);
    //             }
    //         }
    //     }
    //     y4d(b,m,h,w) = acc;
    // }

    #undef y4d
    #undef x4d
    #undef k4d
}

__global__ void matrixmulti(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K , int W_grid) {
    
}



void unroll(int B, int C, int H, int W, int K, float *X, float *X_unroll)  {
    int H_out = H – K + 1;
    int W_out = W – K + 1;
    for (int b = 0; b < B; ++b)
        for (int c = 0; c < C; ++c) {
            int w_base = c * (K*K);
            for (int p = 0; p < K; ++p) 
                for (int q = 0; q < K; ++q) {  
                    for (int h = 0; h <  H_out; ++h)
                        for (int w = 0; w < W_out; ++w) {  
                            int w_unroll = w_base + p * K + q;
                            int h_unroll = h * W_out + w;
                            // X_unroll[b, h_unroll, w_unroll] = X[b, c, h + p, w + q]; 
                            X_unroll[b * (C*H_out*W_out)+ h_unroll*(H_out*W_out)+w_unroll] = X[(b) * (C * H * W) + (c)*(H * W) + (h+p)*(W) + w+ q];
                            }
                    }
            }
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
    cudaStream_t s = y.stream_->stream_;

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

    
    int sizex = B * C * H * W * sizeof(float);
    int sizey = B * M * H_out * W_out * sizeof(float);
    int sizek = C * M * K * K * sizeof(float);
    size_t shmemsize = ((TILE_WIDTH+K-1)*(TILE_WIDTH+K-1)+K*K)*sizeof(float);
    // printf ("Bsize:%d\n", B);
    // printf ("Msize:%d\n", M);
    // printf ("Csize:%d\n", C);
    // printf ("Ksize:%d\n", K);
    // printf ("Hsize:%d,Width:%d\n", H,W);
    MSHADOW_CUDA_CALL(cudaMalloc ((void **) &devicex,sizex));
    MSHADOW_CUDA_CALL(cudaMalloc ((void **) &devicey,sizey));
    MSHADOW_CUDA_CALL(cudaMalloc ((void **) &devicek,sizek));
    
    MSHADOW_CUDA_CALL(cudaMemcpy (devicex,x.dptr_,sizex,cudaMemcpyHostToDevice));
    MSHADOW_CUDA_CALL(cudaMemcpy (devicek,w.dptr_,sizek,cudaMemcpyHostToDevice));


    int W_grid = W_out/TILE_WIDTH;
    int H_grid = H_out/TILE_WIDTH;
    int Y = W_grid * H_grid;
    dim3 gridDim(B, M, Y);
    dim3 blockDim(TILE_WIDTH,TILE_WIDTH,1);
    // Set the kernel dimensions
    // dim3 gridDim(0);
    // dim3 blockDim(0);

    // Call the kernel
    forward_kernel<<<gridDim, blockDim, shmemsize,s>>>(devicey,devicex,devicek, B,M,C,H,W,K,W_grid);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

    //
    MSHADOW_CUDA_CALL(cudaMemcpy (y.dptr_,devicey,sizey,cudaMemcpyDeviceToHost));
    cudaFree(devicek);
    cudaFree(devicex);
    cudaFree(devicey);

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
