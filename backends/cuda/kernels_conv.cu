#define NUM_THREADS 32
#define CUDA_DEBUG true


// input(B, C, H, W) kernel(F, C, HH, WW) = out(B, F, h_out, w_out)
// input(B, C, H, W) kernel(F, C, HH, WW) = out(B, F, h_out, w_out)
__global__ void ConvKernel(float* x, float* kernel, float* out, int F, int H_OUT, int W_OUT, int H, int W, int C, int HH, int WW, bool is_batched){

    // out's idxs
    int curr_height = blockIdx.x * blockDim.x + threadIdx.x;
    int curr_width = blockIdx.y * blockDim.y + threadIdx.y;

    // gets canceled out when is_batch is false
    int b = blockIdx.z * is_batched;

    if (curr_height<H_OUT && curr_width<W_OUT){

        // e.g. 5/2=2 (with int division)

        // todo-now: bug was here.
        // int h_start = curr_height - (HH/2);
        // int w_start = curr_width - (WW/2);

        int h_start = curr_height - ((HH-1)/2);
        int w_start = curr_width - ((WW-1)/2);

        for (int f=0; f<F; f++){

            float curr_out = 0.0;

            // iterate over all elements where kernel overlays on the input

            // kernel's idxs
            for (int h=0; h<HH; h++){
                for (int w=0; w<WW; w++){

                    // x's idxs
                    int x_h = h_start + h;
                    int x_w = w_start + w;

                    // handles ghost cells (same semantics as padding with zeros)
                    if (x_h>-1 && x_h<H && x_w>-1 && x_w<W){

                            // for each input channel
                            for (int c=0; c<C; c++)
                                curr_out += x[b*C*H*W + c*H*W + x_h*W + x_w] * kernel[f*C*HH*WW + c*HH*WW + h*WW + w];

                    }
                }
            }

            out[b*F*H_OUT*W_OUT + f*H_OUT*W_OUT + curr_height*W_OUT + curr_width] = curr_out;

        }
    }

}


void input_checks_conv(tensor* input, tensor* kernel, int WW, int HH) {

    if (input->num_dims!=3 || kernel->num_dims!=4){
        printf("[cuda conv_k] expected 3-d input and 4-d kernel\n");
        exit(1);
    }
    if (WW!=HH){
        printf("[cuda conv_k] for now conv assumes square kernels\n");
        exit(1);
    }
    if (input->shape[0]!=kernel->shape[1]){
        printf("[cuda conv_k] C-dim doesn't match\n");
        exit(1);
    }

}


// input(C, H, W) kernel(F, C, HH, WW) = out(F, h_out, w_out)
tensor* conv_k(tensor* input, tensor* kernel) {
    int C = input->shape[0], H = input->shape[1], W = input->shape[2];
    int F = kernel->shape[0], HH = kernel->shape[2], WW = kernel->shape[3];

    if (CUDA_DEBUG) printf("[conv_k]\n");
    input_checks_conv(input, kernel, WW, HH);

    // todo: change h_out, w_out computation?
    // int h_out = (H - HH + 1) / stride;
    // int w_out = (W - WW + 1) / stride;
    int h_out = H - HH + 1;
    int w_out = W - WW + 1;

    // todo: allocate empty, here and other kenrels
    tensor* out = Tensor(F, h_out, w_out);

    float num_threads = (float)NUM_THREADS;
    // todo-high:
    // One possible design is to just add one more dim for F and remove loop over F from the kernel
    //  Another possible design is to keep the loop over F and (later in batch_conv) add grid.z for B (not F);
    //  With the 2nd approach parallel over B is cleaner in the code bc can have separate block-dim for that B specifically
    //  In the 1st approach need to cram both F and B into grid.z
    dim3 dimGrid(ceil(h_out/num_threads), ceil(w_out/num_threads), 1);
    dim3 dimBlock(num_threads, num_threads, 1);

    if (CUDA_DEBUG){
        printf("[cuda ConvKernel] grid: (%f, %f, 1)\n", ceil(h_out/num_threads), ceil(w_out/num_threads));
        printf("[cuda ConvKernel] block: (%f, %f, 1)\n", num_threads, num_threads);
    }

    ConvKernel<<<dimGrid, dimBlock>>>(input->data, kernel->data, out->data, F, h_out, w_out, H, W, C, HH, WW, false);

    return out;
}


void input_checks_batched_conv(tensor* input, tensor* kernel, int WW, int HH) {
    if (input->num_dims!=4 || kernel->num_dims!=4){
        printf("[cuda batched_conv_k] expected 3-d input and 4-d kernel\n");
        exit(1);
    }
    if (input->shape[1]!=kernel->shape[1]){
        printf("[cuda batched_conv_k] C-dim doesn't match\n");
        exit(1);
    }
    if (WW!=HH){
        printf("[cuda batched_conv_k] for now conv assumes square kernels\n");
        exit(1);
    }
}


// input (C, H, W) kernel (F, C, HH, WW) = out (F, h_out, w_out)
// input (B, C, H, W) kernel (F, C, HH, WW) = out (B, F, h_out, w_out)
tensor* batched_conv_k(tensor* input, tensor* kernel){
    int B = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];
    int F = kernel->shape[0], HH = kernel->shape[2], WW = kernel->shape[3];

    if (CUDA_DEBUG) printf("[batched_conv_k]\n");
    input_checks_batched_conv(input, kernel, WW, HH);

    int h_out = H - HH + 1;
    int w_out = W - WW + 1;

    tensor* out = Tensor(B, F, h_out, w_out);

    float num_threads = (float)NUM_THREADS;
    dim3 dimGrid(ceil(h_out/num_threads), ceil(w_out/num_threads), B);
    dim3 dimBlock(num_threads, num_threads, 1);

    if (CUDA_DEBUG){
        printf("[cuda BatchedConvKernel] grid: (%f, %f, %i)\n", ceil(h_out/num_threads), ceil(w_out/num_threads), B);
        printf("[cuda BatchedConvKernel] block: (%f, %f, 1)\n", num_threads, num_threads);
    }

    ConvKernel<<<dimGrid, dimBlock>>>(input->data, kernel->data, out->data, F, h_out, w_out, H, W, C, HH, WW, true);
    return out;
}


__global__ void BwdConvKernel(float* x, float* kernel, float* upstream, float* grad_x, float* grad_kernel, int F, int H_OUT, int W_OUT, int H, int W, int C, int HH, int WW, bool is_batched){

    // out's idxs
    int curr_height = blockIdx.x * blockDim.x + threadIdx.x;
    int curr_width = blockIdx.y * blockDim.y + threadIdx.y;

    int b = blockIdx.z * is_batched;

    if (curr_height<H_OUT && curr_width<W_OUT){

        int h_start = curr_height - ((HH-1)/2);
        int w_start = curr_width - ((WW-1)/2);

        for (int f=0; f<F; f++){

            // kernel's idxs
            for (int h=0; h<HH; h++){
                for (int w=0; w<WW; w++){

                    // x's idxs
                    int x_h = h_start + h;
                    int x_w = w_start + w;

                    if (x_h>-1 && x_h<H && x_w>-1 && x_w<W){

                            // for each input channel
                            for (int c=0; c<C; c++) {

                                // note: exact same indexing as in fwd (fwd: "curr_out += x[x_idx] * kernel[k_idx]")
                                int k_idx = f*C*HH*WW + c*HH*WW + h*WW + w;
                                int x_idx = b*C*H*W + c*H*W + x_h*W + x_w;

                                int u_idx = b*F*H_OUT*W_OUT + f*H_OUT*W_OUT + curr_height*W_OUT + curr_width;

                                grad_x[x_idx] += (kernel[k_idx] * upstream[u_idx]);
                                // grad_kernel[k_idx] += (x[x_idx] * upstream[u_idx]);

                                // todo-now: slow
                                atomicAdd(&grad_kernel[k_idx], (x[x_idx] * upstream[u_idx]));
                            }


                    }
                }
            }

        }
    }

}


// conv output, upstream: (F, h_out, w_out)
void bwd_conv_k(tensor* upstream, tensor* out) {
    tensor* input = out->inputs[0];
    tensor* kernel = out->inputs[1];

    int C = input->shape[0], H = input->shape[1], W = input->shape[2];
    int F = kernel->shape[0], HH = kernel->shape[2], WW = kernel->shape[3];

    if (CUDA_DEBUG) printf("[bwd_conv_k]\n");
    input_checks_conv(input, kernel, WW, HH);

    kernel->grad = TensorLikeFill(kernel, 0.0);
    input->grad = TensorLikeFill(input, 0.0);

    int h_out = H - HH + 1;
    int w_out = W - WW + 1;

    float num_threads = (float)NUM_THREADS;
    dim3 dimGrid(ceil(h_out/num_threads), ceil(w_out/num_threads), 1);
    dim3 dimBlock(num_threads, num_threads, 1);

    if (CUDA_DEBUG){
        printf("[cuda BwdConvKernel] grid: (%f, %f, 1)\n", ceil(h_out/num_threads), ceil(w_out/num_threads));
        printf("[cuda BwdConvKernel] block: (%f, %f, 1)\n", num_threads, num_threads);
    }

    BwdConvKernel<<<dimGrid, dimBlock>>>(input->data, kernel->data, upstream->data, input->grad->data, kernel->grad->data, F, h_out, w_out, H, W, C, HH, WW, false);
}

void bwd_batched_conv_k(tensor* upstream, tensor* out){
    tensor* input = out->inputs[0];
    tensor* kernel = out->inputs[1];

    int B = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];
    int F = kernel->shape[0], HH = kernel->shape[2], WW = kernel->shape[3];

    if (CUDA_DEBUG) printf("[batched_conv_k]\n");
    input_checks_batched_conv(input, kernel, WW, HH);

    kernel->grad = TensorLikeFill(kernel, 0.0);
    input->grad = TensorLikeFill(input, 0.0);

    int h_out = H - HH + 1;
    int w_out = W - WW + 1;

    float num_threads = (float)NUM_THREADS;
    dim3 dimGrid(ceil(h_out/num_threads), ceil(w_out/num_threads), B);
    dim3 dimBlock(num_threads, num_threads, 1);

    if (CUDA_DEBUG){
        printf("[cuda BwdConvKernel] grid: (%f, %f, %i)\n", ceil(h_out/num_threads), ceil(w_out/num_threads), B);
        printf("[cuda BwdConvKernel] block: (%f, %f, 1)\n", num_threads, num_threads);
    }
    BwdConvKernel<<<dimGrid, dimBlock>>>(input->data, kernel->data, upstream->data, input->grad->data, kernel->grad->data, F, h_out, w_out, H, W, C, HH, WW, true);
}




// input(?B, C, H, W) = out(?B, C, h_out, w_out)
__global__ void MaxPoolKernel(float* x, float* out, int H_OUT, int W_OUT, int H, int W, int C, int K, bool is_batched){

    // out's idxs
    int curr_height = blockIdx.x * blockDim.x + threadIdx.x;
    int curr_width = blockIdx.y * blockDim.y + threadIdx.y;

    // gets canceled out when is_batch is false
    int b = blockIdx.z * is_batched;

    if (curr_height<H_OUT && curr_width<W_OUT){

        int h_start = curr_height - ((K-1)/2);
        int w_start = curr_width - ((K-1)/2);


        // for each input channel
        for (int c=0; c<C; c++){

            // todo-now: set to 0-th element, here and in bwd_
            float max = -1000.0;

            // kernel's idxs
            for (int h=0; h<K; h++){
                for (int w=0; w<K; w++){

                    // x's idxs
                    int x_h = h_start + h;
                    int x_w = w_start + w;

                    if (x_h>-1 && x_h<H && x_w>-1 && x_w<W){

                        int x_idx = b*C*H*W + c*H*W + x_h*W + x_w;
                        if (x[x_idx] > max) max = x[x_idx];

                    }

                }
            }

            out[b*C*H_OUT*W_OUT + c*H_OUT*W_OUT + curr_height*W_OUT + curr_width] = max;

        }


    }

}

// input(C, H, W) = out(C, h_out, w_out)
tensor* maxpool_k(tensor* input) {
    int C = input->shape[0], H = input->shape[1], W = input->shape[2];

    if (CUDA_DEBUG) printf("[maxpool_k]\n");
    if (input->num_dims!=3){
        printf("[cuda maxpool_k] expected 3-d input\n");
        exit(1);
    }

    int K = 2;
    int h_out = H - K + 1;
    int w_out = W - K + 1;

    tensor* out = Tensor(C, h_out, w_out);

    float num_threads = (float)NUM_THREADS;
    dim3 dimGrid(ceil(h_out/num_threads), ceil(w_out/num_threads), 1);
    dim3 dimBlock(num_threads, num_threads, 1);

    if (CUDA_DEBUG){
        printf("[cuda MaxPoolKernel] grid: (%f, %f, 1)\n", ceil(h_out/num_threads), ceil(w_out/num_threads));
        printf("[cuda MaxPoolKernel] block: (%f, %f, 1)\n", num_threads, num_threads);
    }

    MaxPoolKernel<<<dimGrid, dimBlock>>>(input->data, out->data, h_out, w_out, H, W, C, K, false);

    return out;
}


// input(B, C, H, W) = out(B, C, h_out, w_out)
tensor* batched_maxpool_k(tensor* input) {
    int B = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];

    if (CUDA_DEBUG) printf("[maxpool_k]\n");
    if (input->num_dims!=4){
        printf("[cuda maxpool_k] expected 3-d input\n");
        exit(1);
    }

    int K = 2;
    int h_out = H - K + 1;
    int w_out = W - K + 1;

    tensor* out = Tensor(B, C, h_out, w_out);

    float num_threads = (float)NUM_THREADS;
    dim3 dimGrid(ceil(h_out/num_threads), ceil(w_out/num_threads), B);
    dim3 dimBlock(num_threads, num_threads, 1);

    if (CUDA_DEBUG){
        printf("[cuda MaxPoolKernel] grid: (%f, %f, %i)\n", ceil(h_out/num_threads), ceil(w_out/num_threads), B);
        printf("[cuda MaxPoolKernel] block: (%f, %f, 1)\n", num_threads, num_threads);
    }

    MaxPoolKernel<<<dimGrid, dimBlock>>>(input->data, out->data, h_out, w_out, H, W, C, K, true);

    return out;
}



// todo-high: can rm this func if just save idxs during forward, then the backward shared fn can just element-wise multiply ones at these idxs with upstream

// input(?B, C, H, W) = out(?B, C, h_out, w_out)
__global__ void BwdMaxPoolKernel(float* x, float* upstream, float* grad_x, int H_OUT, int W_OUT, int H, int W, int C, int K, bool is_batched){

    // out's idxs
    int curr_height = blockIdx.x * blockDim.x + threadIdx.x;
    int curr_width = blockIdx.y * blockDim.y + threadIdx.y;

    // gets canceled out when is_batch is false
    int b = blockIdx.z * is_batched;

    if (curr_height<H_OUT && curr_width<W_OUT){

        int h_start = curr_height - ((K-1)/2);
        int w_start = curr_width - ((K-1)/2);

        // for each input channel
        for (int c=0; c<C; c++){

            float x_max = -1000;
            int x_idx_max = -1;

            // kernel's idxs
            for (int h=0; h<K; h++){
                for (int w=0; w<K; w++){

                    // x's idxs
                    int x_h = h_start + h;
                    int x_w = w_start + w;

                    if (x_h>-1 && x_h<H && x_w>-1 && x_w<W){

                        int x_idx = b*C*H*W + c*H*W + x_h*W + x_w;
                        if (x[x_idx] > x_max) {
                            x_idx_max = x_idx;
                            x_max = x[x_idx];
                        }

                    }

                }
            }

            // assuming grad_x is initialized with 0s, below is the same as setting grad_x[x_idx_max] to 1, and then multiplying with upstream
            int u_idx = b*C*H_OUT*W_OUT + c*H_OUT*W_OUT + curr_height*W_OUT + curr_width;
            // grad_x[x_idx_max] += upstream[u_idx];
            atomicAdd(&grad_x[x_idx_max], upstream[u_idx]);

        }


    }

}



// input(C, H, W)
// upstream and out (C, h_out, w_out)
void bwd_maxpool_k(tensor* upstream, tensor* out) {
    tensor* input = out->inputs[0];

    int C = input->shape[0], H = input->shape[1], W = input->shape[2];

    if (CUDA_DEBUG) printf("[bwd_maxpool_k]\n");
    // input_checks_maxpool(tensor* input)

    int K = 2;
    int h_out = H - K + 1;
    int w_out = W - K + 1;

    input->grad = TensorLikeFill(input, 0.0);

    float num_threads = (float)NUM_THREADS;
    dim3 dimGrid(ceil(h_out/num_threads), ceil(w_out/num_threads), 1);
    dim3 dimBlock(num_threads, num_threads, 1);

    if (CUDA_DEBUG){
        printf("[cuda MaxPoolKernel] grid: (%f, %f, 1)\n", ceil(h_out/num_threads), ceil(w_out/num_threads));
        printf("[cuda MaxPoolKernel] block: (%f, %f, 1)\n", num_threads, num_threads);
    }

    BwdMaxPoolKernel<<<dimGrid, dimBlock>>>(input->data, upstream->data, input->grad->data, h_out, w_out, H, W, C, K, false);
}


void bwd_batched_maxpool_k(tensor* upstream, tensor* out) {
    tensor* input = out->inputs[0];

    int B = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];

    if (CUDA_DEBUG) printf("[bwd_maxpool_k]\n");
    // input_checks_maxpool(tensor* input)

    int K = 2;
    int h_out = H - K + 1;
    int w_out = W - K + 1;

    input->grad = TensorLikeFill(input, 0.0);

    float num_threads = (float)NUM_THREADS;
    dim3 dimGrid(ceil(h_out/num_threads), ceil(w_out/num_threads), B);
    dim3 dimBlock(num_threads, num_threads, 1);

    if (CUDA_DEBUG){
        printf("[cuda MaxPoolKernel] grid: (%f, %f, %i)\n", ceil(h_out/num_threads), ceil(w_out/num_threads), B);
        printf("[cuda MaxPoolKernel] block: (%f, %f, 1)\n", num_threads, num_threads);
    }

    BwdMaxPoolKernel<<<dimGrid, dimBlock>>>(input->data, upstream->data, input->grad->data, h_out, w_out, H, W, C, K, true);
}
