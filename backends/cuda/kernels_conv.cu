#define NUM_THREADS 32
#define CUDA_DEBUG true


// input(B, C, H, W) kernel(F, C, HH, WW) = out(B, F, h_out, w_out)
// input(B, C, H, W) kernel(F, C, HH, WW) = out(B, F, h_out, w_out)
__global__ void ConvKernel(float* x, float* kernel, float* out, int F, int H_OUT, int W_OUT, int H, int W, int C, int HH, int WW, bool is_batched){

    // out's idxs
    int curr_height = blockIdx.x * blockDim.x + threadIdx.x;
    int curr_width = blockIdx.y * blockDim.y + threadIdx.y;
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


// input(C, H, W) kernel(F, C, HH, WW) = out(F, h_out, w_out)
tensor* conv_k(tensor* input, tensor* kernel){
    if (CUDA_DEBUG) printf("[conv_k]\n");
    // unary_batched_input_checks(input);

    int C = input->shape[0], H = input->shape[1], W = input->shape[2];
    int F = kernel->shape[0], HH = kernel->shape[2], WW = kernel->shape[3];

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


// input (C, H, W) kernel (F, C, HH, WW) = out (F, h_out, w_out)
// input (B, C, H, W) kernel (F, C, HH, WW) = out (B, F, h_out, w_out)
tensor* batched_conv_k(tensor* input, tensor* kernel){
    if (CUDA_DEBUG) printf("[batched_conv_k]\n");
    // unary_batched_input_checks(input);

    int B = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];
    int F = kernel->shape[0], HH = kernel->shape[2], WW = kernel->shape[3];

    if (input->num_dims!=4 || kernel->num_dims!=4){
        printf("[cuda batched_conv_k] expected 3-d input and 4-d kernel\n");
        exit(1);
    }
    if (WW!=HH){
        printf("[cuda batched_conv_k] for now conv assumes square kernels\n");
        exit(1);
    }
    if (input->shape[1]!=kernel->shape[1]){
        printf("[cuda batched_conv_k] C-dim doesn't match\n");
        exit(1);
    }

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
