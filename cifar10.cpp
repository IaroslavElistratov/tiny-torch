#include <iostream> // todo: use C only
#include <stdio.h> // structure declaration called FILE
#include <string.h> // memcopy

using namespace std;

#define N_SAMPLES 4096


/*
The binary version contains the files data_batch_1.bin, data_batch_2.bin, ..., data_batch_5.bin, as well as test_batch.bin. Each of these files is formatted as follows:
<1 x label><3072 x pixel>
...
<1 x label><3072 x pixel>
In other words, the first byte is the label of the first image, which is a number in the range 0-9. The next 3072 bytes are the values of the pixels of the image.
The first 1024 bytes are the red channel values, the next 1024 the green, and the final 1024 the blue.
The values are stored in row-major order, so the first 32 bytes are the red channel values of the first row of the image.
Each file contains 10000 such 3073-byte "rows" of images, although there is nothing delimiting the rows. Therefore each file should be exactly 30730000 bytes long.
There is another file, called batches.meta.txt. This is an ASCII file that maps numeric labels in the range 0-9 to meaningful class names. It is merely a list of the 10 class names, one per row. The class name on row i corresponds to numeric label i.
*/
// 3072 (bits per img) / 3(channels) / 8 (bits in 1 byte) / 4 (bytes in 1 float) = 32
// 10000, 3072
// 10000, 3, 32, 32


// book, chapter 7.5 file access:
// fp is a pointer to a FILE, and fopen returns a pointer to a FILE
FILE *fopen(char *name, char *mode);

// getc returns the next character from a file; it needs the file pointer to tell it which file.
// it returns EOF for end of file or error
int getc(FILE *fp);

struct cifar10 {
    tensor* input;
    tensor* label;
};

// todo-now: doublecheck that logic here is correct
cifar10* get_cifar10(){
    set_backend_cpu();

    tensor* input = EmptyTensor(N_SAMPLES, 3, 32, 32);
    set_name(input, "input");
    tensor* label = EmptyTensor(N_SAMPLES, 1);
    set_name(label, "label");

    FILE *fp;
    if ((fp = fopen("../data/cifar-10-batches-bin/data_batch_1.bin", "rb")) == NULL) {
        printf("[cifar] Error: can't access file\n");
        exit(1);
    }

    // - byte_idx is for indexing the input buffer
    // - tensor_data_idx is for indexing tensor (IOW output buffer)
    //      - not the same as byte_idx -- byte_idx is larger bc it was also advanced for each label byte (associated with each img)
    //      - "data->data[byte_idx] = c/255." is incorrect, as it grows larger than "10000*3*32*32", bc it contains counts for labels
    //        as well (which my tensor* data has no space for), so need separate idxs for byte_idx and img_idx
    int c, byte_idx=0, img_idx=0, tensor_data_idx = 0;

    while ((c = getc(fp)) != EOF){

        if (img_idx > N_SAMPLES){
            printf("[cifar] Reached N_SAMPLES, stopping.");
            break;
        }

        // iow: (3*32*32 + 1)
        if (byte_idx % 3073 == 0){
            label->data[img_idx] = c;
            img_idx++;
        } else {
            if (tensor_data_idx>=input->size){
                printf("\n[cifar10] ERR!");
                exit(1);
            }
            // c is in range 0-255
            // printf("%f\n", c/255.);
            input->data[tensor_data_idx] = c/255.;
            tensor_data_idx++;
        }
        byte_idx++;
    }
    // breaks the connection between the file pointer and the external name that was established by f open, freeing the file pointer for another file
    // also, it flushes the buffer in which p u t c is collecting output
    fclose(fp);

    printf("\n[cifar] byte_idx: %i\n", byte_idx); // 30730000
    printf("\n[cifar] img_idx: %i\n", img_idx); // 10000

    // pack into a single structure to be returned by the function
    cifar10* dataset = (cifar10*)checkMallocErrors(malloc(sizeof(cifar10)));
    dataset->input = input;
    dataset->label = label;

    // note: the dataset is stored on the host, each batch is separately copied to device
    return dataset;
}


cifar10* sample_batch(cifar10* dataset, int batch_size, bool is_random){
    if (batch_size > N_SAMPLES){
        printf("[get_batch] error: saw batch_size larger than num samples in the dataset\n");
        exit(1);
    }

    tensor* x = EmptyTensor(batch_size, 3, 32, 32);
    set_name(x, "x");
    tensor* y = EmptyTensor(batch_size, 1);
    set_name(y, "y");

    for (int i=0; i<batch_size; i++){
        float* curr_x = x->data + i*x->stride[0];
        float* curr_y = y->data + i*y->stride[0];


        int idx;
        if (is_random){
            idx = (int)rand() % N_SAMPLES;
        } else {
            idx = i;
        }

        // select x, y at idx form the dataset
        float* sampled_x = dataset->input->data + idx * dataset->input->stride[0];
        int size_of_x = dataset->input->size / N_SAMPLES;
        memcpy(curr_x, sampled_x, size_of_x * sizeof(float));

        float* sampled_y = dataset->label->data + idx * dataset->label->stride[0];
        int size_of_y = dataset->label->size / N_SAMPLES;
        memcpy(curr_y, sampled_y, size_of_y * sizeof(float));
    }

    cifar10* batch = (cifar10*)checkMallocErrors(malloc(sizeof(cifar10)));
    batch->input = x;
    batch->label = y;

    set_backend_device();
    COPY_TO_DEVICE(batch->input);
    COPY_TO_DEVICE(batch->label);
    return batch;
}
