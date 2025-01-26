#include <iostream> // todo: use C only
#include <stdio.h> // structure declaration called FILE
#include <string.h> // memcopy



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
// FILE *fopen(char *name, char *mode);

// getc returns the next character from a file; it needs the file pointer to tell it which file.
// it returns EOF for end of file or error
int getc(FILE *fp);

struct cifar10 {
    tensor* input;
    tensor* label;
};


void read_file(tensor*, tensor*, const char *);

// made them global variables to persist between executions
// of read_file, and to avoid passing them as output of this fn
int byte_idx=0, img_idx=0, tensor_data_idx=0;

cifar10* get_cifar10(void){

    if (N_SAMPLES > 50000) {
        printf("[cifar] N_SAMPLES cannot be greater than 50,000\n");
        exit(1);
    }

    // reset global variables, in case get_validation_batch
    // was called before the current function
    byte_idx=0, img_idx=0, tensor_data_idx=0;

    set_backend_cpu();
    tensor* input = EmptyTensor(N_SAMPLES, 3, 32, 32);
    set_name(input, "input");
    tensor* label = EmptyTensor(N_SAMPLES, 1);
    set_name(label, "label");

    int num_per_file = 10000;

    // ceil: if divides without remainder no need to read one more file
    int num_files = N_SAMPLES / num_per_file + (N_SAMPLES % num_per_file != 0);
    // int num_files = ceil(N_SAMPLES, num_per_file);

    // maximum 5 batches
    num_files = min(5, num_files);

    for (int file_idx=1; file_idx<=num_files; file_idx++){
        // todo: determine automatically
        char file_name[46];
        snprintf(file_name, sizeof(char) * 46, "../data/cifar-10-batches-bin/data_batch_%i.bin", file_idx);

        read_file(input, label, file_name);
    }

    // pack into a single structure to be returned by the function
    cifar10* dataset = (cifar10*)checkMallocErrors(malloc(sizeof(cifar10)));
    dataset->input = input;
    dataset->label = label;

    sprint(dataset->input);
    sprint(dataset->label);

    set_backend_device();
    // note: the dataset is stored on the host, each batch is separately copied to device
    return dataset;
}

void read_file(tensor* input, tensor* label, const char* file_name){

    printf("file_name = %s\n", file_name);

    FILE *fp;
    // casting to const char bc mode ("rb") is const char -- and
    // fopen doesn't have overload for args: char, const char
    if (!(fp = fopen((file_name), "rb"))) {
        printf("[cifar] Error: can't access file\n");
        exit(1);
    }

    // - byte_idx is for indexing the input buffer
    // - tensor_data_idx is for indexing tensor (IOW output buffer)
    //      - not the same as byte_idx -- byte_idx is larger bc it was also advanced for each label byte (associated with each img)
    //      - "data->data[byte_idx] = c/255." is incorrect, as it grows larger than "10000*3*32*32", bc it contains counts for labels
    //        as well (which my tensor* data has no space for), so need separate idxs for byte_idx and img_idx

    int c;
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

            // note: chose the first normalization below bc
            // empirically it led to lower loss values

            // standardizing to range: -0.5:0.5:
            //   originally c is in range 0:255 -- transform into
            //   range 0:1, and then shift into range -0.5:0.5
            float value = (c / 255.) - 0.5;
            // printf("%f\n", value);

            // // standardizing to mean 0 and a std 1:
            // float mean[] = {125.306, 122.950, 113.865};
            // float std[] = {62.993, 62.088, 66.704};
            // // which of the 3 channels this value belongs to
            // int idx = (tensor_data_idx % 3072) / 1024;
            // float value = (c - mean[idx]) / std[idx];

            input->data[tensor_data_idx] = value;
            tensor_data_idx++;
        }
        byte_idx++;
    }
    // breaks the connection between the file pointer and the external name that was established by f open, freeing the file pointer for another file
    // also, it flushes the buffer in which p u t c is collecting output
    fclose(fp);

    printf("\n[cifar] byte_idx: %i\n", byte_idx); // 30730000
    printf("\n[cifar] img_idx: %i\n", img_idx); // 10000
}


cifar10* sample_batch(cifar10* dataset, int batch_size, bool is_random){
    if (batch_size > N_SAMPLES){
        printf("[get_batch] error: saw batch_size larger than num samples in the dataset\n");
        exit(1);
    }

    set_backend_cpu();
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


cifar10* get_validation_cifar10(void){

    set_backend_cpu();
    tensor* input = EmptyTensor(10000, 3, 32, 32);
    set_name(input, "val_input");
    tensor* label = EmptyTensor(10000, 1);
    set_name(label, "val_label");

    // reset global variables, they were incremented by
    // "get_cifar10" fn, assuming it ran before this fn
    byte_idx=0, img_idx=0, tensor_data_idx=0;
    read_file(input, label, "../data/cifar-10-batches-bin/test_batch.bin");

    cifar10* batch = (cifar10*)checkMallocErrors(malloc(sizeof(cifar10)));
    batch->input = input;
    batch->label = label;

    set_backend_device();
    COPY_TO_DEVICE(batch->input);
    COPY_TO_DEVICE(batch->label);
    return batch;
}
