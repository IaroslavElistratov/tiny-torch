#include <iostream> // todo: use C only
#include <stdio.h> // structure declaration called FILE

using namespace std;

#define N_SAMPLES 10000 // 10000


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
    cifar10* dataset = (cifar10*)malloc(sizeof(cifar10));

    dataset->input = input;
    dataset->label = label;

    set_backend_device();
    COPY_TO_DEVICE(dataset->input);
    COPY_TO_DEVICE(dataset->label);
    return dataset;
}
