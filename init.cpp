#include "nn.h"


// uniform random numbers in [-0.5-0.5]
void uniform_init(tensor* t) {
    float* dst = t->data;
    for (int i=0; i<t->size; i++) {
        // https://linux.die.net/man/3/random
        // returns a pseudo-random int between 0 and RAND_MAX
        // normalize to: 0 - 1
        // shift to: -0.5 - 0.5

        // not truncating to 0 due to int division, bc C promotes args
        dst[i] = ((float)rand() / RAND_MAX) - 0.5;
    }
}



// Mersenne Twister
#include <random>
// Seed C++ generator
std::mt19937 mt(std::random_device{}());
// Something like the Box-Muller method
std::normal_distribution<float> normal_dist{0.0, 1.0};
std::uniform_real_distribution<float> uniform_distribution(0.0, 1.0);

void normal_init(tensor* t){
    for (int i=0; i<t->size; i++){
        t->data[i] = (float)normal_dist(mt) * 0.1;
    }
}

// void unifrom_init(tensor* t){
//     for (int i=0; i<t->size; i++){
//         t->data[i] = (float)uniform_distribution(mt) - 0.5;
//     }
// }




// https://github.com/pytorch/pytorch/blob/5802be698eff17cf4b6284056dc8e89c48befc00/torch/nn/init.py#L345
tuple* _calculate_fan_in_and_fan_out(tensor* t){
    if (t->num_dims < 2){
        printf("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions\n");
        exit(1);
    }

    int num_input_fmaps = t->shape[1];
    int num_output_fmaps = t->shape[0];
    int receptive_field_size = 1;

    if (t->num_dims > 2) {
        for (int dim_idx=2; dim_idx<t->num_dims; dim_idx++){
            int s = t->shape[dim_idx];
            // printf("iterating over s: %i\n", s);
            receptive_field_size *= s;
        }
    }
    int fan_in = num_input_fmaps * receptive_field_size;
    int fan_out = num_output_fmaps * receptive_field_size;
    return get_tuple(fan_in, fan_out);
}

// https://github.com/pytorch/pytorch/blob/78bff1e8c1bd0b30e27fbc79d5a14a1c5a92d4a7/torch/nn/init.py#L119C5-L120C30
float _calculate_gain(const char* nonlinearity){
    if (!nonlinearity) {
        return 1.0;
    } else if (strcmp(nonlinearity, "relu") == 0){
        return sqrt(2.0);
    } else {
        printf("[calculate_gain] Unsupported nonlinearity %s\n", nonlinearity);
        exit(1);
    }
}

// can't create Tensor inside this fn:
//  - causes infinite recursion -- this fn is called from the
//    constructor, and creating a tensor would call the constructor
//  - further, it would create a bunch of new tensors which don't get
//    added to GC (bc they happen when initializing params -- IOW before the "gc_until" is set)

// todo-high:
//  - for now hardcoding gain, but -- gain should be 2 if ReLU is followed by the layer, or 1 if not
//  - all the intermidiate tensors (e.g. created inside an op) -- will also call
//    this initialization which is probably undesirable (probably don't have relu after them)?

// todo-high: init bias any differently?
//  https://github.com/pytorch/pytorch/blob/a86fa779ce3482324a0d1fbb12d87a95a981f0a3/torch/nn/modules/linear.py#L114

// https://github.com/pytorch/pytorch/blob/78bff1e8c1bd0b30e27fbc79d5a14a1c5a92d4a7/torch/nn/init.py#L516-L518
void kaiming_uniform_init(tensor* t){
    float gain = _calculate_gain("relu");
    float fan_in = _calculate_fan_in_and_fan_out(t)->item_1;
    float std = gain / sqrt(fan_in);

    // Calculate uniform bounds from standard deviation
    float bound = sqrt(3.0) * std;

    // uses more sophisticated mt19937 PRNG
    for (int i=0; i<t->size; i++){
        float uniform = uniform_distribution(mt); // ((float)mt() / mt.max());

        // transform random variable from a uniform distribution in range [0, 1] to a uniform distribution in range [−bound, bound]
        //  - shift from [0, 1] to [-0.5, 0.5] by subtracting 0.5
        //  - scale by 2 * bound to spread to [-bound, bound]
        uniform = 2*bound * (uniform-0.5);
        t->data[i] = uniform;

        // // similar range as my uniform_init -- -0.5:0.5
        // uniform -= 0.5;
        // t->data[i] = uniform * std;
    }
}


// https://github.com/pytorch/pytorch/blob/78bff1e8c1bd0b30e27fbc79d5a14a1c5a92d4a7/torch/nn/init.py#L521
void kaiming_normal_init(tensor* t){
    float gain = _calculate_gain("relu");
    float fan_in = _calculate_fan_in_and_fan_out(t)->item_1;
    float std = gain / sqrt(fan_in);

    // uses more sophisticated mt19937 PRNG
    for (int i=0; i<t->size; i++){
        float normal = (float)normal_dist(mt);
        t->data[i] = normal * std;
    }
}
