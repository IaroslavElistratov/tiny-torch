tensor* unpack_cifar_x(cifar10* batch){
    return batch->input;
}

tensor* unpack_cifar_y(cifar10* batch){
    return batch->label;
}


int get_rank(tensor* t){
    return t->num_dims;
}

int get_shape_at_idx(tensor* t, int idx){
    return t->shape[idx];
}


float item(tensor* t){
    if (t->size != 1){
        // not meaningful otherwise
        printf("[item] expected size == 1");
        exit(1);
    }
    return t->data[0];
}

// int get_gc_idx(){
//     return GC_IDX;
// }

// int set_gc_idx(int new_idx){
//     GC_IDX = new_idx;
// }

// float accuracy(tensor* log_probs, tensor* label){
//     // pred idxs
//     tensor* probs = exp(log_probs);
//     set_name(probs, "probs");
//     tensor* pred = batched_reduce_max(probs)->scratch_space[0];
//     set_name(pred, "pred");

//     pred = COPY_FROM_DEVICE(pred);
//     label = COPY_FROM_DEVICE(label);

//     // it's not a binary elementwise but same checks
//     // assert_binary_elementwise(pred, label);

//     int B = pred->shape[0];
//     int correct = 0;
//     for (int b=0; b<B; b++){
//         (pred->data[b] == label->data[b]) ? correct++ : 0;
//     }
//     float acc = (float)correct / B;
//     printf("accuracy: %f (%i/%i)\n", acc, correct, B);
//     return acc;
// };

void print_grad(tensor* t){
    print(t->grad);
}
