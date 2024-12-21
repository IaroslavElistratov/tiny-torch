
bool IS_INPUT_DIM_CHECK = true;

extern void assert_device(tensor* a);



void assert_contiguous(tensor* a){
    // https://github.com/pytorch/pytorch/blob/dc7461d6f571abb8a6649d0c026793e77d0fd411/torch/_prims_common/__init__.py#L249-L271

    // iterate in the reverse order of shapes and strides

    // a tensor is not contiguous if (to access elements in next dim)
    // you need to skip more/less elements, than num elements in all
    // the previous dims of the tensor

    // "-1" bc if e.g. t->num_dims is 2, then only valid locations
    // for shape are t->shape[0] and t->shape[1] -- IOW t->num_dims - 1
    for (int expected_stride = 1, i=a->num_dims-1; i>=0; i--){
        int x = a->shape[i];
        int y = a->stride[i];
        // Skips checking strides when a dimension has length 1
        if (x == 1){
            continue;
        }
        if (UTILS_DEBUG) printf("[assert_contiguous] (i=%i) y=%i , expected_stride=%i\n", i, y, expected_stride);
        if (y != expected_stride){
            printf("[assert_contiguous] Error: expected contiguous data. Saw:\n");
            sprint(a);
            exit(1);
        }
        expected_stride = expected_stride * x;
    }

}

void assert_dim(tensor* a, int expected_dim){
    if (a->num_dims!=expected_dim){
        printf("[assert_dim] Error: expected %i-dim inputs, saw %i-dim\n", expected_dim, a->num_dims);
        exit(1);
    }
}

// todo: would be convenient if this fn also expected a string to be printed (in case of err raised) as an argument
//  e.g. "[cuda conv_k] expected 3-d input and 4-d kernel\n"
// current workaround: launch cuda-gdb; b exit; run; bt
void assert_input(tensor* a, int expected_dim){
    assert_contiguous(a);
    assert_device(a);
    // use -1 when there's not requirement on a particular input ndims -- avoids to reuse assert_input for these cases instead of using "assert_contiguous(a); assert_device(a);"
    if (expected_dim != -1){
        assert_dim(a, expected_dim);
    }
}



void assert_same_size(tensor* a, tensor* b){
    if (a->size != b->size){
        printf("[assert_same_size] Error: expected inputs sizes to match. Saw:\n");
        sprint(a);
        sprint(b);
        exit(1);
    }
}

void assert_same_shape(tensor* a, tensor* b){
    if (a->num_dims != b->num_dims){
        printf("[assert_same_shape] Error: expected inputs of same dimensionality. Saw:\n");
        sprint(a);
        sprint(b);
        exit(1);
    }

    for (int i=0; i<a->num_dims; i++){
        if (a->shape[i] == b->shape[i]){
            continue;
        }
        printf("[assert_same_shape] Error: expected input shapes to match. Saw:\n");
        sprint(a);
        sprint(b);
        exit(1);
    }
}

void assert_binary_elementwise(tensor* a, tensor* b){
    // don't assert n_dim == 2, it's expected that 3d and 4d input will be fed to them,
    // e.g. mul_k_ is used in many bwd functions; add_k_ is used in batched_flatten_bwd
    assert_contiguous(a);
    assert_device(a);

    assert_contiguous(b);
    assert_device(b);

    // this is basically a side-hatch for _unsafe_add_k;
    // batched_flatten_k calls add_k_ with a(B, 24) b(B, 6, 2, 2)
    // because kernels iterate over size, seems the below is more suitable check when comparing shapes
    if (!IS_INPUT_DIM_CHECK){
        assert_same_size(a, b);
        return;
    }

    assert_same_shape(a, b);
}


void assert_binary_elementwise_non_contiguous(tensor* a, tensor* b){
    assert_device(a);
    assert_device(b);

    // this is basically a side-hatch for _unsafe_add_k
    if (!IS_INPUT_DIM_CHECK){
        assert_same_size(a, b);
        return;
    }

    assert_same_shape(a, b);
}
