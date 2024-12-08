#include "nn.h"

#define MAX_NODES 30
#define UTILS_DEBUG false



void assert_contiguous(tensor* a){
    for (int expected_stride = 1, i=a->num_dims-1; i>=0; i--){
        int x = a->shape[i];
        int y = a->stride[i];
        // Skips checking strides when a dimension has length 1
        if (x == 1){
            continue;
        }
        if (y != expected_stride){
            printf("[assert_contiguous] Error: expected contiguous data. Saw:\n");
            sprint(a);
            exit(1);
        }
        expected_stride = expected_stride * x;
    }

}

void assert_device(tensor* a){
    if (a->device!=CUDA){
        printf("[assert_device] Error: expected device cuda\n");
        exit(1);
    }
}

void assert_dim(tensor* a, int expected_dim){
    if (a->num_dims!=expected_dim){
        printf("[assert_dim] Error: expected %i-dim inputs, saw %i-dim\n", expected_dim, a->num_dims);
        exit(1);
    }
}

void assert_input(tensor* a, int expected_dim){
    assert_contiguous(a);
    assert_device(a);
    assert_dim(a, expected_dim);
}



void maybe_init_grad(tensor* t){
    if (!t->grad){
        t->grad = TensorLikeFill(t, 0.0);
    } else {
        if (UTILS_DEBUG) printf("[maybe_init_grad] %s->grad exists!\n", t->name);
    }
}

void GetRandomFloat(float* dst, int num)
{
    for (int i=0; i<num; i++)
    {
        dst[i] = ((float)rand() / RAND_MAX) - 0.5;
    }
}

char* random_chars(int num){
    char* s = (char*)malloc(sizeof(char) * ++num);

    char offset = 'a';
    for (int i=0; i<num-1; i++){
        char sampled = rand() % 26; // 'z' - 'a' // 26 letters
        s[i] = offset + sampled;
    }
    s[num] = '\0';
    return s;
}

void set_name(tensor* t, const char* name){
    if (t->name){
        free(t->name);
    }

    t->name = (char*)malloc(sizeof(char) * MAX_TENSOR_NAME);

    int i=0;
    bool is_break = false;
    for (; !is_break && i<MAX_TENSOR_NAME-1; i++) {
        t->name[i] = name[i];
        if (name[i] == '\0')
            is_break = true;
    }

    if (!is_break && name[i+1] != '\0') {
        printf("[set_name] Warning, specified name larger than MAX_TENSOR_NAME -- truncating\n");
        t->name[i+1] = '\0';
    }
}


// todo-low: keras like vis https://graphviz.org/Gallery/directed/neural-network.html
void graphviz(tensor* tens){
    FILE *f = fopen("./generated/graph.txt", "w");
    if (f == NULL) {
        printf("[graphviz] Error opening file\n");
        exit(1);
    }

    fprintf(f, "digraph {\n");
    fprintf(f, "node [ordering=\"in\", fixedsize=shape shape=circle style=filled]\n");

    // will record pointers to all seen names -- to avid visiting same nodes twice, when
    //       exp
    //     /    \
    //    x1     x2
    char* all_visited[MAX_NODES]; // (float*)malloc(sizeof(float*) * MAX_NODES);
    // is used to index into all_visited
    int idx_visited = 0;

    deque <tensor*> ready;
    ready.push_front(tens);

    while (ready.size() > 0) {
        tensor* t = ready.back(); ready.pop_back();

        const char* op_name = OP_NAMES[t->op_type];
        const char* op_color = VIS_COLORS[t->op_type];

        // op -> output (there's only 1 output)
        fprintf(f, "%s_%s -> %s\n", op_name, t->name, t->name);

        // for ops, hide unique postfixes -- vis regular name (one of NUM_OPS) instead of unique name
        fprintf(f, "%s_%s [label=%s, fillcolor=%s, width=1.2, shape=diamond, style=\"rounded, filled\"]\n", op_name, t->name, op_name, op_color);

        for (int i=0; i<t->num_inputs; i++){
            tensor* inp = t->inputs[i];

            // check if we already visited this node
            bool is_visited = false;
            for (int i=0; i<idx_visited; i++){
                if (all_visited[i] == inp->name) {
                    is_visited = true;
                    break;
                }
            }

            // an input -> op
            fprintf(f, "%s -> %s_%s\n", inp->name, op_name, t->name);

            if (inp->num_dims==2)
                fprintf(f, "%s [shape=record, label=\"{shape=(%i, %i)}\"]\n", inp->name, inp->shape[0], inp->shape[1]);
            else if (inp->num_dims==3)
                fprintf(f, "%s [shape=record, label=\"{shape=(%i, %i, %i)}\"]\n", inp->name, inp->shape[0], inp->shape[1], inp->shape[2]);
            else if (inp->num_dims==4)
                fprintf(f, "%s [shape=record, label=\"{shape=(%i, %i, %i, %i)}\"]\n", inp->name, inp->shape[0], inp->shape[1], inp->shape[2], inp->shape[3]);

            // leafs don't have inputs to iterate over in the next iteration
            if (!inp->is_leaf && !is_visited) {
                ready.push_front(inp);
                all_visited[idx_visited++] = inp->name;
            }

        }
    }
    fprintf(f, "}\n");
    fclose(f);
}
