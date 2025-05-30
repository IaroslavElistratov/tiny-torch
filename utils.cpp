#include <sys/stat.h> // mkdir

#include "nn.h"

#define MAX_NODES 30
#define UTILS_DEBUG false


// void log_print(FILE* file, const char* format, ...) {
//     va_list args;

//     // Write to stdout
//     va_start(args, format);
//     vprintf(format, args);
//     va_end(args);

//     // Write to the file
//     va_start(args, format);
//     vfprintf(file, format, args);
//     va_end(args);
// }

void log_print(const char* format, ...) {

    va_list args;

    // Write to stdout
    va_start(args, format);
    vprintf(format, args);
    va_end(args);

    // Write to the file
    FILE* file = fopen("./generated/checkpoints/stdout.txt", "a");
    if (!file) {
        printf("Error opening file\n");
        exit(1);
    }

    va_start(args, format);
    vfprintf(file, format, args);
    va_end(args);

    fclose(file);
}


void log_print_macros(void){
    // size of the training job
    log_print("N_SAMPLES: %i\n", N_SAMPLES);
    log_print("BATCH_SIZE: %i\n", BATCH_SIZE);
    log_print("NUM_EP: %i\n", NUM_EP);

    // optimization related
    log_print("IS_LOAD: %i\n", IS_LOAD);
    log_print("IS_STOCHASTIC: %i\n", IS_STOCHASTIC);
    log_print("LR: %f\n", LR);

    // misc
    log_print("SAVE_EVERY: %i\n", SAVE_EVERY);
    log_print("DEVICE: %i\n", DEVICE);
}

void flush_io_buffers(void){
    mkdir("./generated", 0755);
    mkdir("./generated/checkpoints", 0755);
    fclose(fopen("./generated/log.txt", "w"));
    fclose(fopen("./generated/checkpoints/train_loss.txt", "w"));
    fclose(fopen("./generated/checkpoints/val_loss.txt", "w"));
    fclose(fopen("./generated/checkpoints/stdout.txt", "w"));
}

void save_loss(const char* f_name, float loss){
    // save them into 2 different files -- the resolution for the train loss is ofc higher, so doesn't make sense to write them into the same array (bc this would require skipping saving most of the training losses)

    char path[40];
    snprintf(path, sizeof(char) * 40, "./generated/checkpoints/%s.txt", f_name);

    FILE *f = fopen(static_cast<const char*>(path), "a");
    if (!f) {
        printf("Error opening file\n");
        exit(1);
    }
    fprintf(f, "%.3f, ", loss);
    fclose(f);
}

tuple* get_tuple(float val1, float val2){
    // todo: a better way of doing this?
    tuple* out = (tuple*)checkMallocErrors(malloc(sizeof(tuple)));
    out->item_1 = val1;
    out->item_2 = val2;
    return out;
}

// returns a null terminated string containing shape of the tensor
char* str_shape(tensor* t){
    char* buffer = (char*)checkMallocErrors(malloc(sizeof(char) * 13));

    if (t->num_dims==2){
        sprintf(buffer, "(%i, %i)\0", t->shape[0], t->shape[1]);
    } else if (t->num_dims==3){
        sprintf(buffer, "(%i, %i, %i)\0", t->shape[0], t->shape[1], t->shape[2]);
    } else if (t->num_dims==4){
        sprintf(buffer, "(%i, %i, %i, %i)\0", t->shape[0], t->shape[1], t->shape[2], t->shape[3]);
    } else{
        printf("[str_shape] unexpected shape\n");
        exit(1);
    }
    return buffer;
}


void* checkMallocErrors(void* ptr) {
    if (ptr == NULL){
        printf("[malloc] error: null pointer\n");
        exit(1);
    }
    return ptr;
}

void maybe_init_grad(tensor* t){
    if (!t->grad){
        t->grad = TensorLikeFill(t, 0.0);
    } else {
        if (UTILS_DEBUG) printf("[maybe_init_grad] %s->grad exists!\n", t->name);
    }
}

char* random_chars(int num){
    // no need to increment for the null terminator, bc the for loop below is not inclusive of the last char;
    //  e.g.: num=3; the loop below will iterate 0-2; s[3] = '\0'
    // not necessary to do sizeof(char) bc guarantied to be 1
    char* s = (char*)checkMallocErrors(malloc(sizeof(char) * num));

    char offset = 'a';
    for (int i=0; i<num; i++){
        // my first thought was to use modulus, but it's wrong https://stackoverflow.com/a/6852396
        // todo: still see non-printable chars in the tensor names
        char sampled = rand() % 26; // 'z' - 'a' // 26 letters
        s[i] = offset + sampled;
    }
    s[num] = '\0';
    return s;
}

void set_name(tensor* t, const char* name){
    // free the automatically set random name
    // added by the constructor
    // todo: "if (t->name){...}" for some reason leads to a double free error, although I'd expect the two syntaxes "if (t->name==NULL){...}" be the same in my case
    // if (t->name==NULL){
    //     free(t->name);
    // }

    // todo-low: small inefficiency of always allocating MAX_TENSOR_NAME
    //  even if user provided str is shorter
    t->name = (char*)checkMallocErrors(malloc(sizeof(char) * MAX_TENSOR_NAME));

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
    char* all_visited[MAX_NODES]; // (float*)checkMallocErrors(malloc(sizeof(float*) * MAX_NODES));
    // is used to index into all_visited
    int idx_visited = 0;

    std::deque <tensor*> ready;
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
                // question-now: use of == on char arrays -- it should be CPP's overloaded?
                if (all_visited[i] == inp->name) {
                    is_visited = true;
                    break;
                }
            }

            // an input -> op
            fprintf(f, "%s -> %s_%s\n", inp->name, op_name, t->name);

            // for tensors, vis shapes instead of names
            // label=\"{%s\\nshape=(%i, %i)}\"]\n", inp->name
            fprintf(f, "%s [shape=record, label=\"{%s\\nshape=%s}\"]\n", inp->name, inp->name, str_shape(inp));

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
