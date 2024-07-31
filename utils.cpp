// #include <iostream> // todo: use C only
// using namespace std;

#include "nn.h"

void _print(tensor* t, const char* msg)
{
    printf("\n%s: ", msg);

    for (int i=0, row_len=t->shape[1]; i<t->size; i++) {
        if (i % row_len == 0) cout << endl;
        // easy numpy export:
        // if (i % row_len == 0) cout << "], " << endl << "[";

        // todo: use right justified, and print only 4 points of precision
        // cout << " " << setw(5) << right << t->data[i] << ", ";

        // %6.1f describes number at least six characters wide, with 1 digit after the decimal point
        printf("%8.4f, ", t->data[i]);
    }
    cout << endl;
}

char* random_chars(int num){
    // increment for the null terminator;
    // not necessary to do sizeof(char) bc guarantied to be 1
    char* s = (char*)malloc(sizeof(char) * ++num);

    char offset = 'a';
    for (int i=0; i<num-1; i++){
        // my first thought was to use modulus, but it's wrong https://stackoverflow.com/a/6852396
        char sampled = rand() % 26; // 'z' - 'a' // 26 letters
        s[i] = offset + sampled;
    }
    s[num] = '\0';
    return s;
}

// todo-low: keras like vis https://graphviz.org/Gallery/directed/neural-network.html
void graphviz(tensor* tens){
    FILE *f = fopen("./generated/graph.txt", "w");
    if (f == NULL) {
        printf("[graphviz] Error opening file\n");
        return;
    }

    fprintf(f, "digraph {\n");
    fprintf(f, "node [ordering=\"in\", fixedsize=shape shape=circle style=filled]\n");

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

            // an input -> op
            fprintf(f, "%s -> %s_%s\n", inp->name, op_name, t->name);

            // for tensors, vis shapes instead of names
            fprintf(f, "%s [shape=record, label=\"{shape=(%i, %i )}\"]\n", inp->name, inp->shape[0], inp->shape[1]);

            // leafs don't have inputs to iterate over in the next iteration
            if (!inp->is_leaf)
                ready.push_front(inp);
        }
    }
    fprintf(f, "}\n");
    fclose(f);
}
