#include "nn.h"

void _print(tensor* t)
{
    printf("\n%s: ", t->name);

    for (int i=0, row_len=t->shape[1]; i<t->size; i++) {
        if (i % row_len == 0) cout << endl;
        // %6.1f describes number at least six characters wide, with 1 digit after the decimal point
        printf("%8.4f, ", t->data[i]);
    }
    printf("\n");
}


void sprint_2d(tensor* t){
    printf("\n%s:\n", t->name);
    printf("%s->shape: %i, %i\n", t->name, t->shape[0], t->shape[1]);
    printf("%s->strides: %i, %i\n", t->name, t->stride[0], t->stride[1]);
}

void sprint_3d(tensor* t)
{
    printf("\n%s:\n", t->name);
    printf("%s->shape: %i, %i, %i\n", t->name, t->shape[0], t->shape[1], t->shape[2]);
    printf("%s->strides: %i, %i, %i\n", t->name, t->stride[0], t->stride[1], t->stride[2]);
}


void sprint_4d(tensor* t){
    printf("\n%s:\n", t->name);
    printf("%s->shape: %i, %i, %i, %i\n", t->name, t->shape[0], t->shape[1],  t->shape[2],  t->shape[3]);
    printf("%s->strides: %i, %i, %i, %i\n", t->name, t->stride[0], t->stride[1],  t->stride[2],  t->stride[3]);
}


void print_2d(tensor* t)
{
    sprint_2d(t);

    int y = t->shape[0];
    int z = t->shape[1];

    for (int yi=0; yi<y; yi++){
        printf("[");
        for (int zi=0; zi<z; zi++){
            int idx = index_2d(t, yi, zi);
            printf("%8.4f, ", t->data[idx]);
        }
        printf("],\n");
    }
    printf("\n");
}

void print_3d(tensor* t)
{
    sprint_3d(t);

    int x = t->shape[0];
    int y = t->shape[1];
    int z = t->shape[2];

    for (int xi=0; xi<x; xi++){
        for (int yi=0; yi<y; yi++){
            printf("[");
            for (int zi=0; zi<z; zi++){
                int idx = index_3d(t, xi, yi, zi);
                printf("%8.4f, ", t->data[idx]);
            }
            printf("],\n");
        }
        printf("\n");
    }
    printf("\n");
}

void print_4d(tensor* t)
{
    sprint_4d(t);

    int o = t->shape[0];
    int x = t->shape[1];
    int y = t->shape[2];
    int z = t->shape[3];

    for (int oi=0; oi<o; oi++){
        for (int xi=0; xi<x; xi++){
            for (int yi=0; yi<y; yi++){
                printf("[");
                for (int zi=0; zi<z; zi++){
                    int idx = index_4d(t, oi, xi, yi, zi);
                    printf("%8.4f, ", (float)t->data[idx]);
                }
                printf("],\n");
            }
            printf("\n");
        }
        printf("\n");
    }
}


// todo-high: de duplicate lprint and lsprint:

void lsprint_2d(tensor* t){
    FILE *f = fopen("./generated/log.txt", "a");
    if (!f) {
        printf("Error opening file\n");
        return;
    }
    fprintf(f, "\n%s:\n", t->name);
    fprintf(f, "%s->shape: %i, %i\n", t->name, t->shape[0], t->shape[1]);
    fprintf(f, "%s->strides: %i, %i\n", t->name, t->stride[0], t->stride[1]);
    fclose(f);
}

void lsprint_3d(tensor* t)
{
    FILE *f = fopen("./generated/log.txt", "a");
    if (!f) {
        printf("Error opening file\n");
        return;
    }
    fprintf(f, "\n%s:\n", t->name);
    fprintf(f, "%s->shape: %i, %i, %i\n", t->name, t->shape[0], t->shape[1], t->shape[2]);
    fprintf(f, "%s->strides: %i, %i, %i\n", t->name, t->stride[0], t->stride[1], t->stride[2]);
    fclose(f);

}

void lsprint_4d(tensor* t){
    FILE *f = fopen("./generated/log.txt", "a");
    if (!f) {
        printf("Error opening file\n");
        return;
    }
    fprintf(f, "\n%s:\n", t->name);
    fprintf(f, "%s->shape: %i, %i, %i, %i\n", t->name, t->shape[0], t->shape[1],  t->shape[2],  t->shape[3]);
    fprintf(f, "%s->strides: %i, %i, %i, %i\n", t->name, t->stride[0], t->stride[1],  t->stride[2],  t->stride[3]);
    fclose(f);
}

void lprint_2d(tensor* t)
{

    lsprint_2d(t);

    FILE *f = fopen("./generated/log.txt", "a");
    if (!f) {
        printf("Error opening file\n");
        return;
    }

    int y = t->shape[0];
    int z = t->shape[1];

    for (int yi=0; yi<y; yi++){
        fprintf(f, "[");
        for (int zi=0; zi<z; zi++){
            int idx = index_2d(t, yi, zi);
            fprintf(f, "%8.4f, ", t->data[idx]);
        }
        fprintf(f, "],\n");
    }
    fprintf(f, "\n");
    fclose(f);
}

void lprint_3d(tensor* t)
{
    lsprint_3d(t);

    FILE *f = fopen("./generated/log.txt", "a");
    if (!f) {
        printf("Error opening file\n");
        return;
    }

    int x = t->shape[0];
    int y = t->shape[1];
    int z = t->shape[2];

    for (int xi=0; xi<x; xi++){
        for (int yi=0; yi<y; yi++){
            fprintf(f, "[");
            for (int zi=0; zi<z; zi++){
                int idx = index_3d(t, xi, yi, zi);
                fprintf(f, "%8.4f, ", t->data[idx]);
            }
            fprintf(f, "],\n");
        }
        fprintf(f, "\n");
    }
    fprintf(f, "\n");
    fclose(f);
}

void lprint_4d(tensor* t)
{
    lsprint_4d(t);

    FILE *f = fopen("./generated/log.txt", "a");
    if (!f) {
        printf("Error opening file\n");
        return;
    }

    int o = t->shape[0];
    int x = t->shape[1];
    int y = t->shape[2];
    int z = t->shape[3];

    for (int oi=0; oi<o; oi++){
        for (int xi=0; xi<x; xi++){
            for (int yi=0; yi<y; yi++){
                fprintf(f, "[");
                for (int zi=0; zi<z; zi++){
                    int idx = index_4d(t, oi, xi, yi, zi);
                    fprintf(f, "%8.4f, ", t->data[idx]);
                }
                fprintf(f, "],\n");
            }
            fprintf(f, "\n");
        }
        fprintf(f, "\n");
    }
    fclose(f);
}
