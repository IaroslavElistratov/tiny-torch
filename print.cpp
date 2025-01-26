#include "nn.h"


// %6.1f describes number at least six characters wide, with 1 digit after the decimal point
// printf("%12.8f, ", t->data[i]);

void sprint_2d(tensor* t){
    printf("\n%s:\n", t->name);
    printf("%s->shape: %i, %i\n", t->name, t->shape[0], t->shape[1]);
    printf("%s->strides: %i, %i\n", t->name, t->stride[0], t->stride[1]);
}

void sprint_3d(tensor* t){
    printf("\n%s:\n", t->name);
    printf("%s->shape: %i, %i, %i\n", t->name, t->shape[0], t->shape[1], t->shape[2]);
    printf("%s->strides: %i, %i, %i\n", t->name, t->stride[0], t->stride[1], t->stride[2]);
}

void sprint_4d(tensor* t){
    printf("\n%s:\n", t->name);
    printf("%s->shape: %i, %i, %i, %i\n", t->name, t->shape[0], t->shape[1],  t->shape[2],  t->shape[3]);
    printf("%s->strides: %i, %i, %i, %i\n", t->name, t->stride[0], t->stride[1],  t->stride[2],  t->stride[3]);
}

void sprint(tensor* t){
    if (t->num_dims==2) sprint_2d(t);
    else if (t->num_dims==3) sprint_3d(t);
    else if (t->num_dims==4) sprint_4d(t);
    else {
        printf("[sprint] Error");
        exit(1);
    }
}


void print_2d(tensor* t){
    tensor* t_copy = COPY_FROM_DEVICE(t);
    sprint_2d(t);

    int y = t->shape[0];
    int z = t->shape[1];

    for (int yi=0; yi<y; yi++){
        printf("[");
        for (int zi=0; zi<z; zi++){
            int idx = index(t_copy, yi, zi);
            printf("%12.8f, ", t_copy->data[idx]);
        }
        printf("],\n");
    }
    printf("\n");
}

void print_3d(tensor* t){
    tensor* t_copy = COPY_FROM_DEVICE(t);
    sprint_3d(t);

    int x = t->shape[0];
    int y = t->shape[1];
    int z = t->shape[2];

    for (int xi=0; xi<x; xi++){
        for (int yi=0; yi<y; yi++){
            printf("[");
            for (int zi=0; zi<z; zi++){
                int idx = index(t_copy, xi, yi, zi);
                printf("%12.8f, ", t_copy->data[idx]);
            }
            printf("],\n");
        }
        printf("\n");
    }
    printf("\n");
}

void print_4d(tensor* t){
    tensor* t_copy = COPY_FROM_DEVICE(t);
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
                    int idx = index(t_copy, oi, xi, yi, zi);
                    printf("%12.8f, ", t_copy->data[idx]);
                }
                printf("],\n");
            }
            printf("\n");
        }
        printf("\n");
    }
}

void print(tensor* t){
    if (t->num_dims==2) print_2d(t);
    else if (t->num_dims==3) print_3d(t);
    else if (t->num_dims==4) print_4d(t);
    else {
        printf("[print] Error");
        exit(1);
    }
}


// todo-high: de duplicate lprint and lsprint:

void lsprint_2d(tensor* t){
    FILE *f = fopen("./generated/log.txt", "a");
    if (!f) {
        printf("Error opening file\n");
        exit(1);
    }
    fprintf(f, "\n%s:\n", t->name);
    fprintf(f, "%s->shape: %i, %i\n", t->name, t->shape[0], t->shape[1]);
    fprintf(f, "%s->strides: %i, %i\n", t->name, t->stride[0], t->stride[1]);
    fclose(f);
}

void lsprint_3d(tensor* t){
    FILE *f = fopen("./generated/log.txt", "a");
    if (!f) {
        printf("Error opening file\n");
        exit(1);
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
        exit(1);
    }
    fprintf(f, "\n%s:\n", t->name);
    fprintf(f, "%s->shape: %i, %i, %i, %i\n", t->name, t->shape[0], t->shape[1],  t->shape[2],  t->shape[3]);
    fprintf(f, "%s->strides: %i, %i, %i, %i\n", t->name, t->stride[0], t->stride[1],  t->stride[2],  t->stride[3]);
    fclose(f);
}

void lsprint(tensor* t){
    if (t->num_dims==2) lsprint_2d(t);
    else if (t->num_dims==3) lsprint_3d(t);
    else if (t->num_dims==4) lsprint_4d(t);
    else {
        printf("[lsprint] Error");
        exit(1);
    }
}


void lprint_2d(tensor* t){
    tensor* t_copy = COPY_FROM_DEVICE(t);
    lsprint_2d(t);

    FILE *f = fopen("./generated/log.txt", "a");
    if (!f) {
        printf("Error opening file\n");
        exit(1);
    }

    int y = t->shape[0];
    int z = t->shape[1];

    for (int yi=0; yi<y; yi++){
        fprintf(f, "    [");
        for (int zi=0; zi<z; zi++){
            int idx = index(t_copy, yi, zi);
            fprintf(f, "%12.8f, ", t_copy->data[idx]);
        }
        fprintf(f, "],\n");
    }
    fprintf(f, "\n");
    fclose(f);
}

void lprint_3d(tensor* t){
    tensor* t_copy = COPY_FROM_DEVICE(t);
    lsprint_3d(t);

    FILE *f = fopen("./generated/log.txt", "a");
    if (!f) {
        printf("Error opening file\n");
        exit(1);
    }

    int x = t->shape[0];
    int y = t->shape[1];
    int z = t->shape[2];

    for (int xi=0; xi<x; xi++){
        for (int yi=0; yi<y; yi++){
            fprintf(f, "    [");
            for (int zi=0; zi<z; zi++){
                int idx = index(t_copy, xi, yi, zi);
                fprintf(f, "%12.8f, ", t_copy->data[idx]);
            }
            fprintf(f, "],\n");
        }
        fprintf(f, "\n");
    }
    fprintf(f, "\n");
    fclose(f);
}

void lprint_4d(tensor* t){
    tensor* t_copy = COPY_FROM_DEVICE(t);
    lsprint_4d(t);

    FILE *f = fopen("./generated/log.txt", "a");
    if (!f) {
        printf("Error opening file\n");
        exit(1);
    }

    int o = t->shape[0];
    int x = t->shape[1];
    int y = t->shape[2];
    int z = t->shape[3];

    for (int oi=0; oi<o; oi++){
        for (int xi=0; xi<x; xi++){
            for (int yi=0; yi<y; yi++){
                fprintf(f, "    [");
                for (int zi=0; zi<z; zi++){
                    int idx = index(t_copy, oi, xi, yi, zi);
                    fprintf(f, "%12.8f, ", t_copy->data[idx]);
                }
                fprintf(f, "],\n");
            }
            fprintf(f, "\n");
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

void lprint(tensor* t){
    if (t->num_dims==2) lprint_2d(t);
    else if (t->num_dims==3) lprint_3d(t);
    else if (t->num_dims==4) lprint_4d(t);
    else {
        printf("[lprint] Error");
        exit(1);
    }
}
