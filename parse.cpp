#include <stdlib.h> // atoi
#include <stdio.h> // sprintf

#define MAX_DIGIT_IDX 4
#define MAX_STR_LEN 20

char buff[MAX_DIGIT_IDX+1];


int _parse_idx(const char* str, int start_idx){

    int buff_idx = 0;
    int str_idx = start_idx;

    // flush buffer, otherwise potential problem that contents of that
    // global do not get cleared for each call to parse_idx, so potentially
    // it can start overwriting existing contents but not scribble the entire
    // string, but have partial contents from the previous run -- then feed to
    // atoi, it will result in the str contents combined
    for (int i = 0; i < MAX_DIGIT_IDX; i++)
        buff[i] = 0;

    char c;
    while ((c = str[str_idx++])){  //  != '\0'

        // err check
        if (buff_idx >= MAX_DIGIT_IDX || start_idx >= MAX_STR_LEN) {
            printf("\n[_parse_idx] error, idx has more digits than expected\n");
            exit(1);
        }

        // these are the two possible separators which delimit the digits
        if (c != ':' && c != ',') // && c != ' ' && c != '\n'
            buff[buff_idx++] = c;
        else
            return str_idx;
    }
    // no more digits to parse
    return 0;
}


// todo:
//  support ":"
//  support ":n" and "n:"
//  support omitting at the both ends
//      auto-filling missing dims -- allows "kernel[0]" instead of below
//      curr_filter = slice_4d(kernel, "f, 0:C, 0:HH, 0:WW"); // (F, C, HH, WW) -> (C, HH, WW)

int* parse_idxs(const char* dims, int num_pairs){

    // todo: use 2d tensors instead of "+ num_pairs"?
    //     Example from the book with 2d tensor -- converting day of the month
    //     int* p[2] = (int*)malloc(sizeof(int) * num_pairs * 2);
    //     int* starts = p[0];
    //     int* ends = p[1]; // pointer arithmetic

    int* p = (int*)malloc(sizeof(int) * num_pairs * 2);
    int* starts = p;
    int* ends = p + num_pairs; // pointer arithmetic

    // // also converts char to int
    // int starts[num_pairs];
    // int ends[num_pairs];

    int start_idx = 0;
    for (int i=0; i<num_pairs; i++){
        start_idx = _parse_idx(dims, start_idx);
        starts[i] = atoi(buff);
        start_idx = _parse_idx(dims, start_idx);
        ends[i] = atoi(buff);

        // printf("\n[parse_idxs] starts[%i] = %i", i, starts[i]);
        // printf("\n[parse_idxs] ends[%i] = %i\n", i, ends[i]);
    }

    return p;
}

