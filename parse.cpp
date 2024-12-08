#include <stdlib.h> // atoi
#include <stdio.h> // sprintf

#define MAX_DIGIT_IDX 4
#define MAX_STR_LEN 20

char buff[MAX_DIGIT_IDX+1];


int _parse_idx(const char* str, int start_idx){

    int buff_idx = 0;
    int str_idx = start_idx;

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


int* parse_idxs(const char* dims, int num_pairs){

    int* p = (int*)malloc(sizeof(int) * num_pairs * 2);
    int* starts = p;
    int* ends = p + num_pairs; // pointer arithmetic

    int start_idx = 0;
    for (int i=0; i<num_pairs; i++){
        start_idx = _parse_idx(dims, start_idx);
        starts[i] = atoi(buff);
        start_idx = _parse_idx(dims, start_idx);
        ends[i] = atoi(buff);
    }

    return p;
}

