#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix_utils.h"

void print_help() {
    printf("Saves random N-by-N matrix\n"
           "Usage: ./save_random_matrix <output_matrix_file> <N> "
           "<min_value> <max_value> <seed>\n");
}

int main(int argc, char *argv[]) {
    if (argc != 6) {
        print_help();
        return EXIT_FAILURE;
    }

    const char *output_filename = argv[1];
    int n = atoi(argv[2]);
    double min_value = atof(argv[3]);
    double max_value = atof(argv[4]);
    int seed = atoi(argv[5]);

    srand(seed);

    Matrix *matrix = allocate_matrix(n, n);

    for (int i = 0; i < n * n; ++i) {
        double r = ((double)rand() / RAND_MAX);
        matrix->data[i] = min_value 
            + r * (max_value - min_value);
    }

    write_square_matrix(output_filename, matrix);

    free_matrix(matrix);

    return EXIT_SUCCESS;
}
