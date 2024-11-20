#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include "matrix_utils.h"

void print_help() {
    printf("Computes and saves A = B * C^2 + M(C) * I + I + D(B) * E\n"
        "Usage: ./main <B_matrix_file> <C_matrix_file> <output_A_matrix_file> <num_threads>\n"
        "Matrix file for N-by-N matrix should be in the following format:\n"
        "First line: N\n"
        "Next N lines: each line contains N space-separated double values\n");
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        print_help();
        return EXIT_FAILURE;
    }

    const char *B_filename = argv[1];
    const char *C_filename = argv[2];
    const char *A_filename = argv[3];
    int num_threads = atoi(argv[4]);

    Matrix *B = read_matrix(B_filename);
    Matrix *C = read_matrix(C_filename);
    int n = B->n;

    if (C->n != n) {
        fprintf(stderr, "Matrix sizes do not match.\n");
        exit(EXIT_FAILURE);
    }

    omp_set_num_threads(num_threads);

    double start_time = omp_get_wtime();

    double M_C = compute_mean(C);
    double D_B = compute_variance(B);

    Matrix *C_squared = multiply_matrices(C, C);
    Matrix *result = multiply_matrices(B, C_squared);

    add_scalar_to_diagonal(result, M_C + 1.0);
    add_scalar_to_all_elements(result, D_B);

    double end_time = omp_get_wtime();
    double computation_time = end_time - start_time;
    printf("Computation time: %f seconds\n", computation_time);

    write_matrix(A_filename, result);

    free_matrix(B);
    free_matrix(C);
    free_matrix(C_squared);
    free_matrix(result);

    return EXIT_SUCCESS;
}
