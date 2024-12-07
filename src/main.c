#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include "matrix_utils.h"

void print_help() {
    printf("Computes and saves A = B * C^2 + M(C) * I + I + D(B) * E\n"
        "Usage: ./main <B_matrix_file> <C_matrix_file> <output_A_matrix_file>\n"
        "Matrix file for N-by-N matrix should be in the following format:\n"
        "First line: N\n"
        "Next N lines: each line contains N space-separated double values\n");
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4) {
        if (rank == 0) print_help();
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const char *B_filename = argv[1];
    const char *C_filename = argv[2];
    const char *A_filename = argv[3];
    
    Matrix *B = NULL;
    Matrix *C = NULL;
    Matrix *C_transposed = NULL;
    double M_C = 0.0;
    double D_B = 0.0;
    int n = 0;

    if (rank == 0) {
        B = read_square_matrix(B_filename);
        C = read_square_matrix(C_filename);

        if (B->nrows != C->nrows) {
            fprintf(stderr, "Matrix sizes do not match.\n");
            free_matrix(B);
            free_matrix(C);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    double start_time = 0.0;
    double t0 = 0.0;
    if (rank == 0) {
        start_time = MPI_Wtime();
        
        C_transposed = transpose_matrix(C);
        n = B->nrows;
        M_C = compute_mean(C);
        D_B = compute_variance(B);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        C_transposed = allocate_matrix(n, n);
    }

    MPI_Bcast(C_transposed->data, n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    Matrix *B_local = scatter_rows(B, MPI_COMM_WORLD, 0);
    Matrix *BC_local = multiply_by_transposed(B_local, C_transposed);
    Matrix *BCC_local = multiply_by_transposed(BC_local, C_transposed);
    Matrix *result = gather_rows(BCC_local, MPI_COMM_WORLD, 0, n);

    if (rank == 0) {
        add_scalar_to_diagonal(result, M_C + 1.0);
        add_scalar_to_all_elements(result, D_B);

        double end_time = MPI_Wtime();
        double computation_time = end_time - start_time;
        printf("Computation time: %f seconds\n", computation_time);

        write_square_matrix(A_filename, result);
    }

    free_matrix(B);
    free_matrix(C);
    free_matrix(B_local);
    free_matrix(BC_local);
    free_matrix(BCC_local);
    free_matrix(result);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
