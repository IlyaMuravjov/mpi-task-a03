#include "matrix_utils.h"
#include <assert.h>
#include <mpi.h>

Matrix* allocate_matrix(int nrows, int ncols) {
    Matrix *matrix = (Matrix*)malloc(sizeof(Matrix));
    matrix->nrows = nrows;
    matrix->ncols = ncols;
    matrix->data = (double*) malloc(nrows * ncols * sizeof(double));
    return matrix;
}

void free_matrix(Matrix *matrix) {
    if (matrix) {
        free(matrix->data);
        free(matrix);
    }
}

Matrix* read_square_matrix(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Failed to open matrix file");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int size;
    if (fscanf(fp, "%d", &size) == EOF) {
        fprintf(stderr, "Failed to read matrix size");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    Matrix *matrix = allocate_matrix(size, size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            double value;
            if (fscanf(fp, "%lf", &value) == EOF) {
                fprintf(stderr, "Failed to read matrix element at pos (%d, %d)", i, j);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            matrix_set(matrix, i, j, value);
        }
    }
    fclose(fp);
    return matrix;
}

int write_square_matrix(const char *filename, const Matrix *matrix) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Failed to write matrix file");
        return -1;
    }
    fprintf(fp, "%d\n", matrix->nrows);
    for (int i = 0; i < matrix->nrows; ++i) {
        for (int j = 0; j < matrix->ncols; ++j) {
            fprintf(fp, "%.6f ", matrix_get(matrix, i, j));
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    return 0;
}

double compute_mean(const Matrix *matrix) {
    double sum = 0.0;
    int total_elements = matrix->nrows * matrix->ncols;
    for (int i = 0; i < total_elements; i++) {
        sum += matrix->data[i];
    }
    return sum / total_elements;
}

double compute_variance(const Matrix *matrix) {
    double mean = compute_mean(matrix);
    double variance = 0.0;
    int total_elements = matrix->nrows * matrix->ncols;
    for (int i = 0; i < total_elements; i++) {
        double diff = matrix->data[i] - mean;
        variance += diff * diff;
    }
    return variance / total_elements;
}

Matrix* transpose_matrix(const Matrix *matrix) {
    Matrix *B_T = allocate_matrix(matrix->ncols, matrix->nrows);
    for (int i = 0; i < matrix->nrows; ++i) {
        for (int j = 0; j < matrix->ncols; ++j) {
            matrix_set(B_T, j, i, matrix_get(matrix, i, j));
        }
    }
    return B_T;
}

Matrix* multiply_by_transposed(const Matrix *A, const Matrix *B_transposed) {
    assert(A->ncols == B_transposed->ncols);
    Matrix *result = allocate_matrix(A->nrows, B_transposed->nrows);

    for (int i = 0; i < A->nrows; i++) {
        for (int j = 0; j < B_transposed->nrows; j++) {
            double sum = 0.0;
            for (int k = 0; k < A->ncols; k++) {
                sum += matrix_get(A, i, k) * matrix_get(B_transposed, j, k);
            }
            matrix_set(result, i, j, sum);
        }
    }
    return result;
}

void add_scalar_to_diagonal(Matrix *matrix, double scalar) {
    for (int i = 0; i < matrix->nrows; ++i) {
        double value = matrix_get(matrix, i, i);
        matrix_set(matrix, i, i, value + scalar);
    }
}

void add_scalar_to_all_elements(Matrix *matrix, double scalar) {
    int total = matrix->nrows * matrix->ncols;
    for (int i = 0; i < total; i++) {
        matrix->data[i] += scalar;
    }
}

void calculate_counts_and_displacements(int global_nrows, int ncols, int size, int **counts, int **displs) {
    *counts = (int *) malloc(size * sizeof(int));
    *displs = (int *) malloc(size * sizeof(int));
    int base_rows = global_nrows / size;
    int remainder = global_nrows % size;
    int offset = 0;

    for (int i = 0; i < size; ++i) {
        (*counts)[i] = (i < remainder ? base_rows + 1 : base_rows) * ncols;
        (*displs)[i] = offset;
        offset += (*counts)[i];
    }
}

Matrix* scatter_rows(Matrix *matrix, MPI_Comm comm, int root) {
    int rank, size, global_nrows, ncols;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == root) {
        global_nrows = matrix->nrows;
        ncols = matrix->ncols;
    }
    MPI_Bcast(&global_nrows, 1, MPI_INT, root, comm);
    MPI_Bcast(&ncols, 1, MPI_INT, root, comm);

    int base_rows = global_nrows / size;
    int remainder = global_nrows % size;
    int local_nrows = (rank < remainder) ? base_rows + 1 : base_rows;

    Matrix *local_matrix = allocate_matrix(local_nrows, ncols);

    int *counts = NULL, *displs = NULL;
    if (rank == root) {
        calculate_counts_and_displacements(global_nrows, ncols, size, &counts, &displs);
    }

    MPI_Scatterv(
        /* sendbuf = */ (rank == root ? matrix->data : NULL),
        /* sendcounts = */ counts,
        /* displs = */ displs,
        /* sendtype = */ MPI_DOUBLE,
        /* recvbuf = */ local_matrix->data,
        /* recvcount = */ local_nrows * ncols,
        /* recvtype = */ MPI_DOUBLE,
        /* root = */ root,
        /* comm = */ comm
    );

    if (rank == root) {
        free(counts);
        free(displs);
    }

    return local_matrix;
}

Matrix* gather_rows(Matrix *local_matrix, MPI_Comm comm, int root, int global_nrows) {
    int rank, size, ncols;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    ncols = local_matrix->ncols;

    int *counts = NULL, *displs = NULL;
    if (rank == root) {
        calculate_counts_and_displacements(global_nrows, ncols, size, &counts, &displs);
    }

    Matrix *global_matrix = NULL;
    if (rank == root) {
        global_matrix = allocate_matrix(global_nrows, ncols);
    }

    MPI_Gatherv(
        /* sendbuf = */ local_matrix->data,
        /* sendcount = */ local_matrix->nrows * ncols,
        /* sendtype = */ MPI_DOUBLE,
        /* recvbuf = */ (rank == root ? global_matrix->data : NULL),
        /* recvcounts = */ counts,
        /* displs = */ displs,
        /* recvtype = */ MPI_DOUBLE,
        /* root = */ root,
        /* comm = */ comm
    );

    if (rank == root) {
        free(counts);
        free(displs);
    }

    return global_matrix;
}
