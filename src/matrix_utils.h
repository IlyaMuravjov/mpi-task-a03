#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int nrows;
    int ncols;
    double *data;
} Matrix;

static inline double matrix_get(
    const Matrix *matrix, int row, int col
) {
    return matrix->data[row * matrix->ncols + col];
}

static inline void matrix_set(
    Matrix *matrix, int row, int col, double value
) {
    matrix->data[row * matrix->ncols + col] = value;
}

Matrix* allocate_matrix(int nrows, int ncols);
void free_matrix(Matrix *matrix);
Matrix* read_square_matrix(const char *filename);
int write_square_matrix(const char *filename, const Matrix *matrix);
double compute_mean(const Matrix *matrix);
double compute_variance(const Matrix *matrix);
Matrix* transpose_matrix(const Matrix *matrix);

Matrix* multiply_by_transposed(
    const Matrix *A, const Matrix *B_transposed
);

void add_scalar_to_diagonal(Matrix *matrix, double scalar);
void add_scalar_to_all_elements(Matrix *matrix, double scalar);

Matrix* scatter_rows(Matrix *matrix, MPI_Comm comm, int root);

Matrix* gather_rows(
    Matrix *local_matrix, MPI_Comm comm, int root, int global_nrows
);

#endif // MATRIX_UTILS_H
