#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <stdio.h>
#include <stdlib.h>

// A square n-by-n matrix
typedef struct {
    int n;
    double **data;
} Matrix;

Matrix* allocate_matrix(int n);
void free_matrix(Matrix *matrix);
Matrix* read_matrix(const char *filename);
int write_matrix(const char *filename, const Matrix *matrix);
double compute_mean(const Matrix *matrix);
double compute_variance(const Matrix *matrix);
Matrix* multiply_matrices(const Matrix *A, const Matrix *B);
void add_scalar_to_diagonal(Matrix *matrix, double scalar);
void add_scalar_to_all_elements(Matrix *matrix, double scalar);

#endif // MATRIX_UTILS_H
