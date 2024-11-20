#include "matrix_utils.h"
#include <math.h>
#include <omp.h>

Matrix* allocate_matrix(int n) {
    Matrix *matrix = (Matrix*)malloc(sizeof(Matrix));
    matrix->n = n;
    matrix->data = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; ++i) {
        matrix->data[i] = (double*)malloc(n * sizeof(double));
    }
    return matrix;
}

void free_matrix(Matrix *matrix) {
    for (int i = 0; i < matrix->n; ++i) {
        free(matrix->data[i]);
    }
    free(matrix->data);
    free(matrix);
}

Matrix* read_matrix(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Failed to open matrix file");
        exit(EXIT_FAILURE);
    }
    int size;
    if (fscanf(fp, "%d", &size) == EOF) {
        fprintf(stderr, "Failed to read matrix size");
        exit(EXIT_FAILURE);
    }
    Matrix *matrix = allocate_matrix(size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (fscanf(fp, "%lf", &matrix->data[i][j]) == EOF) {
                fprintf(stderr, "Failed to read matrix element at position (%d, %d)", i, j);
                exit(EXIT_FAILURE);
            }
        }
    }
    fclose(fp);
    return matrix;
}

int write_matrix(const char *filename, const Matrix *matrix) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Failed to write matrix file");
        return -1;
    }
    fprintf(fp, "%d\n", matrix->n);
    for (int i = 0; i < matrix->n; ++i) {
        for (int j = 0; j < matrix->n; ++j) {
            fprintf(fp, "%.6f ", matrix->data[i][j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    return 0;
}

double compute_mean(const Matrix *matrix) {
    double sum = 0.0;
    int n = matrix->n;
    int total_elements = n * n;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            sum += matrix->data[i][j];
        }
    }
    return sum / total_elements;
}

double compute_variance(const Matrix *matrix) {
    double mean = compute_mean(matrix);
    double variance = 0.0;
    int n = matrix->n;
    int total_elements = n * n;
    #pragma omp parallel for reduction(+:variance)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double diff = matrix->data[i][j] - mean;
            variance += diff * diff;
        }
    }
    return variance / total_elements;
}

Matrix* multiply_matrices(const Matrix *A, const Matrix *B) {
    int n = A->n;
    Matrix *result = allocate_matrix(n);

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A->data[i][k] * B->data[k][j];
            }
            result->data[i][j] = sum;
        }
    }

    return result;
}

void add_scalar_to_diagonal(Matrix *matrix, double scalar) {
    int n = matrix->n;
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        matrix->data[i][i] += scalar;
    }
}

void add_scalar_to_all_elements(Matrix *matrix, double scalar) {
    int n = matrix->n;
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix->data[i][j] += scalar;
        }
    }
}
