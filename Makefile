CC = mpicc
CFLAGS = -O3

all: main save_random_matrix

main: src/main.c src/matrix_utils.c
	$(CC) $(CFLAGS) -o main src/main.c src/matrix_utils.c

save_random_matrix: src/save_random_matrix.c src/matrix_utils.c
	$(CC) $(CFLAGS) -o save_random_matrix src/save_random_matrix.c src/matrix_utils.c

clean:
	rm -f main save_random_matrix