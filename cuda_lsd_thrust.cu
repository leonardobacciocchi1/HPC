// LSD Radix Sort (Standard Industriale) usando Thrust
// Thrust seleziona automaticamente LSD Radix Sort per tipi interi

#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h> // Per puntatori raw
#include <cstdio>
#include <cstdint>
#include <cstring>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)


//Ordina un array di uint32_t sulla GPU usando Thrust.

//thrust::sort chiama un'implementazione LSD Radix Sort altamente ottimizzata (spesso CUB in background).


void thrust_radix_sort(uint32_t *d_keys, size_t n) {
    // Crea un device_ptr di Thrust dal puntatore raw
    thrust::device_ptr<uint32_t> d_ptr(d_keys);

    // Chiama l'ordinamento e gestisce internamente la memoria temporanea.
    thrust::sort(d_ptr, d_ptr + n);
}

bool is_sorted(const uint32_t *data, size_t n) {
    for (size_t i = 1; i < n; i++) {
        if (data[i-1] > data[i]) return false;
    }
    return true;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Uso: %s <input.bin> [--repeats=N]\n", argv[0]);
        return 1;
    }

    const char *input_file = argv[1];
    int repeats = 1;

    // Parse arguments
    for (int i = 2; i < argc; i++) {
        if (strncmp(argv[i], "--repeats=", 10) == 0) {
            repeats = atoi(argv[i] + 10);
        }
    }
    if (repeats < 1) repeats = 1;

    FILE *f = fopen(input_file, "rb");
    if (!f) {
        perror("fopen");
        fprintf(stderr, "Impossibile aprire %s\n", input_file);
        return 1;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size % 4 != 0) {
        fprintf(stderr, "File size non multiplo di 4\n");
        fclose(f);
        return 1;
    }

    size_t size = file_size / 4;
    printf("=== LSD Radix Sort (Thrust) ===\n"); // Modificato
    printf("Input: %s\n", input_file);
    printf("Size: %zu elements (%.2f MB)\n", size, size * 4.0 / 1024 / 1024);
    printf("Repeats: %d\n\n", repeats);

    uint32_t *h_data = new uint32_t[size];
    uint32_t *h_backup = new uint32_t[size];
    size_t read_count = fread(h_data, sizeof(uint32_t), size, f);
    fclose(f);

    if (read_count != size) {
        fprintf(stderr, "Errore lettura file\n");
        delete[] h_data;
        delete[] h_backup;
        return 1;
    }

    memcpy(h_backup, h_data, size * sizeof(uint32_t));
    uint32_t *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(uint32_t)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double sum = 0.0, tmin = 1e300, tmax = 0.0;
    bool all_sorted = true;

    for (int r = 0; r < repeats; r++) {
        memcpy(h_data, h_backup, size * sizeof(uint32_t));
        CUDA_CHECK(cudaMemcpy(d_data, h_data, size * sizeof(uint32_t), cudaMemcpyHostToDevice));

        // Sort (CHIAMATA A THRUST)
        CUDA_CHECK(cudaEventRecord(start));
        thrust_radix_sort(d_data, size); // Sostituito
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaDeviceSynchronize());

        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        double elapsed_sec = elapsed_ms / 1000.0;

        sum += elapsed_sec;
        if (elapsed_sec < tmin) tmin = elapsed_sec;
        if (elapsed_sec > tmax) tmax = elapsed_sec;

        printf("Run %d: %.6f sec (%.2f M keys/sec)\n",
               r + 1, elapsed_sec, (size / 1e6) / elapsed_sec);

        if (r == repeats - 1) {
            CUDA_CHECK(cudaMemcpy(h_data, d_data, size * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            all_sorted = is_sorted(h_data, size);
        }
    }

    double avg = sum / repeats;
    printf("\nAverage: %.6f sec (%.2f M keys/sec)\n", avg, (size / 1e6) / avg);
    printf("Min:     %.6f sec\n", tmin);
    printf("Max:     %.6f sec\n", tmax);
    printf("Sorted:  %s\n", all_sorted ? "YES" : "NO");

    delete[] h_data;
    delete[] h_backup;
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return all_sorted ? 0 : 1;
}