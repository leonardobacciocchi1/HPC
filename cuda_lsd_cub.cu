// LSD Radix Sort (CUB) 
// CUB implementa LSD radix sort ottimizzato con chained scan+scatter

#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_type.cuh> 
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

// ============================================================================
// FUNZIONE MODIFICATA

void cub_radix_sort(uint32_t *d_keys, size_t n) {
    uint32_t *d_keys_out;
    CUDA_CHECK(cudaMalloc(&d_keys_out, n * sizeof(uint32_t)));

    // 2. Definisci i buffer di ping pong per l'ordinamento in-place
    cub::DoubleBuffer<uint32_t> d_keys_buffers(d_keys, d_keys_out);

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Determine temporary storage size
    // Passa d_keys_buffers
    cub::DeviceRadixSort::SortKeys(
        d_temp_storage, temp_storage_bytes,
        d_keys_buffers, n);

    // Allocate temporary storage
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // Execute sort
    // Passa d_keys_buffers
    cub::DeviceRadixSort::SortKeys(
        d_temp_storage, temp_storage_bytes,
        d_keys_buffers, n);

    
    // CUDA_CHECK(cudaMemcpy(d_keys, d_keys_out, n * sizeof(uint32_t), cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaFree(d_temp_storage));
    CUDA_CHECK(cudaFree(d_keys_out));
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
    printf("=== LSD Radix Sort (CUB - In-Place) ===\n"); // Titolo aggiornato
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

    // Backup data for repeated runs
    memcpy(h_backup, h_data, size * sizeof(uint32_t));

    // Allocate GPU memory
    uint32_t *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(uint32_t)));

    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double sum = 0.0, tmin = 1e300, tmax = 0.0;
    bool all_sorted = true;

    for (int r = 0; r < repeats; r++) {
        // Restore original data
        memcpy(h_data, h_backup, size * sizeof(uint32_t));

        //passage data
        CUDA_CHECK(cudaMemcpy(d_data, h_data, size * sizeof(uint32_t), cudaMemcpyHostToDevice));

        // Sort
        CUDA_CHECK(cudaEventRecord(start));
        cub_radix_sort(d_data, size);
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

        // Verify only last run
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

    // Cleanup
    delete[] h_data;
    delete[] h_backup;
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return all_sorted ? 0 : 1;
}