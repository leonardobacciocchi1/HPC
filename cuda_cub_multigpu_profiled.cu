// cuda_cub_multigpu_profiled.cu
// =============================================================================
// =============================================================================

#include <cub/cub.cuh>              // CUB: primitive CUDA a basso livello
#include <thrust/device_ptr.h>      // Per wrappare puntatori per thrust::merge
#include <thrust/merge.h>           // thrust::merge (CUB non ha merge)
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <algorithm>

// Macro per controllo errori CUDA
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Struttura per raccogliere i tempi delle diverse fasi
struct ProfilingData {
    double sort_time = 0.0;   // Tempo di sort parallelo (CUB)
    double comm_time = 0.0;   // Tempo di trasferimento inter-GPU
    double merge_time = 0.0;  // Tempo di merge (Thrust)
    double total_time = 0.0;  // Tempo totale
};

bool is_sorted(const uint32_t* data, size_t n) {
    for (size_t i = 1; i < n; i++) {
        if (data[i-1] > data[i]) return false;
    }
    return true;
}

// =============================================================================
// Funzione di merge usando Thrust
// CUB non fornisce primitive di merge, usiamo thrust::merge
// =============================================================================
void merge_sorted_arrays_thrust(uint32_t* d_arr1, size_t n1,
                                  uint32_t* d_arr2, size_t n2,
                                  uint32_t* d_output) {
    //puntat raw CUDA in device_ptr per Thrust
    thrust::device_ptr<uint32_t> ptr1(d_arr1);
    thrust::device_ptr<uint32_t> ptr2(d_arr2);
    thrust::device_ptr<uint32_t> ptr_out(d_output);

    // Merge di due array ordinati
    thrust::merge(ptr1, ptr1 + n1, ptr2, ptr2 + n2, ptr_out);
}

// =============================================================================
// Multi-GPU CUB Sort con profiling dettagliato
// =============================================================================
ProfilingData multi_gpu_cub_sort_profiled(uint32_t* h_data, size_t n, int num_gpus) {
    ProfilingData prof;
    auto total_start = std::chrono::high_resolution_clock::now();

    if (num_gpus <= 0) {
        std::cerr << "Error: num_gpus must be > 0" << std::endl;
        exit(1);
    }

    int available_gpus;
    CUDA_CHECK(cudaGetDeviceCount(&available_gpus));
    if (num_gpus > available_gpus) {
        num_gpus = available_gpus;
    }

    // =========================================================================
    // CASO SINGLE-GPU Usa CUB direttamente
    // =========================================================================
    if (num_gpus == 1) {
        CUDA_CHECK(cudaSetDevice(0));

        auto sort_start = std::chrono::high_resolution_clock::now();

        // CUB richiede due buffer per il ping-pong interno
        uint32_t *d_keys_in, *d_keys_out;
        CUDA_CHECK(cudaMalloc(&d_keys_in, n * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_keys_out, n * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_keys_in, h_data, n * sizeof(uint32_t), cudaMemcpyHostToDevice));

        // DoubleBuffer gestisce lo scambio automatico tra buffer ad ogni passata
        cub::DoubleBuffer<uint32_t> d_keys(d_keys_in, d_keys_out);

        // Pattern CUB 
        // Query della dimensione del temp storage (d_temp_storage = nullptr)
        void *d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, n);

        // Alloca temp storage per istogrammi e contatori
        CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

        // Esecuzione effettiva del sort
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, n);

        // d_keys.Current() restituisce il buffer con il risultato finale
        // Per uint32_t (4 passate, numero pari) -> risultato in d_keys_in
        CUDA_CHECK(cudaMemcpy(h_data, d_keys.Current(), n * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_temp_storage));
        CUDA_CHECK(cudaFree(d_keys_in));
        CUDA_CHECK(cudaFree(d_keys_out));

        auto sort_end = std::chrono::high_resolution_clock::now();

        prof.sort_time = std::chrono::duration<double>(sort_end - sort_start).count();
        prof.total_time = prof.sort_time;
        return prof;
    }

    // =========================================================================
    // CASO MULTI-GPU
    // =========================================================================

    // Partizionamento bilanciato 
    std::vector<size_t> chunk_sizes(num_gpus);
    std::vector<size_t> chunk_offsets(num_gpus);

    size_t base_chunk = n / num_gpus;
    size_t remainder = n % num_gpus;

    size_t offset = 0;
    for (int i = 0; i < num_gpus; i++) {
        chunk_sizes[i] = base_chunk + (i < (int)remainder ? 1 : 0);
        chunk_offsets[i] = offset;
        offset += chunk_sizes[i];
    }

    // =========================================================================
    // SORT PARALLELO su ogni GPU
    // =========================================================================
    auto sort_start = std::chrono::high_resolution_clock::now();

    std::vector<uint32_t*> d_chunks(num_gpus);
    std::vector<cudaStream_t> streams(num_gpus);

    //Allocazione e trasferimento  asincrono 
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));
        CUDA_CHECK(cudaStreamCreate(&streams[gpu]));
        CUDA_CHECK(cudaMalloc(&d_chunks[gpu], chunk_sizes[gpu] * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpyAsync(d_chunks[gpu], h_data + chunk_offsets[gpu],
                                    chunk_sizes[gpu] * sizeof(uint32_t),
                                    cudaMemcpyHostToDevice, streams[gpu]));
    }

    //  Sort CUB su ogni GPU 

    // DoubleBuffer per ping-pong
    // temp_storage per istogrammi
    // Pattern query alloc execute
    std::vector<void*> d_temp_storage(num_gpus);
    std::vector<size_t> temp_storage_bytes(num_gpus);
    std::vector<uint32_t*> d_chunks_sorted(num_gpus);

    for (int gpu = 0; gpu < num_gpus; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));

        // Buffer di output per DoubleBuffer
        uint32_t* d_keys_out;
        CUDA_CHECK(cudaMalloc(&d_keys_out, chunk_sizes[gpu] * sizeof(uint32_t)));

        // DoubleBuffer: CUB alterna tra d_chunks[gpu] e d_keys_out
        cub::DoubleBuffer<uint32_t> d_keys(d_chunks[gpu], d_keys_out);

        // Query dimensione temp storage
        d_temp_storage[gpu] = nullptr;
        // Parametri: 0 = start_bit, 32 = end_bit (tutti i 32 bit di uint32_t)
        cub::DeviceRadixSort::SortKeys(d_temp_storage[gpu], temp_storage_bytes[gpu],
                                        d_keys, chunk_sizes[gpu], 0, 32, streams[gpu]);
        CUDA_CHECK(cudaMalloc(&d_temp_storage[gpu], temp_storage_bytes[gpu]));

        // Sort effettivo (asincrono sullo stream)
        cub::DeviceRadixSort::SortKeys(d_temp_storage[gpu], temp_storage_bytes[gpu],
                                        d_keys, chunk_sizes[gpu], 0, 32, streams[gpu]);

        // Copia risultato in un buffer dedicato per la fase di merge
        // d_keys.Current() punta al buffer con i dati ordinati
        CUDA_CHECK(cudaMalloc(&d_chunks_sorted[gpu], chunk_sizes[gpu] * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpyAsync(d_chunks_sorted[gpu], d_keys.Current(),
                                    chunk_sizes[gpu] * sizeof(uint32_t),
                                    cudaMemcpyDeviceToDevice, streams[gpu]));

        // Libera buffer temporanei del sort (non servono piu)
        CUDA_CHECK(cudaFree(d_chunks[gpu]));
        CUDA_CHECK(cudaFree(d_keys_out));
    }

    // Sincronizza tutte le GPU prima di procedere al merge
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));
        CUDA_CHECK(cudaStreamSynchronize(streams[gpu]));
    }

    auto sort_end = std::chrono::high_resolution_clock::now();
    prof.sort_time = std::chrono::duration<double>(sort_end - sort_start).count();

    
    auto comm_start = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaSetDevice(0));

    // Buffer per risultato merged (dimensione finale = n)
    uint32_t* d_merged;
    CUDA_CHECK(cudaMalloc(&d_merged, n * sizeof(uint32_t)));

    // Inizializza con chunk[0] (gia su GPU 0)
    CUDA_CHECK(cudaMemcpy(d_merged, d_chunks_sorted[0], chunk_sizes[0] * sizeof(uint32_t),
                          cudaMemcpyDeviceToDevice));

    size_t merged_size = chunk_sizes[0];

    // Buffer temporaneo per ping-pong del merge
    uint32_t* d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, n * sizeof(uint32_t)));

    auto comm_end = std::chrono::high_resolution_clock::now();
    double comm_time_accum = std::chrono::duration<double>(comm_end - comm_start).count();
    
    // Per ogni GPU i > 0
    // cudaMemcpyPeer: chunk[i] da GPU i -> GPU 0
    // swap(d_merged, d_temp)
    // =========================================================================
    double merge_time_accum = 0.0;

    for (int gpu = 1; gpu < num_gpus; gpu++) {
        //  Comunicazione inter-GPU 
        auto comm_iter_start = std::chrono::high_resolution_clock::now();

        uint32_t* d_chunk_on_gpu0;
        CUDA_CHECK(cudaMalloc(&d_chunk_on_gpu0, chunk_sizes[gpu] * sizeof(uint32_t)));

        // Trasferimento diretto GPU-to-GPU (senza CPU)
        
        CUDA_CHECK(cudaMemcpyPeer(d_chunk_on_gpu0, 0, d_chunks_sorted[gpu], gpu,
                                   chunk_sizes[gpu] * sizeof(uint32_t)));

        auto comm_iter_end = std::chrono::high_resolution_clock::now();
        comm_time_accum += std::chrono::duration<double>(comm_iter_end - comm_iter_start).count();

        //  Merge con Thrust 
        auto merge_iter_start = std::chrono::high_resolution_clock::now();

        // Usa thrust::merge (CUB non ha primitive di merge)
        merge_sorted_arrays_thrust(d_merged, merged_size,
                                     d_chunk_on_gpu0, chunk_sizes[gpu],
                                     d_temp);

        // Ping pong: scambia buffer
        std::swap(d_merged, d_temp);
        merged_size += chunk_sizes[gpu];

        auto merge_iter_end = std::chrono::high_resolution_clock::now();
        merge_time_accum += std::chrono::duration<double>(merge_iter_end - merge_iter_start).count();

        CUDA_CHECK(cudaFree(d_chunk_on_gpu0));
    }

    prof.comm_time = comm_time_accum;
    prof.merge_time = merge_time_accum;

    
    CUDA_CHECK(cudaMemcpy(h_data, d_merged, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Libera buffer di merge su GPU 0
    CUDA_CHECK(cudaFree(d_merged));
    CUDA_CHECK(cudaFree(d_temp));

    // Libera risorse su tutte le GPU
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));
        CUDA_CHECK(cudaFree(d_chunks_sorted[gpu]));
        CUDA_CHECK(cudaFree(d_temp_storage[gpu]));
        CUDA_CHECK(cudaStreamDestroy(streams[gpu]));
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    prof.total_time = std::chrono::duration<double>(total_end - total_start).count();

    return prof;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input.bin> [--gpus=N] [--repeats=N]" << std::endl;
        return 1;
    }

    const char* input_file = argv[1];
    int num_gpus = 1;
    int repeats = 1;

    for (int i = 2; i < argc; i++) {
        if (strncmp(argv[i], "--gpus=", 7) == 0) {
            num_gpus = atoi(argv[i] + 7);
        } else if (strncmp(argv[i], "--repeats=", 10) == 0) {
            repeats = atoi(argv[i] + 10);
        }
    }

    std::ifstream file(input_file, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Error: Cannot open " << input_file << std::endl;
        return 1;
    }

    size_t file_size = file.tellg();
    if (file_size % sizeof(uint32_t) != 0) {
        std::cerr << "Error: File size not multiple of 4" << std::endl;
        return 1;
    }

    size_t n = file_size / sizeof(uint32_t);
    file.seekg(0);

    std::vector<uint32_t> data(n);
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    file.close();

    std::vector<uint32_t> backup = data;

    int available_gpus;
    CUDA_CHECK(cudaGetDeviceCount(&available_gpus));

    std::cout << "Input: " << input_file << std::endl;
    std::cout << "Size: " << n << " elements (" << (file_size / (1024.0 * 1024.0)) << " MB)" << std::endl;
    std::cout << "Available GPUs: " << available_gpus << std::endl;
    std::cout << "Using GPUs: " << std::min(num_gpus, available_gpus) << std::endl;
    std::cout << "Repeats: " << repeats << std::endl;
    std::cout << std::endl;

    double total_sort = 0, total_comm = 0, total_merge = 0, total_total = 0;
    double min_time = 1e9, max_time = 0;

    for (int r = 0; r < repeats; r++) {
        data = backup;

        ProfilingData prof = multi_gpu_cub_sort_profiled(data.data(), n, num_gpus);

        total_sort += prof.sort_time;
        total_comm += prof.comm_time;
        total_merge += prof.merge_time;
        total_total += prof.total_time;
        min_time = std::min(min_time, prof.total_time);
        max_time = std::max(max_time, prof.total_time);

        bool sorted = is_sorted(data.data(), n);
        double throughput = (n / 1e6) / prof.total_time;

        std::cout << "Run " << (r+1) << "/" << repeats << ": " << prof.total_time << " sec";
        if (num_gpus > 1) {
            std::cout << " [Sort: " << prof.sort_time << "s, Comm: " << prof.comm_time
                      << "s, Merge: " << prof.merge_time << "s]";
        }
        std::cout << " (" << throughput << " M keys/sec) - Sorted: " << (sorted ? "YES" : "NO") << std::endl;

        if (!sorted) {
            std::cerr << "Error: Array not sorted!" << std::endl;
            return 2;
        }
    }

    std::cout << std::endl;
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "Average Total: " << (total_total / repeats) << " sec" << std::endl;
    if (num_gpus > 1) {
        std::cout << "  Avg Sort:    " << (total_sort / repeats) << " sec ("
                  << (100.0 * total_sort / total_total) << "%)" << std::endl;
        std::cout << "  Avg Comm:    " << (total_comm / repeats) << " sec ("
                  << (100.0 * total_comm / total_total) << "%)" << std::endl;
        std::cout << "  Avg Merge:   " << (total_merge / repeats) << " sec ("
                  << (100.0 * total_merge / total_total) << "%)" << std::endl;
    }
    std::cout << "Min:           " << min_time << " sec" << std::endl;
    std::cout << "Max:           " << max_time << " sec" << std::endl;
    std::cout << "Throughput:    " << (n / 1e6) / (total_total / repeats) << " M keys/sec" << std::endl;

    return 0;
}
