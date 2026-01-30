// cuda_thrust_multigpu_profiled.cu
// =============================================================================
// =============================================================================

#include <thrust/device_vector.h>   // Container GPU gestito automaticamente
#include <thrust/host_vector.h>     // Container CPU
#include <thrust/sort.h>            // thrust::sort (LSD Radix Sort per interi)
#include <thrust/merge.h>           // thrust::merge (per array ordinati)
#include <thrust/execution_policy.h> // thrust::cuda::par.on(stream)
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <chrono>

// Macro per controllo errori CUDA con informazioni di debug
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Struttura per raccogliere i tempi delle diverse fasi
// Permette di analizzare dove si spende il tempo nell'esecuzione multi-GPU
struct ProfilingData {
    double sort_time = 0.0;   // Tempo di sort parallelo su tutte le GPU
    double comm_time = 0.0;   // Tempo di trasferimento inter-GPU (cudaMemcpyPeer)
    double merge_time = 0.0;  // Tempo di merge su GPU 0
    double total_time = 0.0;  // Tempo totale end-to-end
};

bool is_sorted(const uint32_t* data, size_t n) {
    for (size_t i = 1; i < n; i++) {
        if (data[i-1] > data[i]) return false;
    }
    return true;
}

// =============================================================================
// Multi-GPU Thrust Sort con profiling dettagliato
// =============================================================================
ProfilingData multi_gpu_thrust_sort_profiled(uint32_t* h_data, size_t n, int num_gpus) {
    ProfilingData prof;
    auto total_start = std::chrono::high_resolution_clock::now();

    if (num_gpus <= 0) {
        std::cerr << "Error: num_gpus must be > 0" << std::endl;
        exit(1);
    }

    // Verifica GPU disponibili nel sistema
    int available_gpus;
    CUDA_CHECK(cudaGetDeviceCount(&available_gpus));
    if (num_gpus > available_gpus) {
        num_gpus = available_gpus;
    }

    // =========================================================================
    // CASO SINGLE-GPU
    // =========================================================================
    if (num_gpus == 1) {
        CUDA_CHECK(cudaSetDevice(0));

        auto sort_start = std::chrono::high_resolution_clock::now();

        // device_vector gestisce automaticamente allocazione e trasferimento 
        thrust::device_vector<uint32_t> d_data(h_data, h_data + n);

        // thrust::sort seleziona automaticamente LSD Radix Sort per uint32_t
        thrust::sort(d_data.begin(), d_data.end());

        // Copia risultato 
        thrust::copy(d_data.begin(), d_data.end(), h_data);
        auto sort_end = std::chrono::high_resolution_clock::now();

        prof.sort_time = std::chrono::duration<double>(sort_end - sort_start).count();
        prof.total_time = prof.sort_time;
        return prof;
    }

    // =========================================================================
    // CASO MULTI-GPU
    // =========================================================================

    //  Partizionamento bilanciato dei dati tra le GPU 
    // Ogni GPU riceve n/num_gpus elementi, con le prime GPU che ricevono
    // un elemento extra se n non e divisibile per num_gpus
    std::vector<size_t> chunk_sizes(num_gpus);
    std::vector<size_t> chunk_offsets(num_gpus);

    size_t base_chunk = n / num_gpus;
    size_t remainder = n % num_gpus;

    size_t offset = 0;
    for (int i = 0; i < num_gpus; i++) {
        // Le prime GPU ricevono un elemento extra
        chunk_sizes[i] = base_chunk + (i < (int)remainder ? 1 : 0);
        chunk_offsets[i] = offset;
        offset += chunk_sizes[i];
    }

    // =========================================================================
    // Ogni GPU ordina il proprio chunk
    // =========================================================================
    auto sort_start = std::chrono::high_resolution_clock::now();

    std::vector<uint32_t*> d_chunks(num_gpus);      // Puntatori ai chunk su ogni GPU
    std::vector<cudaStream_t> streams(num_gpus);    // Stream per operazioni async
    std::vector<cudaEvent_t> events(num_gpus);      // Eventi per sincronizzazione

    //  Allocazione memoria e trasferimento asincrono 
    // Ogni GPU ha il proprio stream, permettendo overlap tra trasferimenti
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));  // Seleziona GPU corrente
        CUDA_CHECK(cudaStreamCreate(&streams[gpu]));
        CUDA_CHECK(cudaEventCreate(&events[gpu]));

        // Alloca memoria sulla GPU corrente
        CUDA_CHECK(cudaMalloc(&d_chunks[gpu], chunk_sizes[gpu] * sizeof(uint32_t)));

        // Trasferimento asincrono: la CPU non aspetta il completamento
        // Permette di lanciare trasferimenti su piu GPU in parallelo
        CUDA_CHECK(cudaMemcpyAsync(d_chunks[gpu],
                                    h_data + chunk_offsets[gpu],
                                    chunk_sizes[gpu] * sizeof(uint32_t),
                                    cudaMemcpyHostToDevice,
                                    streams[gpu]));
    }

    // Sort parallelo: ogni GPU ordina indipendentemente
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));

        // device_ptr wrappa il puntatore raw per l'uso con Thrust
        thrust::device_ptr<uint32_t> dev_ptr(d_chunks[gpu]);

        // thrust::cuda::par.on(stream) 

        // Questo permette esecuzione concorrente su GPU diverse
        thrust::sort(thrust::cuda::par.on(streams[gpu]),
                     dev_ptr, dev_ptr + chunk_sizes[gpu]);

        // Registra evento per sapere quando il sort e' completato
        CUDA_CHECK(cudaEventRecord(events[gpu], streams[gpu]));
    }

    //  Sincronizzazione: attendi che tutte le GPU completino il sort
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));
        CUDA_CHECK(cudaEventSynchronize(events[gpu]));
    }

    auto sort_end = std::chrono::high_resolution_clock::now();
    prof.sort_time = std::chrono::duration<double>(sort_end - sort_start).count();

   
    // Prepara buffer su GPU 0
   
    auto comm_start = std::chrono::high_resolution_clock::now();

    // Tutte le operazioni di merge avvengono su GPU 0
    CUDA_CHECK(cudaSetDevice(0));

    // Buffer per l'array merged 
    uint32_t* d_merged;
    CUDA_CHECK(cudaMalloc(&d_merged, n * sizeof(uint32_t)));

    // Inizializza con il primo chunk 
    CUDA_CHECK(cudaMemcpy(d_merged, d_chunks[0],
                          chunk_sizes[0] * sizeof(uint32_t),
                          cudaMemcpyDeviceToDevice));

    size_t merged_size = chunk_sizes[0];

    // Buffer temporaneo per ping pong durante il merge
    uint32_t* d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, n * sizeof(uint32_t)));

    auto comm_end = std::chrono::high_resolution_clock::now();
    double comm_time_accum = std::chrono::duration<double>(comm_end - comm_start).count();

   
    
   
    auto merge_start = std::chrono::high_resolution_clock::now();
    double merge_time_accum = 0.0;
// Per ogni GPU i (i > 0):
    for (int gpu = 1; gpu < num_gpus; gpu++) {
        //  Comunicazione: trasferimento GPU a GPU
        auto comm_iter_start = std::chrono::high_resolution_clock::now();

        // Alloca buffer temporaneo su GPU 0 per ricevere il chunk
        uint32_t* d_chunk_on_gpu0;
        CUDA_CHECK(cudaMalloc(&d_chunk_on_gpu0, chunk_sizes[gpu] * sizeof(uint32_t)));

        // cudaMemcpyPeer: trasferimento diretto tra GPU senza passare per CPU
        
        CUDA_CHECK(cudaMemcpyPeer(d_chunk_on_gpu0, 0,    // destinazione: GPU 0
                                   d_chunks[gpu], gpu,    // sorgente: GPU i
                                   chunk_sizes[gpu] * sizeof(uint32_t)));

        auto comm_iter_end = std::chrono::high_resolution_clock::now();
        comm_time_accum += std::chrono::duration<double>(comm_iter_end - comm_iter_start).count();

        // Merge: combina array ordinati 
        auto merge_iter_start = std::chrono::high_resolution_clock::now();

        // Wrappa puntatori raw per Thrust
        thrust::device_ptr<uint32_t> merged_ptr(d_merged);
        thrust::device_ptr<uint32_t> chunk_ptr(d_chunk_on_gpu0);
        thrust::device_ptr<uint32_t> temp_ptr(d_temp);

        // thrust::merge: combina due array ordinati 
        // merged[0..merged_size) + chunk[0..chunk_size) -> temp[0..new_size)
        thrust::merge(merged_ptr, merged_ptr + merged_size,
                      chunk_ptr, chunk_ptr + chunk_sizes[gpu],
                      temp_ptr);

        // Ping pong: scambia i buffer per la prossima iterazione
        std::swap(d_merged, d_temp);
        merged_size += chunk_sizes[gpu];

        auto merge_iter_end = std::chrono::high_resolution_clock::now();
        merge_time_accum += std::chrono::duration<double>(merge_iter_end - merge_iter_start).count();

        // Libera buffer temp
        CUDA_CHECK(cudaFree(d_chunk_on_gpu0));
    }

    auto merge_end = std::chrono::high_resolution_clock::now();

    prof.comm_time = comm_time_accum;
    prof.merge_time = merge_time_accum;


    // Copia risultato ordinato dalla GPU 0 alla memoria host
    CUDA_CHECK(cudaMemcpy(h_data, d_merged, n * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    //  Cleanup: libera tutte le risorse allocate 
    CUDA_CHECK(cudaFree(d_merged));
    CUDA_CHECK(cudaFree(d_temp));

    for (int gpu = 0; gpu < num_gpus; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));
        CUDA_CHECK(cudaFree(d_chunks[gpu]));
        CUDA_CHECK(cudaStreamDestroy(streams[gpu]));
        CUDA_CHECK(cudaEventDestroy(events[gpu]));
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

    // Load input
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

    // Info
    int available_gpus;
    CUDA_CHECK(cudaGetDeviceCount(&available_gpus));

    std::cout << "Input: " << input_file << std::endl;
    std::cout << "Size: " << n << " elements (" << (file_size / (1024.0 * 1024.0)) << " MB)" << std::endl;
    std::cout << "Available GPUs: " << available_gpus << std::endl;
    std::cout << "Using GPUs: " << std::min(num_gpus, available_gpus) << std::endl;
    std::cout << "Repeats: " << repeats << std::endl;
    std::cout << std::endl;

    // Profiling accumulators
    double total_sort = 0, total_comm = 0, total_merge = 0, total_total = 0;
    double min_time = 1e9, max_time = 0;

    for (int r = 0; r < repeats; r++) {
        data = backup;

        ProfilingData prof = multi_gpu_thrust_sort_profiled(data.data(), n, num_gpus);

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