// ============================================================================
// MSD-LSD Hybrid Radix Sort
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
#include <sys/stat.h>



// Termina il programma con un messaggio di errore
static void die(const char* msg){ fprintf(stderr,"Errore: %s\n", msg); exit(1); }

// Verifica se l'array è ordinato in modo crescente
static int is_sorted_u32(const uint32_t *v, size_t n){
    for(size_t i=1;i<n;i++) if(v[i-1] > v[i]) return 0;
    return 1;
}

// Carica un file binario uint32_t
// Ritorna il puntatore all'array allocato e imposta *out_n al numero di elementi
static uint32_t* load_file_u32(const char* path, size_t *out_n){
    struct stat st;
    if (stat(path, &st) != 0) { perror("stat"); die("impossibile leggere il file"); }
    if ((st.st_size % 4) != 0) die("size file non multipla di 4");
    size_t n = (size_t)st.st_size / 4;
    FILE* f = fopen(path, "rb");
    if(!f){ perror("fopen"); die("impossibile aprire input"); }
    uint32_t *a = (uint32_t*)malloc(n*sizeof(uint32_t));
    if(!a) die("malloc input");
    size_t rd = fread(a, sizeof(uint32_t), n, f);
    fclose(f);
    if (rd != n) die("fread incompleto");
    *out_n = n;
    return a;
}


#define MSD_BITS 8              // Bit per la fase MSD 
#define LSD_BITS 8              // Bit per ogni passata LSD 
#define MSD_RADIX (1 << MSD_BITS)  // 256 bucket per MSD
#define LSD_RADIX (1 << LSD_BITS)  // 256 bucket per LSD


static void lsd_bucket(uint32_t *a, size_t n, int start_shift, int num_passes){
    if(n <= 1) return;

    // Alloca buffer temporaneo per il ping-pong
    uint32_t *temp = (uint32_t*)malloc(n * sizeof(uint32_t));
    if(!temp) die("malloc temp");

    // Contatori per counting sort
    size_t *count = (size_t*)malloc(LSD_RADIX * sizeof(size_t));
    size_t *offset = (size_t*)malloc(LSD_RADIX * sizeof(size_t));
    if(!count || !offset) die("malloc count/offset");

    // Puntatori 
    uint32_t *src = a;      // Sorgente corrente
    uint32_t *dst = temp;   // Destinazione corrente

    // Esegue le passate LSD 
    for(int pass = 0; pass < num_passes; pass++){
        int shift = start_shift + pass * LSD_BITS;  // Posizione del gruppo di bit
        uint32_t mask = (1u << LSD_BITS) - 1;       // Maschera per estrarre i bit

        // conta elementi per bucket
        memset(count, 0, LSD_RADIX * sizeof(size_t));
        for(size_t i = 0; i < n; i++){
            unsigned digit = (unsigned)((src[i] >> shift) & mask);
            count[digit]++;
        }

        // calcola posizioni di scrittura
        size_t acc = 0;
        for(int b = 0; b < LSD_RADIX; b++){
            offset[b] = acc;
            acc += count[b];
        }

        // copia elementi nelle posizioni corrette
        for(size_t i = 0; i < n; i++){
            uint32_t val = src[i];
            unsigned digit = (unsigned)((val >> shift) & mask);
            dst[offset[digit]] = val;
            offset[digit]++;
        }

        // Swap buffers per la prossima passata
        uint32_t *tmp = src;
        src = dst;
        dst = tmp;
    }

    // Se numero passate dispari, il risultato è in temp copia in a
    if(num_passes % 2 != 0){
        memcpy(a, temp, n * sizeof(uint32_t));
    }

    free(temp);
    free(count);
    free(offset);
}


static void msd_lsd_hybrid(uint32_t *a, size_t n, int num_threads){
    if(n <= 1) return;

    omp_set_num_threads(num_threads);

    // Calcola numero di passate LSD necessarie

    uint32_t maxv = a[0];
    #pragma omp parallel for reduction(max:maxv)
    for(size_t i=1; i<n; i++) if(a[i] > maxv) maxv = a[i];

    if(maxv == 0) return;  // Tutti zeri: già ordinato

    // Calcola bit significativi e passate LSD necessarie
    int total_bits = 31 - __builtin_clz(maxv) + 1;  // Bit usati dal valore max
    int lsd_bits = (total_bits > MSD_BITS) ? (total_bits - MSD_BITS) : 0;  // Bit per LSD
    int lsd_passes = (lsd_bits + LSD_BITS - 1) / LSD_BITS;  // Passate LSD 

    // MSD sui primi 8 bit 
  
    const int msd_shift = 24;  // Shift 
    const uint32_t msd_mask = (1u << MSD_BITS) - 1;  // Maschera 0xFF

    size_t *count = (size_t*)calloc(MSD_RADIX, sizeof(size_t));   // Conteggio globale
    size_t *offset = (size_t*)malloc(MSD_RADIX * sizeof(size_t)); // Posizione inizio bucket
    if(!count || !offset) die("malloc count/offset");


    // Counting parallelo
    // Ogni thread conta gli elementi nel proprio chunk

    const int Tmax = num_threads;
    size_t *thread_counts = (size_t*)calloc((size_t)Tmax * MSD_RADIX, sizeof(size_t));
    if(!thread_counts) die("malloc thread_counts");

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int Tloc = omp_get_num_threads();
        size_t *lc = thread_counts + (size_t)tid * MSD_RADIX;  // Istogramma locale

        // Range di questo thread
        size_t s = (n * (size_t)tid) / Tloc;
        size_t e = (n * (size_t)(tid + 1)) / Tloc;

        // Conta elementi per bucket
        for(size_t i = s; i < e; i++){
            unsigned d = (unsigned)((a[i] >> msd_shift) & msd_mask);
            lc[d]++;
        }
    }


    // Riduzione istogrammi 
    for(int b = 0; b < MSD_RADIX; b++){
        size_t sum = 0;
        for(int t = 0; t < Tmax; t++) sum += thread_counts[(size_t)t * MSD_RADIX + b];
        count[b] = sum;
    }

    // calcola posizione di inizio di ogni bucket
    size_t acc = 0;
    for(int b = 0; b < MSD_RADIX; b++){
        offset[b] = acc;
        acc += count[b];
    }


    // Copia ogni elemento nella sua posizione nel bucket corretto
    uint32_t *temp = (uint32_t*)malloc(n * sizeof(uint32_t));
    if(!temp) die("malloc temp");

    // write_pos[b] = prossima posizione di scrittura nel bucket b
    size_t *write_pos = (size_t*)malloc(MSD_RADIX * sizeof(size_t));
    if(!write_pos) die("malloc write_pos");
    memcpy(write_pos, offset, MSD_RADIX * sizeof(size_t));

    // Scatter: ogni elemento va nel suo bucket
    for(size_t i = 0; i < n; i++){
        uint32_t val = a[i];
        unsigned digit = (unsigned)((val >> msd_shift) & msd_mask);
        temp[write_pos[digit]] = val;
        write_pos[digit]++;
    }

    // Copia risultato in a
    memcpy(a, temp, n * sizeof(uint32_t));
    free(temp);
    free(write_pos);

    // Ogni bucket viene ordinato indipendentemente usando LSD.
    if(lsd_passes > 0){
        #pragma omp parallel for schedule(dynamic)
        for(int bucket = 0; bucket < MSD_RADIX; bucket++){
            size_t bucket_start = offset[bucket];  // Inizio del bucket in a[]
            size_t bucket_size = count[bucket];    // Dimensione del bucket

            if(bucket_size > 1){
                // Ordina il bucket con LSD 
                lsd_bucket(a + bucket_start, bucket_size, 0, lsd_passes);
            }
        }
    }

    // Libera memoria ausiliaria
    free(thread_counts);
    free(count);
    free(offset);
}


int main(int argc, char** argv){
    // Verifica argomenti minimi
    if (argc < 2) {
        fprintf(stderr, "Uso: %s <input.bin> [--threads=N] [--repeats=N] [--no-check]\n", argv[0]);
        return 1;
    }

    // Valori di default
    const char* in_path = argv[1];
    int num_threads = omp_get_max_threads();  // Tutti i core disponibili
    int repeats = 1;
    int do_check = 1;  // Verifica correttezza abilitata

    // Parsing degli argomenti opzionali
    for (int i = 2; i < argc; i++){
        if (strncmp(argv[i], "--threads=", 10) == 0)      num_threads = atoi(argv[i] + 10);
        else if (strncmp(argv[i], "--repeats=", 10) == 0) repeats = atoi(argv[i] + 10);
        else if (strcmp(argv[i], "--no-check") == 0)      do_check = 0;
        else { fprintf(stderr, "Argomento sconosciuto: %s\n", argv[i]); return 1; }
    }

    if (repeats < 1) repeats = 1;

    size_t n = 0;
    uint32_t *a0 = load_file_u32(in_path, &n);  // Array originale

    printf("n=%zu  MSD_bits=%d  LSD_bits=%d  repeats=%d  threads=%d  [HYBRID]\n",
           n, MSD_BITS, LSD_BITS, repeats, num_threads);


    double sum = 0.0, tmin = 1e300, tmax = 0.0;
    uint32_t *a = (uint32_t*)malloc(n * sizeof(uint32_t));  // Array di lavoro
    if(!a) die("malloc a");

    // Esegui benchmark per il numero di ripetizioni richieste
    for (int r = 1; r <= repeats; r++){
        memcpy(a, a0, n * sizeof(uint32_t));  // Ripristina dati originali

        // Misura tempo di esecuzione
        double t0 = omp_get_wtime();
        msd_lsd_hybrid(a, n, num_threads);
        double t1 = omp_get_wtime();

        // Aggiorna statistiche
        double dt = t1 - t0;
        sum += dt;
        if (dt < tmin) tmin = dt;
        if (dt > tmax) tmax = dt;

        int ok = do_check ? is_sorted_u32(a, n) : 1;
        printf("[pass %d/%d] sorted=%d  time=%.3f s\n", r, repeats, ok, dt);

        if (do_check && !ok){
            fprintf(stderr, "Risultato non ordinato!\n");
            free(a0);
            free(a);
            return 2;
        }
    }

    // Stampa statistiche finali
    printf("==> avg=%.3f s   min=%.3f s   max=%.3f s (over %d runs)\n",
           sum / repeats, tmin, tmax, repeats);

    // Libera memoria
    free(a0);
    free(a);
    return 0;
}
