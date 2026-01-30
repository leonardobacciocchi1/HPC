// ============================================================================
// MSD Radix Sort OUT-OF-PLACE con OpenMP
// ===========================================================================


#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <sys/stat.h>

// Soglia minima per creare un task OpenMP
// Bucket più piccoli vengono processati sequenzialmente per evitare overhead
#ifndef TASK_THRESHOLD
#define TASK_THRESHOLD 65536
#endif

// Estrae un gruppo di bit da un numero uint32_t
// x     = numero da cui estrarre
// shift = posizione del bit meno significativo del gruppo
// mask  = maschera per isolare i bit 
static inline unsigned digit_u32(uint32_t x, int shift, unsigned mask) {
    return (unsigned)((x >> shift) & mask);
}


// Parametri:
//   src       - Array sorgente 
//   dst       - Array destinazione 
//   lo, hi    - Range [lo, hi) da ordinare
//   shift     - Posizione del bit LSB del gruppo corrente 
//   step_bits - Numero di bit per gruppo 
static void msd_rec_level(uint32_t *src, uint32_t *dst,
                          size_t lo, size_t hi,
                          int shift, unsigned step_bits)
{
    const size_t n = hi - lo;
    if (n <= 1 || shift < 0) return;  // Caso base: 0-1 elementi o bit esauriti

    // Calcola quanti bit usare a questo livello
    // Non possiamo usare più bit di quelli rimanenti 
    unsigned use_bits = (unsigned)((32 - shift) < (int)step_bits ? (32 - shift) : (int)step_bits);
    if (use_bits == 0) return;

    const unsigned RADIX = 1u << use_bits;  // Numero di bucket 
    const unsigned MASK  = RADIX - 1u;      // Maschera per estrarre i bit

    
    // Conta quanti elementi appartengono a ciascun bucket
   
    size_t *count  = (size_t*)calloc(RADIX, sizeof(size_t));  // Conteggio per bucket
    size_t *offset = (size_t*)malloc(RADIX * sizeof(size_t)); // Offset corrente 
    size_t *bbegin = (size_t*)malloc(RADIX * sizeof(size_t)); // Inizio di ogni bucket
    if (!count || !offset || !bbegin) { fprintf(stderr,"oom\n"); exit(1); }

    // Conta gli elementi per ogni bucket
    for (size_t i = lo; i < hi; ++i) {
        count[digit_u32(src[i], shift, MASK)]++;
    }

    
    // Se tutti gli elementi sono nello stesso bucket, salta scatter e copy-back
    for (unsigned d = 0; d < RADIX; ++d) {
        if (count[d] == n) {
            free(bbegin); free(offset); free(count);
            int next_shift = shift - (int)step_bits;
            if (next_shift >= 0) msd_rec_level(src, dst, lo, hi, next_shift, step_bits);
            return;
        }
    }

    // Prefix sum
    size_t sum = lo;
    for (unsigned d = 0; d < RADIX; ++d) {
        offset[d] = sum;    // Posizione corrente per lo scatter
        sum += count[d];    // Avanza di count[d] posizioni
    }
    memcpy(bbegin, offset, RADIX * sizeof(size_t));  // Salva le posizioni iniziali

    // Copia ogni elemento da src alla sua posizione corretta in dst

    for (size_t i = lo; i < hi; ++i) {
        unsigned d = digit_u32(src[i], shift, MASK);
        dst[offset[d]++] = src[i];  // Copia e incrementa offset
    }


    // Ogni bucket viene ordinato ricorsivamente per i bit successivi
    int next_shift = shift - (int)step_bits;  // Prossimo gruppo di bit
    #pragma omp taskgroup  // Attende il completamento di tutti i task figli
    {
        for (unsigned d = 0; d < RADIX; ++d) {
            size_t b_lo = bbegin[d];
            size_t b_hi = b_lo + count[d];
            size_t m    = b_hi - b_lo;
            if (m <= 1) continue;  // Bucket con 0-1 elementi: già ordinato

            if (m >= TASK_THRESHOLD) {
                // Bucket grande: crea un task parallelo
                #pragma omp task firstprivate(b_lo,b_hi,next_shift,src,dst,step_bits)
                { if (next_shift >= 0) msd_rec_level(dst, src, b_lo, b_hi, next_shift, step_bits); }
            } else {
                // Bucket piccolo: esegui sequenzialmente
                if (next_shift >= 0) msd_rec_level(dst, src, b_lo, b_hi, next_shift, step_bits);
            }
        }
    }


    // Copia i dati ordinati da dst a src per mantenere l'invariante
    memcpy(src + lo, dst + lo, n * sizeof(uint32_t));

    free(bbegin); free(offset); free(count);
}


// Parametri:
//   a           - Array da ordinare (modificato in-place)
//   n           - Numero di elementi
//   bits        - Bit per livello di radix (1-16)
//   num_threads - Numero di thread OpenMP da usare
//
// Gestisce l'allocazione della memoria temporanea e calcola il primo shift

static void msd_radix_sort_u32_omp_bits(uint32_t *a, size_t n, int bits, int num_threads)
{
    if (!a || n <= 1) return;
    if (bits < 1)  bits = 8;   // Minimo 1 bit per livello
    if (bits > 16) bits = 16;  // Massimo 16 bit (64K bucket)

    omp_set_num_threads(num_threads);

    const unsigned step_bits  = (unsigned)bits;
    const unsigned rem        = 32u % step_bits;  // Resto della divisione 32/bits
    // Se 32 non è divisibile per bits, il primo livello usa i bit rimanenti
    
    const unsigned first_bits = (rem == 0u) ? step_bits : rem;
    const int      first_shift= 32 - (int)first_bits;  // Posizione del primo gruppo 

    // Alloca array temporaneo per lo scatter out-of-place
    uint32_t *tmp = (uint32_t*)malloc(n * sizeof(uint32_t));
    if (!tmp) { fprintf(stderr,"malloc tmp\n"); exit(1); }

    // Avvia la regione parallela con un singolo thread che crea i task
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            msd_rec_level(a, tmp, 0, n, first_shift, step_bits);
        }
    }

    free(tmp);
}



// Verifica se l'array è ordinato in modo crescente
static int is_sorted_u32(const uint32_t *v, size_t n) {
    for (size_t i=1;i<n;++i) if (v[i-1] > v[i]) return 0;
    return 1;
}


int main(int argc, char** argv)
{
    // Verifica argomenti minimi
    if (argc < 2) {
        fprintf(stderr,
            "Uso: %s <input.bin> [--bits=K] [--threads=N] [--repeats=N] [--out=sorted.bin]\n", argv[0]);
        return 1;
    }

    // Valori di default
    const char* in_path  = argv[1];
    int bits             = 8;                      // 8 bit = 256 bucket
    int num_threads      = omp_get_max_threads();  // Tutti i core disponibili
    int repeats          = 1;
    const char* out_path = NULL;

    // Parsing degli argomenti opzionali
    for (int i=2; i<argc; ++i) {
        if (strncmp(argv[i], "--bits=", 7) == 0)       bits = atoi(argv[i] + 7);
        else if (strncmp(argv[i], "--threads=", 10)==0) num_threads = atoi(argv[i] + 10);
        else if (strncmp(argv[i], "--repeats=", 10)==0) { repeats = atoi(argv[i] + 10); if (repeats < 1) repeats = 1; }
        else if (strncmp(argv[i], "--out=", 6) == 0)   out_path = argv[i] + 6;
        else { fprintf(stderr, "Argomento sconosciuto: %s\n", argv[i]); return 1; }
    }

    // Leggi dimensione file e verifica validità
    struct stat st;
    if (stat(in_path, &st) != 0) { perror("stat"); return 1; }
    if (st.st_size % 4 != 0) { fprintf(stderr,"Errore: size non multiplo di 4.\n"); return 1; }
    size_t n = (size_t)st.st_size / 4;  // Numero di uint32_t nel file

    // Alloca e carica l'array originale
    uint32_t *a0 = (uint32_t*)malloc(n * sizeof(uint32_t));
    if (!a0) { fprintf(stderr,"malloc a0\n"); return 1; }
    FILE* f = fopen(in_path, "rb"); if (!f) { perror("fopen input"); free(a0); return 1; }
    size_t rd = fread(a0, sizeof(uint32_t), n, f); fclose(f);
    if (rd != n) { fprintf(stderr,"fread incompleto\n"); free(a0); return 1; }

    // Alloca array di lavoro 
    uint32_t *a = (uint32_t*)malloc(n * sizeof(uint32_t));
    if (!a) { fprintf(stderr,"malloc a\n"); free(a0); return 1; }

    // Variabili per statistiche benchmark
    double sum=0.0, tmin=1e300, tmax=0.0;
    printf("n=%zu  bits=%d  repeats=%d  threads=%d\n",
           n, bits, repeats, num_threads);

    // Esegui benchmark per il numero di ripetizioni richieste
    for (int r=1; r<=repeats; ++r) {
        memcpy(a, a0, n * sizeof(uint32_t));  // Ripristina dati originali

        // Misura tempo di esecuzione
        double t0 = omp_get_wtime();
        msd_radix_sort_u32_omp_bits(a, n, bits, num_threads);
        double t1 = omp_get_wtime();

        // Aggiorna statistiche
        double dt = t1 - t0;
        sum += dt; if (dt < tmin) tmin = dt; if (dt > tmax) tmax = dt;

        // Verifica correttezza
        int ok = is_sorted_u32(a, n);
        printf("[pass %d/%d] sorted=%d  time=%.3f s\n", r, repeats, ok, dt);
        if (!ok) { fprintf(stderr,"Errore: risultato non ordinato alla passata %d.\n", r); free(a); free(a0); return 2; }
    }

    // Stampa statistiche finali
    printf("==> avg=%.3f s   min=%.3f s   max=%.3f s (over %d runs)\n", sum/repeats, tmin, tmax, repeats);

    // Scrivi output se richiesto
    if (out_path) {
        FILE* fo = fopen(out_path, "wb"); if (!fo) { perror("fopen output"); free(a); free(a0); return 1; }
        size_t wr = fwrite(a, sizeof(uint32_t), n, fo); fclose(fo);
        if (wr != n) { fprintf(stderr,"fwrite incompleto\n"); free(a); free(a0); return 1; }
        printf("Output scritto in: %s\n", out_path);
    }

    // Libera memoria
    free(a); free(a0);
    return 0;
}
