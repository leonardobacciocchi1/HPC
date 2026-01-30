// ============================================================================
// MSD Radix Sort IN-PLACE American Flag Sort con OpenMP
// ============================================================================


#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <sys/stat.h>

// Soglia minima per creare un task OpenMP parallelo
// Segmenti più piccoli vengono processati sequenzialmente
#ifndef TASK_THRESHOLD
#define TASK_THRESHOLD   65536
#endif

// Soglia sotto la quale si usa qsort invece del radix sort
// qsort è più efficiente per array piccoli (meno overhead)
#ifndef QSORT_THRESHOLD
#define QSORT_THRESHOLD  2048
#endif

// Estrae un gruppo di bit da un numero uint32_t
// x     = numero da cui estrarre
// shift = posizione del bit meno significativo del gruppo
// mask  = maschera per isolare i bit 
static inline unsigned digit_u32(uint32_t x, int shift, unsigned mask) {
    return (unsigned)((x >> shift) & mask);
}

// Funzione di comparazione per qsort 
// Ritorna: -1 se a<b, 0 se a==b, 1 se a>b
static int cmp_u32(const void* a, const void* b) {
    uint32_t x = *(const uint32_t*)a, y = *(const uint32_t*)b;
    return (x > y) - (x < y);
}

// Parametri:
//   a         - Array da ordinare (modificato in-place)
//   lo, hi    - Range [lo, hi) da ordinare
//   shift     - Posizione del bit LSB del gruppo corrente (0..31)
//   step_bits - Numero di bit per gruppo (da --bits)

static void msd_rec_u32_ip(uint32_t *a, size_t lo, size_t hi, int shift, unsigned step_bits)
{
    const size_t n = hi - lo;
    if (n <= 1 || shift < 0) return;  // Caso base: 0-1 elementi o bit esauriti

    //per array piccoli usa qsort 
    if (n <= QSORT_THRESHOLD) { qsort(a + lo, n, sizeof(uint32_t), cmp_u32); return; }

    // Calcola quanti bit usare a questo livello
    
    unsigned use_bits = (unsigned)((32 - shift) < (int)step_bits ? (32 - shift) : (int)step_bits);
    if (use_bits == 0) return;

    const unsigned RADIX = 1u << use_bits;  // Numero di bucket (es. 256 per 8 bit)
    const unsigned MASK  = RADIX - 1u;      // Maschera per estrarre i bit

    //Counting 
    // Conta quanti elementi appartengono a ciascun bucket
    size_t* count = (size_t*)calloc(RADIX, sizeof(size_t));
    if (!count) { fprintf(stderr,"oom count\n"); exit(1); }
    for (size_t i = lo; i < hi; ++i) count[digit_u32(a[i], shift, MASK)]++;

    // Se tutti gli elementi hanno lo stesso valore per questi bit, salta il posizionamento e procedi al livello successivo
    for (unsigned d=0; d<RADIX; ++d) {
        if (count[d] == n) {
            free(count);
            int next_shift = shift - (int)step_bits;
            if (next_shift >= 0) msd_rec_u32_ip(a, lo, hi, next_shift, step_bits);
            return;
        }
    }
    // Calcolo dei range dei bucket
    // begin[d] = inizio del bucket d
    // end[d]   = fine del bucket d 
    // next[d]  = prossima posizione libera nel bucket d 

    size_t* begin = (size_t*)malloc(RADIX * sizeof(size_t));
    size_t* end   = (size_t*)malloc(RADIX * sizeof(size_t));
    size_t* next  = (size_t*)malloc(RADIX * sizeof(size_t));
    if (!begin || !end || !next) { fprintf(stderr,"oom beg/end/next\n"); exit(1); }

    // Calcola i range contigui per ogni bucket tramite prefix sum
    size_t s = lo;
    for (unsigned d=0; d<RADIX; ++d) {
        begin[d] = s;              // Inizio del bucket
        end[d] = s + count[d];     // Fine del bucket
        next[d] = begin[d];        // Puntatore di scrittura (inizialmente = begin)
        s = end[d];                // Avanza al prossimo bucket
    }

    // Sposta ogni elemento nella sua posizione corretta tramite swap.
    for (unsigned d = 0; d < RADIX; ++d) {
        while (next[d] < end[d]) {
            uint32_t v  = a[next[d]];                    // Elemento corrente
            unsigned dv = digit_u32(v, shift, MASK);     // Bucket di appartenenza

            if (dv == d) {
                // Elemento già nel bucket corretto
                next[d]++;
            } else {
                // Scambia con l'elemento nella posizione di destinazione
                uint32_t tmp = a[next[dv]];
                a[next[dv]] = v;      // Metti v nel suo bucket
                a[next[d]]  = tmp;    // Metti tmp  
                next[dv]++;           // Avanza il puntatore del bucket destinazione
            }
        }
    }

    // Ordina ricorsivamente ogni bucket per i bit successivi.
    int next_shift = shift - (int)step_bits;  // Prossimo gruppo di bit
    #pragma omp taskgroup  // Attende il completamento di tutti i task figli
    {
        for (unsigned d = 0; d < RADIX; ++d) {
            size_t b_lo = begin[d], b_hi = end[d], m = b_hi - b_lo;
            if (m <= 1) continue;  // Bucket con 0-1 elementi già ordinato

            if (m >= TASK_THRESHOLD) {
                // Bucket grande: crea un task parallelo
                #pragma omp task firstprivate(b_lo,b_hi,next_shift,a,step_bits)
                { if (next_shift >= 0) msd_rec_u32_ip(a, b_lo, b_hi, next_shift, step_bits); }
            } else {
                // Bucket piccolo: esegui sequenzialmente 
                if (next_shift >= 0) msd_rec_u32_ip(a, b_lo, b_hi, next_shift, step_bits);
            }
        }
    }

    // Libera memoria delle strutture ausiliarie
    free(next); free(end); free(begin); free(count);
}

// Parametri:
//   a           - Array da ordinare (modificato in-place)
//   n           - Numero di elementi
//   bits        - Bit per livello di radix (1-16)
//   num_threads - Numero di thread OpenMP da usare

static void msd_radix_sort_u32_ip_bits(uint32_t *a, size_t n, int bits, int num_threads)
{
    if (!a || n <= 1) return;

    omp_set_num_threads(num_threads);

    const unsigned step_bits  = (unsigned)bits;
    const unsigned rem        = 32u % step_bits;  // Resto della divisione 32/bits
    // Se 32 non è divisibile per bits, il primo livello usa i bit rimanenti
  
    const unsigned first_bits = (rem == 0u) ? step_bits : rem;
    const int      first_shift= 32 - (int)first_bits;  // Posizione del primo gruppo 

    // Avvia la regione parallela con un singolo thread che crea i task
    #pragma omp parallel
    {
        #pragma omp single nowait
        { msd_rec_u32_ip(a, 0, n, first_shift, step_bits); }
    }
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
        fprintf(stderr, "Uso: %s <input.bin> [--bits=N] [--threads=N] [--repeats=N] [--out=sorted.bin]\n", argv[0]);
        return 1;
    }

    // Valori di default
    const char* in_path  = argv[1];
    int   bits           = 8;                      // 8 bit = 256 bucket
    int   num_threads    = omp_get_max_threads();  // Tutti i core disponibili
    int   repeats        = 1;
    const char* out_path = NULL;

    // Parsing degli argomenti opzionali
    for (int i=2; i<argc; ++i) {
        if (strncmp(argv[i], "--bits=", 7) == 0)        bits = atoi(argv[i] + 7);
        else if (strncmp(argv[i], "--threads=", 10) == 0) num_threads = atoi(argv[i] + 10);
        else if (strncmp(argv[i], "--repeats=", 10) == 0){ repeats = atoi(argv[i] + 10); if (repeats < 1) repeats = 1; }
        else if (strncmp(argv[i], "--out=", 6) == 0)     out_path = argv[i] + 6;
        else { fprintf(stderr,"Argomento sconosciuto: %s\n", argv[i]); return 1; }
    }

    // Validazione parametri
    if (bits < 1)  bits = 8;   // Minimo 1 bit
    if (bits > 16) bits = 16;  // Massimo 16 bit (64K bucket)


    // Caricamento input
    struct stat st;
    if (stat(in_path, &st) != 0) { perror("stat"); return 1; }
    if (st.st_size % 4 != 0) { fprintf(stderr,"Errore: size non multiplo di 4.\n"); return 1; }
    size_t n = (size_t)st.st_size / 4;  // Numero di uint32_t 

    // Alloca e carica l'array originale
    uint32_t *a0 = (uint32_t*)malloc(n * sizeof(uint32_t));
    if (!a0) { fprintf(stderr,"malloc a0\n"); return 1; }
    FILE* f = fopen(in_path, "rb"); if (!f) { perror("fopen input"); free(a0); return 1; }
    size_t rd = fread(a0, sizeof(uint32_t), n, f); fclose(f);
    if (rd != n) { fprintf(stderr,"fread incompleto\n"); free(a0); return 1; }

    // Alloca array di lavoro 
    uint32_t *a = (uint32_t*)malloc(n * sizeof(uint32_t));
    if (!a) { fprintf(stderr,"malloc a\n"); free(a0); return 1; }

   
    double sum=0.0, tmin=1e300, tmax=0.0;
    printf("n=%zu  bits=%d  repeats=%d  threads=%d\n",
           n, bits, repeats, num_threads);

    // Esegui benchmark per il numero di ripetizioni richieste
    for (int r=1; r<=repeats; ++r) {
        memcpy(a, a0, n * sizeof(uint32_t));  // Ripristina dati originali

        // Misura tempo di esecuzione
        double t0 = omp_get_wtime();
        msd_radix_sort_u32_ip_bits(a, n, bits, num_threads);
        double t1 = omp_get_wtime();

        // Aggiorna statistiche
        double dt = t1 - t0;
        sum += dt; if (dt < tmin) tmin = dt; if (dt > tmax) tmax = dt;

        // Verifica correttezza
        int ok = is_sorted_u32(a, n);
        printf("[pass %d/%d] sorted=%d  time=%.3f s\n", r, repeats, ok, dt);
        if (!ok) { fprintf(stderr,"Risultato non ordinato!\n"); free(a); free(a0); return 2; }
    }

    // Stampa statistiche finali
    printf("==> avg=%.3f s   min=%.3f s   max=%.3f s (over %d runs)\n", sum/repeats, tmin, tmax, repeats);


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
