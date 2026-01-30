// ============================================================================
// LSD Radix Sort OUT-OF-PLACE con OpenMP
// ============================================================================

// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
#include <sys/stat.h>

// ============================================================================


// Termina il programma con un messaggio di errore
static void die(const char* msg){ fprintf(stderr,"Errore: %s\n", msg); exit(1); }

// Verifica se l'array è ordinato in modo crescente
static int is_sorted_u32(const uint32_t *v, size_t n){
    for(size_t i=1;i<n;i++) if(v[i-1] > v[i]) return 0;
    return 1;
}

// Carica un file binario  uint32_t
// Ritorna il puntatore all'array allocato e imposta *out_n al numero di elementi
static uint32_t* load_file_u32(const char* path, size_t *out_n){
    struct stat st;
    if (stat(path, &st) != 0) { perror("stat"); die("impossibile leggere il file"); }
    if ((st.st_size % 4) != 0) die("size file non multipla di 4");
    size_t n = (size_t)st.st_size / 4;
    FILE* f = fopen(path, "rb"); if(!f){ perror("fopen"); die("impossibile aprire input"); }
    uint32_t *a = (uint32_t*)malloc(n*sizeof(uint32_t));
    if(!a) die("malloc input");
    size_t rd = fread(a, sizeof(uint32_t), n, f); fclose(f);
    if (rd != n) die("fread incompleto");
    *out_n = n; return a;
}

// ============================================================================
// lsd_radix_u32_omp - LSD Radix Sort parallelo out-of-place
// Parametri:
//   a -> Array da ordinare (modificato in-place)
//   n           -> Numero di elementi
//   bits        -> Bit per passata di radix (1-16)
//   num_threads -> Numero di thread OpenMP da usare

static void lsd_radix_u32_omp(uint32_t *a, size_t n, int bits, int num_threads)
{
    if(n<=1) return;
    if(bits < 1 || bits > 16) die("bits deve essere 1..16");

    omp_set_num_threads(num_threads);

    const int base = 1 << bits;              // Numero di bucket 
    const uint32_t mask = (uint32_t)(base - 1);  // Maschera per estrarre i bit

    // ========================================================================
    // calcola il numero minimo di passate necessarie basato sul valore massimo presente nell'array
  
    uint32_t maxv = a[0];
    for(size_t i=1;i<n;i++) if(a[i] > maxv) maxv = a[i];
    if (maxv == 0) return;  // Tutti zeri allora già ordinato

    int passes = 0;
    {
        // __builtin_clz = count leading zeros 
        int msb = 31 - __builtin_clz(maxv);  // Indice del bit più significativo
        passes = (msb + bits) / bits;         // Numero di passate necessarie
    }

  
    // Allocazione buffer
    
    uint32_t *out = (uint32_t*)malloc(n*sizeof(uint32_t));  // Buffer di output
    if(!out) die("malloc out");

    const int Tmax = omp_get_max_threads();
    // Istogrammi locali per ogni thread
    size_t *thread_counts  = (size_t*)malloc((size_t)Tmax * base * sizeof(size_t));
    // Offset di scrittura per ogni thread
    size_t *thread_offsets = (size_t*)malloc((size_t)Tmax * base * sizeof(size_t));
    // Conteggio globale per bucket 
    size_t *glob_count     = (size_t*)malloc((size_t)base * sizeof(size_t));
    if(!thread_counts || !thread_offsets || !glob_count) die("malloc ausiliari");

    // ========================================================================
  
    int shift = 0;
    for (int p=0; p<passes; ++p, shift += bits)
    {
        #pragma omp parallel
        {
            const int tid  = omp_get_thread_num();   // ID del thread
            const int Tloc = omp_get_num_threads();  // Numero effettivo di thread

            // ================================================================
            // Istogramma locale per thread
            // Ogni thread conta gli elementi nel proprio chunk
           
            size_t *lc = thread_counts + (size_t)tid*base;  // Puntatore all'istogramma locale
            for (int b=0;b<base;b++) lc[b]=0;  // Azzera contatori

            // Calcola il range di questo thread (divisione equa dell'array)
            size_t start = (n*(size_t)tid)/Tloc;
            size_t end   = (n*(size_t)(tid+1))/Tloc;

            // Conta gli elementi per ogni bucket
            for (size_t i=start;i<end;i++){
                unsigned d = (unsigned)((a[i] >> shift) & mask);
                lc[d]++;
            }
            #pragma omp barrier  // Attendi che tutti abbiano finito il conteggio

            
           
            #pragma omp single
            {
                //  somma gli istogrammi locali in glob_count
                for(int b=0;b<base;b++){
                    size_t s=0;
                    for(int t=0;t<Tloc;t++) s += thread_counts[(size_t)t*base + b];
                    glob_count[b]=s;
                }

                // Prefix sum esclusivo: glob_count[b] = posizione di inizio bucket b
                size_t acc=0;
                for(int b=0;b<base;b++){ size_t c=glob_count[b]; glob_count[b]=acc; acc+=c; }

                // Calcola offset di scrittura per ogni thread/bucket
                // Questo garantisce la stabilità: i thread scrivono in ordine
                for(int b=0;b<base;b++){
                    size_t aacc = glob_count[b];
                    for(int t=0;t<Tloc;t++){
                        thread_offsets[(size_t)t*base + b] = aacc;
                        aacc += thread_counts[(size_t)t*base + b];
                    }
                }
            }
            #pragma omp barrier  // Attendi che gli offset siano pronti

          
            // Ogni thread scrive i propri elementi nelle posizioni calcolate
            
            size_t *toff = thread_offsets + (size_t)tid*base;
            for (size_t i=start;i<end;i++){
                uint32_t x = a[i];
                unsigned d = (unsigned)((x >> shift) & mask);
                out[toff[d]++] = x;  // Scrivi e incrementa offset
            }
            #pragma omp barrier

   
            // Swap buffer 
            // Scambia i puntatori per la prossima passata

            #pragma omp single
            {
                uint32_t *tmp = a; a = out; out = tmp;
            }
        } // fine parallel
    } // fine ciclo passate


    // Dopo ogni passata si fa swap dei buffer. Se il numero di passate è
    // dispari, il risultato finale si trova nel buffer temporaneo in questo caso, copiamo indietro.
    if (passes % 2 != 0) {
        // a punta al buffer temp, out punta al buffer originale
        memcpy(out, a, n*sizeof(uint32_t));
        uint32_t *tmp = a; a = out; out = tmp;
        // dati ordinati nel suo buffer originale
    }

    // Libera memoria ausiliaria
    free(out);
    free(thread_counts);
    free(thread_offsets);
    free(glob_count);
}


int main(int argc, char** argv)
{
    // Verifica argomenti minimi
    if (argc < 2) {
        fprintf(stderr, "Uso: %s <input.bin> [--bits=K] [--threads=N] [--repeats=N] [--no-check]\n", argv[0]);
        return 1;
    }

    // Valori di default
    const char* in_path = argv[1];
    int bits = 8;                              // 8 bit = 256 bucket
    int num_threads = omp_get_max_threads();   // Tutti i core disponibili
    int repeats = 1;
    int do_check = 1;                          // Verifica correttezza abilitata

    // Parsing degli argomenti opzionali
    for (int i=2;i<argc;i++){
        if (strncmp(argv[i],"--bits=",7)==0)        bits = atoi(argv[i]+7);
        else if (strncmp(argv[i],"--threads=",10)==0) num_threads = atoi(argv[i]+10);
        else if (strncmp(argv[i],"--repeats=",10)==0) repeats = atoi(argv[i]+10);
        else if (strcmp(argv[i],"--no-check")==0)   do_check = 0;
        else { fprintf(stderr,"Argomento sconosciuto: %s\n", argv[i]); return 1; }
    }

    // Validazione parametri
    if (bits<1 || bits>16) die("bits deve essere 1..16");
    if (repeats<1) repeats=1;

    // ========================================================================
    // Caricamento input

    size_t n=0;
    uint32_t *a0 = load_file_u32(in_path, &n);  // Array originale

    printf("n=%zu  bits=%d  repeats=%d  threads=%d\n",
           n, bits, repeats, num_threads);

    // ========================================================================
    // Benchmark

    double sum=0.0, tmin=1e300, tmax=0.0;

    uint32_t *a = (uint32_t*)malloc(n*sizeof(uint32_t));  // Array di lavoro
    if(!a) die("malloc a");

    // Esegui benchmark per il numero di ripetizioni richieste
    for (int r=1; r<=repeats; ++r){
        memcpy(a, a0, n*sizeof(uint32_t));  // Ripristina dati originali

        // Misura tempo di esecuzione
        double t0 = omp_get_wtime();
        lsd_radix_u32_omp(a, n, bits, num_threads);
        double t1 = omp_get_wtime();

        // Aggiorna statistiche
        double dt = t1 - t0;
        sum += dt; if (dt<tmin) tmin=dt; if (dt>tmax) tmax=dt;

        // Verifica correttezza 
        int ok = do_check ? is_sorted_u32(a, n) : 1;
        printf("[pass %d/%d] sorted=%d  time=%.3f s\n", r, repeats, ok, dt);
        if (do_check && !ok){ fprintf(stderr,"Risultato non ordinato!\n"); free(a0); free(a); return 2; }
    }

    // Stampa statistiche finali
    printf("==> avg=%.3f s   min=%.3f s   max=%.3f s (over %d runs)\n",
           sum/repeats, tmin, tmax, repeats);

    // Libera memoria
    free(a0); free(a);
    return 0;
}
