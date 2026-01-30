// ============================================================================
// LSD Radix Sort IN-PLACE con OpenMP
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
#include <sys/stat.h>



// Termina con un messaggio di errore
static void die(const char* msg){ fprintf(stderr,"Errore: %s\n", msg); exit(1); }

// Verifica l'array è ordinato in modo crescente
static int is_sorted_u32(const uint32_t *v, size_t n){
    for(size_t i=1;i<n;i++) if(v[i-1] > v[i]) return 0;
    return 1;
}

// Carica un file binario uint32_t
// Ritorna il puntatore all'array allocato e imposta numero di elementi
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

// ============================================================================
// Parametri:
//   a    - Array da ordinare 
//   n    - Numero di elementi
//   bits - Bit per passata di radix 


static void lsd_radix_u32_inplace(uint32_t *a, size_t n, int bits)
{
    if(n<=1) return;
    if(bits < 1 || bits > 16) die("bits deve essere 1..16");

    const int base = 1 << bits;              // Numero di bucket (es. 256 per 8 bit)
    const uint32_t mask = (uint32_t)(base - 1);  // Maschera per estrarre i bit


   
    uint32_t maxv = a[0];
    #pragma omp parallel for reduction(max:maxv)
    for(size_t i=1;i<n;i++) if(a[i] > maxv) maxv = a[i];

    if (maxv == 0) return;  // Tutti zeri: già ordinato
    int passes = 0;
    {
        // __builtin_clz = count leading zeros
        int msb = 31 - __builtin_clz(maxv);  // Indice del bit più significativo
        passes = (msb + bits) / bits;         // Numero di passate necessarie
    }


    // Allocazione buffer ausiliari

    size_t *count = (size_t*)malloc((size_t)base * sizeof(size_t));  // Conteggio globale
    size_t *start = (size_t*)malloc((size_t)base * sizeof(size_t));  // Posizione inizio bucket
    if(!count || !start) die("malloc ausiliari");

    const int Tmax = omp_get_max_threads();
    // Istogrammi locali per ogni thread
    size_t *thread_counts = (size_t*)calloc((size_t)Tmax * base, sizeof(size_t));
    if(!thread_counts) die("malloc thread_counts");

 
    int shift = 0;
    for (int p=0; p<passes; ++p, shift += bits)
    {
        // Ogni thread conta gli elementi nel proprio chunk
     
        #pragma omp parallel
        {
            const int tid  = omp_get_thread_num();
            const int Tloc = omp_get_num_threads();
            size_t *lc = thread_counts + (size_t)tid*base;  // Istogramma locale

            // Azzera contatori 
            for(int b=0;b<base;b++) lc[b]=0;

            // Calcola range del thread
            size_t s = (n*(size_t)tid)/Tloc;
            size_t e = (n*(size_t)(tid+1))/Tloc;

            // Conta elementi bucket
            for(size_t i=s;i<e;i++){
                unsigned d = (unsigned)((a[i] >> shift) & mask);
                lc[d]++;
            }
        }

        // Somma gli istogrammi locali in count[] global
        for(int b=0;b<base;b++){
            size_t sum=0;
            for(int t=0;t<Tmax;t++) sum += thread_counts[(size_t)t*base + b];
            count[b]=sum;
        }

        // Prefix sum esclusivo
        size_t acc=0;
        for(int b=0;b<base;b++){
            start[b]=acc;
            acc+=count[b];
        }

        // Permutazione in-place (American Flag Sort)

        // Usa cicli di permutazione per spostare gli elementi nelleposizioni corrette senza array ausiliario.
      
        size_t *head = (size_t*)malloc((size_t)base * sizeof(size_t));
        size_t *tail = (size_t*)malloc((size_t)base * sizeof(size_t));
        if(!head || !tail) die("malloc head/tail");

        // Inizializza head e tail per ogni bucket
        for(int b=0;b<base;b++){
            head[b] = start[b];                              // Inizio bucket
            tail[b] = (b+1 < base) ? start[b+1] : n;         // Fine bucket
        }

        // Processa ogni bucket
        for(int b=0; b<base; b++){
            while(head[b] < tail[b]){
                // Prendi l'elemento nella posizione corrente
                uint32_t elem = a[head[b]];
                unsigned digit = (unsigned)((elem >> shift) & mask);

                if((int)digit == b){
                    // Elemento già nel bucket corretto avanza
                    head[b]++;
                }
                else{
                    // Ciclo di permutazione
                    while((int)digit != b){
                        size_t swap_pos = head[digit];  // Dove va l'elemento

                        // Scambia: metti elem nella sua posizione, prendi quello che c'era
                        uint32_t temp = a[swap_pos];
                        a[swap_pos] = elem;
                        head[digit]++;  // Avanza il puntatore del bucket destinazione

                        // Continua con l'elemento appena prelevato
                        elem = temp;
                        digit = (unsigned)((elem >> shift) & mask);
                    }
                    // Il ciclo si è chiuso: elem appartiene al bucket b
                    a[head[b]] = elem;
                    head[b]++;
                }
            }
        }

        free(head);
        free(tail);
    } // fine ciclo passate

    // Libera memoria ausiliaria
    free(thread_counts);
    free(count);
    free(start);
}


int main(int argc, char** argv)
{
    // Verifica argomenti minimi
    if (argc < 2) {
        fprintf(stderr, "Uso: %s <input.bin> [--bits=K] [--repeats=N] [--no-check]\n", argv[0]);
        return 1;
    }

    // Valori di default
    const char* in_path = argv[1];
    int bits = 8, repeats = 1, do_check = 1;

    // Parsing degli argomenti opzionali
    for (int i=2;i<argc;i++){
        if (strncmp(argv[i],"--bits=",7)==0)          bits = atoi(argv[i]+7);
        else if (strncmp(argv[i],"--repeats=",10)==0) repeats = atoi(argv[i]+10);
        else if (strcmp(argv[i],"--no-check")==0)     do_check = 0;
        else { fprintf(stderr,"Argomento sconosciuto: %s\n", argv[i]); return 1; }
    }

    // Validazione parametri
    if (bits<1 || bits>16) { fprintf(stderr,"bits deve essere 1..16\n"); return 1; }
    if (repeats<1) repeats=1;

    // ========================================================================
    // Caricamento input

    size_t n=0;
    uint32_t *a0 = load_file_u32(in_path, &n);  // Array originale

    printf("n=%zu  bits=%d  repeats=%d  OMP_NUM_THREADS=%d\n",
           n, bits, repeats, omp_get_max_threads());


    double sum=0.0, tmin=1e300, tmax=0.0;
    uint32_t *a = (uint32_t*)malloc(n*sizeof(uint32_t));  // Array di lavoro
    if(!a) die("malloc a");

    // Esegui benchmark per il numero di ripetizioni richieste
    for (int r=1; r<=repeats; ++r){
        memcpy(a, a0, n*sizeof(uint32_t));  // Ripristina dati originali

        // Misura tempo di esecuzione
        double t0 = omp_get_wtime();
        lsd_radix_u32_inplace(a, n, bits);
        double t1 = omp_get_wtime();

        // Aggiorna statistiche
        double dt = t1 - t0;
        sum += dt; if (dt<tmin) tmin=dt; if (dt>tmax) tmax=dt;

        // Verifica correttezza 
        int ok = do_check ? is_sorted_u32(a, n) : 1;
        printf("[pass %d/%d] sorted=%d  time=%.3f s\n", r, repeats, ok, dt);
        if (do_check && !ok){
            fprintf(stderr,"Risultato non ordinato!\n");
            free(a0); free(a);
            return 2;
        }
    }

    // Stampa statistiche finali
    printf("==> avg=%.3f s   min=%.3f s   max=%.3f s (over %d runs)\n",
           sum/repeats, tmin, tmax, repeats);

    // Libera memoria
    free(a0); free(a);
    return 0;
}
