// ============================================================================
// PARADIS — MSD Radix Sort ricorsivo parallelo in-place
// Implementazione basata su https://github.com/albicilla/simple_paradis
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
#include <sys/stat.h>
#include <math.h>

// Numero massimo di thread supportati 
#define MaxThreadNum 224
// Numero massimo di bucket (2^8 = 256 per radix a 8 bit)
#define MaxKisuu 256

// Funzione di utilità per terminare con errore
static void die(const char* msg){ fprintf(stderr,"Errore: %s\n", msg); exit(1); }

// Verifica se l'array è ordinato in modo crescente
// Ritorna 1 se ordinato, 0 altrimenti
static int is_sorted_u32(const uint32_t *v, size_t n){
    for(size_t i=1;i<n;i++) if(v[i-1] > v[i]) return 0;
    return 1;
}

// Carica un file binario contenente uint32_t
// Il file deve avere dimensione multipla di 4 byte
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


static const int kRadixBits = 8;                      // Bit per cifra (1 byte)
static const int kRadixMask = (1 << kRadixBits) - 1;  // Maschera 0xFF per estrarre un byte
static const int kRadixBin = 1 << kRadixBits;         // Numero di bucket = 256

// Estrae il byte alla posizione stage dal numero
// stage=0 -> byte meno significativo, stage=3 -> byte più significativo
inline int determineDigitBucket(int stage, uint32_t num){
    return ((num >> (8 * stage)) & kRadixMask);
}

// Scambia due valori uint32_t
inline void _swap(uint32_t *a, uint32_t *b){
    uint32_t temp = *b;
    *b = *a;
    *a = temp;
}

// Insertion sort per piccoli array (usato quando il bucket ha <= 64 elementi)
// Più efficiente del radix sort per array molto piccoli
// s = puntatore all'inizio, e = puntatore alla fine (esclusa)
void insert_sort_u32(uint32_t *s, uint32_t *e){
    for (uint32_t *i = s + 1; i < e; ++i) {
        if (*i < *(i - 1)) {
            uint32_t *j;
            uint32_t tmp = *i;
            *i = *(i - 1);
            // Sposta gli elementi maggiori verso destra
            for (j = i - 1; j > s && tmp < *(j - 1); --j) {
                *j = *(j - 1);
            }
            *j = tmp;  // Inserisce l'elemento nella posizione corretta
        }
    }
}

// Parametri:
//   kth_byte  Indice del byte su cui ordinare (3=MSB, 0=LSB)
//   s  Puntatore all'inizio del segmento da ordinare
//   t  Puntatore alla fine del segmento (esclusa)
//   begin_itr Puntatore all'inizio dell'array completo (per calcolare offset)
//   processes Numero di thread da utilizzare

void PARADIS_core(int kth_byte, uint32_t *s, uint32_t *t, uint32_t *begin_itr, int processes){
    long long cnt[MaxKisuu] = {0};  // Conteggio globale per ogni bucket

    long long elenum = t - s;        // Numero di elementi da ordinare
    long long start = s - begin_itr; // Offset dall'inizio dell'array

    long long part = elenum / processes;  // Elementi per thread 

    // Strutture dati per la parallelizzazione:
    long long localHists[MaxThreadNum][MaxKisuu];  // Istogrammi locali per thread
    long long gh[MaxKisuu], gt[MaxKisuu], starts[MaxKisuu];  // Range globali dei bucket
    long long ph[MaxThreadNum][MaxKisuu];  // Head pointer per thread/bucket
    long long pt[MaxThreadNum][MaxKisuu];  // Tail pointer per thread/bucket

    long long SumCi = elenum;      // Somma elementi ancora da sistemare
    long long pfp[processes + 1];  // Partizioni per la fase di repair
    int var_p = processes;

    #pragma omp parallel num_threads(processes)
    {
        int th = omp_get_thread_num();  // ID del thread corrente

        // Ogni thread conta quanti elementi appartengono a ciascun bucket
        #pragma omp for
        for(int i = 0; i < kRadixBin; i++){
            for(int t = 0; t < processes; t++)
                localHists[t][i] = 0;  // Inizializza tutti gli istogrammi a 0
        }

        #pragma omp barrier  // Sincronizza prima di iniziare il conteggio

        // Ogni thread conta i propri elementi 
        #pragma omp for
        for(long long i = start; i < start + elenum; i++){
            int digit = determineDigitBucket(kth_byte, *(begin_itr + i));
            localHists[th][digit]++;  // Incrementa il contatore locale
        }

        #pragma omp barrier

        // Somma gli istogrammi locali per ottenere il conteggio globale

        #pragma omp for
        for(int i = 0; i < kRadixBin; i++){
            for(int j = 0; j < processes; j++){
                cnt[i] += localHists[j][i];
            }
        }

        #pragma omp barrier

        // Calcolo dei range dei bucket
 
        #pragma omp single
        {
            gh[0] = start;
            gt[0] = gh[0] + cnt[0];
            starts[0] = gh[0];
        }

        // Prefix sum per calcolare le posizioni di tutti i bucket
        #pragma omp single
        for(int i = 1; i < kRadixBin; i++){
            gh[i] = gh[i - 1] + cnt[i - 1];  // Inizio = fine del precedente
            gt[i] = gh[i] + cnt[i];           // Fine = inizio + conteggio
            starts[i] = gh[i];                // Salva posizione iniziale
        }

        #pragma omp barrier

     
        // Ogni thread lavora su una porzione di ogni bucket, spostando gli elementi nelle loro posizioni corrette attraverso catene di swap.

        while(SumCi != 0){  // Continua finché ci sono elementi da sistemare
            #pragma omp for
            for(int ii = 0; ii < processes; ii++){
                int pID = omp_get_thread_num();

                // Calcola il range di lavoro per questo thread in ogni bucket
                // Divide ogni bucket in porzioni uguali tra i thread
                for(int i = 0; i < kRadixBin; i++){
                    long long part = (long long)(gt[i] - gh[i]) / (long long)var_p;
                    long long res = (long long)(gt[i] - gh[i]) % (long long)(var_p);

                    if(pID < var_p - 1){
                        ph[pID][i] = part * pID + gh[i];      // Inizio porzione
                        pt[pID][i] = part * (pID + 1LL) + gh[i];  // Fine porzione
                    } else {
                        // L'ultimo thread prende anche il resto della divisione
                        ph[pID][i] = part * pID + gh[i];
                        pt[pID][i] = part * (pID + 1LL) + gh[i] + res;
                    }
                }

                // Permutazione con catene di swap
                // Per ogni bucket, sposta gli elementi fuori posto
                for(int i = 0; i < kRadixBin; i++){
                    long long head = ph[pID][i];

                    while(head < pt[pID][i]){
                        uint32_t v = *(begin_itr + head);  // Prendi elemento corrente
                        int k = determineDigitBucket(kth_byte, v);  // Trova il suo bucket

                        // Catena di swap: segui gli elementi fuori posto
                        // finché non trovi uno che appartiene al bucket corrente
                        while(k != i && ph[pID][k] < pt[pID][k]){
                            _swap(&v, (begin_itr + (int)ph[pID][k]));
                            ph[pID][k]++;
                            k = determineDigitBucket(kth_byte, v);
                        }

                        if(k == i){
                            // L'elemento appartiene a questo bucket
                            *(begin_itr + head) = *(begin_itr + ph[pID][i]);
                            head++;
                            *(begin_itr + ph[pID][i]) = v;
                            ph[pID][i]++;
                        } else {
                            // Non possiamo sistemare l'elemento ora, lascialo
                            *(begin_itr + head) = v;
                            head++;
                        }
                    }
                }
            }

            #pragma omp barrier

            // Calcola partizioni per la fase di riparazione
        
            #pragma omp single
            {
                SumCi = 0;
                long long pfpN = kRadixBin / var_p;   // Bucket per thread
                long long pfpM = kRadixBin % var_p;   // Bucket extra da distribuire
                pfp[0] = 0LL;
                long long pfpMR = 0LL;
                // Distribuisce i bucket extra ai primi thread
                for(long long i = 1LL; i < var_p + 1LL; i++){
                    if(pfpMR < pfpM) pfpMR++;
                    pfp[i] = i * pfpN + pfpMR;
                }
            }

            #pragma omp barrier

            // Questa fase corregge questi elementi scambiandoli con elementi corretti dalla coda del bucket.

            #pragma omp for
            for(int k = 0; k < processes; k++){
                for(long long i = pfp[k]; i < pfp[k + 1]; i++){
                    long long tail = gt[i];  // Fine del bucket

                    // Controlla tutte le porzioni di tutti i thread
                    for(int pID = 0; pID < processes; pID++){
                        long long head = ph[pID][i];

                        while(head < pt[pID][i] && head < tail){
                            uint32_t v = *(begin_itr + head);
                            head++;

                            // Se l'elemento non appartiene a questo bucket
                            if(determineDigitBucket(kth_byte, v) != i){
                                // Cerca dalla coda un elemento che appartiene qui
                                while(head <= tail){
                                    tail--;
                                    uint32_t w = *(begin_itr + tail);
                                    if(determineDigitBucket(kth_byte, w) == i){
                                        // Scambia: metti w al posto di v
                                        *(begin_itr + (head - 1)) = w;
                                        *(begin_itr + tail) = v;
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    gh[i] = tail;  // Aggiorna la testa del bucket
                }
            }

            #pragma omp barrier

            // Calcola quanti elementi sono ancora fuori posto
            #pragma omp single
            {
                SumCi = 0;
                for(int i = 0; i < kRadixBin; i++){
                    SumCi += (gt[i] - gh[i]);  // Elementi rimanenti da sistemare
                }
            }

            #pragma omp barrier
        }  // Fine while(SumCi != 0)
    }  // Fine regione parallela


    // Dopo aver partizionato per il byte corrente, ordina ricorsivamente ogni bucket per il byte successivo (meno significativo).
 
    if(kth_byte > 0){  // Se ci sono altri byte da processare
        #pragma omp parallel num_threads(processes)
        #pragma omp single
        {
            for(int i = 0; i < kRadixBin; i++){
                // Calcola quanti thread assegnare al sotto-problema
                // Proporzionale alla dimensione del bucket 
                int nextStageThreads = 1;
                nextStageThreads = processes * (cnt[i] * (log(cnt[i]) / log(kRadixBin)) /
                                                (elenum * (log(elenum) / log(kRadixBin))));

                if(cnt[i] > 64LL){
                    // Bucket grande: usa ricorsione parallela con task
                    #pragma omp task
                    PARADIS_core(kth_byte - 1, begin_itr + starts[i],
                                begin_itr + (starts[i] + cnt[i]), begin_itr,
                                (nextStageThreads > 1 ? nextStageThreads : 1));
                } else if(cnt[i] > 1){
                    // Bucket piccolo (<64 elementi): usa insertion sort
                    insert_sort_u32(begin_itr + starts[i], begin_itr + (starts[i] + cnt[i]));
                }
                // Bucket con 0 o 1 elementi: già ordinato, non fare nulla
            }
            #pragma omp taskwait  // Attendi completamento di tutti i task
        }
    }
}


// Parametri:
//   s         - Puntatore all'inizio dell'array
//   t         - Puntatore alla fine dell'array (esclusa)
//   threadNum - Numero di thread da utilizzare
//
// Ordina l'array in-place usando il radix sort parallelo PARADIS

void PARADIS(uint32_t *s, uint32_t *t, int threadNum){
    omp_set_nested(1);  // Abilita parallelismo annidato per la ricorsione
    PARADIS_core(3, s, t, s, threadNum);  // 3 = byte più significativo di uint32_t
}


int main(int argc, char** argv){
    // Verifica argomenti minimi
    if (argc < 2) {
        fprintf(stderr, "Uso: %s <input.bin> [--threads=N] [--repeats=N] [--no-check]\n", argv[0]);
        return 1;
    }

    // Valori di default
    const char* in_path = argv[1];
    int threadNum = omp_get_max_threads();  // Usa tutti i core disponibili
    int repeats = 1, do_check = 1;

    // Parsing degli argomenti opzionali
    for (int i = 2; i < argc; i++){
        if (strncmp(argv[i], "--threads=", 10) == 0)      threadNum = atoi(argv[i] + 10);
        else if (strncmp(argv[i], "--repeats=", 10) == 0) repeats = atoi(argv[i] + 10);
        else if (strcmp(argv[i], "--no-check") == 0)      do_check = 0;
        else { fprintf(stderr, "Argomento sconosciuto: %s\n", argv[i]); return 1; }
    }

    // Validazione parametri
    if (threadNum > MaxThreadNum) {
        fprintf(stderr, "threads deve essere <= %d\n", MaxThreadNum);
        return 1;
    }
    if (repeats < 1) repeats = 1;

    // Carica i dati dal file binario
    size_t n = 0;
    uint32_t *a0 = load_file_u32(in_path, &n);  // Array originale 

    printf("n=%zu  repeats=%d  threads=%d\n", n, repeats, threadNum);

    // Variabili per le statistiche di benchmark
    double sum = 0.0, tmin = 1e300, tmax = 0.0;
    uint32_t *a = (uint32_t*)malloc(n * sizeof(uint32_t));  // Array di lavoro
    if(!a) die("malloc a");

    // Esegue il benchmark per il numero di ripetizioni richieste
    for (int r = 1; r <= repeats; ++r){
        memcpy(a, a0, n * sizeof(uint32_t));  // Ripristina i dati originali

        // Misura il tempo di esecuzione
        double t0 = omp_get_wtime();
        PARADIS(a, a + n, threadNum);
        double t1 = omp_get_wtime();

        // Aggiorna statistiche
        double dt = t1 - t0;
        sum += dt;
        if (dt < tmin) tmin = dt;
        if (dt > tmax) tmax = dt;

        // Verifica correttezza (opzionale)
        int ok = do_check ? is_sorted_u32(a, n) : 1;
        printf("[pass %d/%d] sorted=%d  time=%.3f s\n", r, repeats, ok, dt);

        if (do_check && !ok){
            fprintf(stderr, "Risultato non ordinato!\n");
            free(a0);
            free(a);
            return 2;  // Errore: ordinamento fallito
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
