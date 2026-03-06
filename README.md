# Analisi VOCs (PTR-MS)

Pipeline standalone per analisi VOC da dati PTR-MS gia preprocessati: split temporale, mediane per blocco, rimozione anomalie, aggregazione repliche, sottrazione blank, confronto tesi e PCA.

## Struttura progetto

```text
Analisi VOCs/
  voc_analysis_pipeline.py
  ptrms_*.py
  H0/
    Run2/
    Run4/
    Run5/
    Run6/
  H1/
    Run2/
    Run4/
    Run5/
    Run6/
  Blank/
  config/
    run_schedule.template.csv
  output/
```

## Come caricare i VOC PTR-MS

### 1) Dati per tesi (generalizzato)

Carica i file esportati dal tuo flusso PTR-MS in cartelle separate per tesi:

- `H0/`
- `H1/`
- `Blank/`

Ogni tesi puo contenere:

- file direttamente nella cartella, oppure
- sottocartelle per run (`Run2`, `Run4`, ecc.)

Formati supportati: `.txt` o `.csv` (esportazioni PTR-MS gia preprocessate, ad esempio `stanza_01.h5_export.txt`).

Naming consigliato:

- `stanza_01.h5_export.txt`
- `stanza_02.h5_export.txt`

### 2) File configurazione orari camere / schedule run

La pipeline supporta un file opzionale in:

- `config/run_schedule.xlsx` (default)
- oppure `config/run_schedule.csv`

Colonne consigliate (alias accettati nel codice):

- `run_id`
- `start_datetime` (inizio reale run)
- `split_hours` (offset ore per split H0/H1 -> H2/H3)
- `acclim_hours` (durata fase acclimatazione)
- `transition_exclude_hours` (finestra esclusione attorno al cambio)

Questo rende la pipeline riusabile anche fuori dal tuo caso specifico.
Se il file manca, vengono usati i default hardcoded in `voc_analysis_pipeline.py`.

Template pronto: `config/run_schedule.template.csv`.

### 3) File composti monitorati (opzionale)

Puoi fornire `composti da monitorare.xlsx` (prima colonna = m/z) per limitare i plot a masse di interesse.
Se assente, la pipeline usa tutte le masse disponibili.

## Setup

```bash
pip install -r requirements.txt
```

## Esecuzione

```bash
python voc_analysis_pipeline.py
```

## Output principali

- `output/01_comparison_raw_median/`
- `output/02_anomaly_removal/`
- `output/03_aggregation/`
- `output/04_blank_removal/`
- `output/05_final_comparison/`
- `output/06_pca_analysis/`
- `output/comparison_metrics.xlsx`

## Nota portfolio/privacy

I file grezzi e i dataset pesanti sono stati rimossi dal repository.
Questa repo contiene codice, struttura cartelle e template di caricamento.
## Research Profile

- Research keywords: PTR-MS, VOC time series, anomaly removal, blank subtraction, PCA, reproducible analytics.
- Positioning: applied research project for experimental chemical sensing data analysis.
- Open-source status: this repository is open source and intended for reproducible research and education.

