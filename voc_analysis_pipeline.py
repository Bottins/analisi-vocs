#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VOC Analysis Pipeline - Standalone Script
==========================================

Pipeline completa per l'analisi dei dati VOC (Volatile Organic Compounds)
da misurazioni PTR-MS.

Funzionalita:
- Caricamento dati da cartelle organizzate per run/stanza
- Split temporale per giorni (pre/post trattamento)
- Calcolo mediane per blocco temporale (con rimozione 20% iniziale/finale)
- Rimozione anomalie/salti
- Aggregazione repliche con intervalli di confidenza
- Sottrazione del bianco
- Confronto tra tesi con metriche statistiche
- Export Excel e grafici

Autore: PHD-VOCS Project
"""

# =============================================================================
# IMPORT LIBRERIE STANDARD
# =============================================================================
# Librerie standard Python per gestione file, regex, logging e path
import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing

# =============================================================================
# IMPORT LIBRERIE SCIENTIFICHE
# =============================================================================
# NumPy: operazioni numeriche e array
import numpy as np
# Pandas: manipolazione dati tabulari
import pandas as pd
# Matplotlib: creazione grafici (backend non-interattivo per performance)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
# SciPy: funzioni statistiche avanzate
from scipy import stats

# =============================================================================
# IMPORT MODULI LOCALI PTR-MS
# =============================================================================
# Modulo per parsing e caricamento file CSV/TXT PTR-MS
from ptrms_parser import load_and_prepare
# Modulo per rilevamento blocchi temporali nelle misurazioni
from ptrms_blocks import detect_blocks, get_block_info
# Modulo per pulizia dati (trimming, spike detection)
from ptrms_cleaning import clean_block
# Modulo per calcolo statistiche (mediane per blocco)
from ptrms_statistics import compute_file_medians
# Modulo per analisi avanzata (sottrazione ambiente, aggregazione, confronti)
from ptrms_analysis import (
    normalize_mass_columns,
    subtract_environment,
    aggregate_replicates,
    compare_treatments
)
# Modulo con utility comuni (estrazione masse, statistiche sicure, etc.)
from ptrms_common import (
    get_mass_columns,
    extract_mass_value,
    safe_mean,
    safe_std,
    safe_median,
    to_python_type,
    METADATA_COLUMNS
)
# Modulo PCA: analisi componenti principali
from ptrms_pca import run_all_pca

# =============================================================================
# CONFIGURAZIONE LOGGING
# =============================================================================
# Configura il sistema di logging per tracciare l'esecuzione della pipeline
# Livello INFO: mostra messaggi informativi, warning e errori
# Formato: timestamp - livello - messaggio
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# COSTANTI GLOBALI E CONFIGURAZIONE
# =============================================================================
# Tolleranza per il matching delle masse tra file diversi.
# Il macchinario PTR-MS ha precisione 0.01 e questa viene usata come
# tolleranza di fusione iniziale tra masse quasi-identiche.
# Esempio: m59.048 e m59.052 vengono considerate la stessa massa.
MASS_TOLERANCE = 0.01

# Costanti temporali per conversioni
SECONDS_PER_DAY = 86400  # Numero di secondi in un giorno (24 * 60 * 60)
HOURS_PER_DAY = 24       # Ore in un giorno

# Percentuale di dati da rimuovere all'inizio e alla fine di ogni blocco.
# Questo elimina i dati transitori dovuti all'attacco e allo stacco del
# macchinario di rilevazione, che non sono rappresentativi delle misurazioni.
TRIM_PERCENT = 0.20  # 20% iniziale + 20% finale = 40% totale rimosso

# Offset temporali (in ore) rispetto all'inizio della registrazione per
# separare H0/H1 da H2/H3, specifici per run.
# Valori derivati dal file orari.xlsx (giorno 0 -> giorno 4).
RUN_SPLIT_HOURS = {
    2: 91.75,      # 24/07 12:32 -> 28/07 08:17
    4: 90.6922,    # 14/08 14:30:08 -> 18/08 09:11:40
    5: 92.9167,    # 21/08 12:17 -> 25/08 09:12
    6: 92.6667,    # 28/08 13:04 -> 01/09 09:44
}

# Offset temporali (in ore) per separare la fase di acclimatizzazione
# (giorno 0 -> giorno 1) dalla fase post-acclimatizzazione.
RUN_ACCLIM_HOURS = {
    2: 21.3667,    # 24/07 12:32 -> 25/07 09:54
    4: 19.8664,    # 14/08 14:30:08 -> 15/08 10:22:07
    5: 22.6500,    # 21/08 12:17 -> 22/08 10:56
    6: 22.0500,    # 28/08 13:04 -> 29/08 11:07
}

# Ore da rimuovere attorno al cambio H1 -> H2/H3
TRANSITION_EXCLUDE_HOURS = 1.0

# Datetime reali di inizio registrazione per ogni run.
# Usati dalla PCA per mappare RelTime (secondi) a orario reale del giorno
# e assegnare le fasce orarie (Mattina/Pomeriggio/Notte).
# Formato: datetime(anno, mese, giorno, ora, minuto)
# Fonte: file orari.xlsx
RUN_START_DATETIMES = {
    2: datetime(2024, 7, 24, 12, 32),    # Run2: 24/07/2024 12:32
    4: datetime(2024, 8, 14, 14, 30),    # Run4: 14/08/2024 14:30
    5: datetime(2024, 8, 21, 12, 17),    # Run5: 21/08/2024 12:17
    6: datetime(2024, 8, 28, 13, 4),     # Run6: 28/08/2024 13:04
}



# File opzionale con configurazione scheduling run (generalizzabile).
# Se presente, sovrascrive i default hardcoded sopra.
RUN_CONFIG_PATH = Path("config/run_schedule.xlsx")


def _find_schedule_column(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    """Trova la colonna cercando alias robusti rispetto a maiuscole/spazi/simboli."""
    normalized = {re.sub(r"[^a-z0-9]+", "", str(c).lower()): c for c in df.columns}
    for alias in aliases:
        key = re.sub(r"[^a-z0-9]+", "", alias.lower())
        if key in normalized:
            return normalized[key]
    return None


def load_run_schedule_config(
    excel_path: Union[str, Path]
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, datetime], Optional[float]]:
    """
    Carica configurazione run da Excel (opzionale).

    Colonne supportate (alias):
    - run_id / run
    - split_hours
    - acclim_hours
    - start_datetime
    - transition_exclude_hours (valore globale, prende il primo valido)
    """
    path = Path(excel_path)
    if not path.exists():
        logger.info(f"Config run non trovata ({path}), uso default hardcoded.")
        return {}, {}, {}, None

    try:
        if path.suffix.lower() == '.csv':
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
    except Exception as exc:
        logger.warning(f"Impossibile leggere config run {path}: {exc}. Uso default.")
        return {}, {}, {}, None

    if df.empty:
        logger.warning(f"Config run vuota ({path}). Uso default.")
        return {}, {}, {}, None

    run_col = _find_schedule_column(df, ["run_id", "run", "run_number", "run_num"])
    split_col = _find_schedule_column(df, ["split_hours", "split_h", "h0h1_to_h2h3_hours"])
    acclim_col = _find_schedule_column(df, ["acclim_hours", "acclimation_hours", "acclim_h"])
    start_col = _find_schedule_column(df, ["start_datetime", "run_start", "start_time"])
    transition_col = _find_schedule_column(
        df, ["transition_exclude_hours", "transition_hours", "exclude_hours"]
    )

    if run_col is None:
        logger.warning(f"Config run {path}: colonna run non trovata. Uso default.")
        return {}, {}, {}, None

    split_hours: Dict[int, float] = {}
    acclim_hours: Dict[int, float] = {}
    start_datetimes: Dict[int, datetime] = {}

    for _, row in df.iterrows():
        run_val = pd.to_numeric(row.get(run_col), errors="coerce")
        if pd.isna(run_val):
            continue
        run_id = int(run_val)

        if split_col is not None:
            v = pd.to_numeric(row.get(split_col), errors="coerce")
            if pd.notna(v):
                split_hours[run_id] = float(v)

        if acclim_col is not None:
            v = pd.to_numeric(row.get(acclim_col), errors="coerce")
            if pd.notna(v):
                acclim_hours[run_id] = float(v)

        if start_col is not None:
            v = pd.to_datetime(row.get(start_col), errors="coerce")
            if pd.notna(v):
                start_datetimes[run_id] = v.to_pydatetime()

    transition_exclude: Optional[float] = None
    if transition_col is not None:
        vals = pd.to_numeric(df[transition_col], errors="coerce").dropna()
        if not vals.empty:
            transition_exclude = float(vals.iloc[0])

    logger.info(
        "Config run caricata: "
        f"split={len(split_hours)}, acclim={len(acclim_hours)}, start={len(start_datetimes)}"
    )
    return split_hours, acclim_hours, start_datetimes, transition_exclude

# =============================================================================
# CARICAMENTO MASSE DA MONITORARE (PER PLOT)
# =============================================================================
def load_plot_masses(
    excel_path: Union[str, Path],
    available_masses: set,
    tolerance: float = 0.05
) -> Tuple[List[str], Dict[str, str]]:
    """
    Carica le masse da monitorare da un file Excel e le matcha con
    le masse effettivamente presenti nei dati tramite prossimitÃ .

    L'Excel contiene valori m/z ad alta precisione (es. 31.018, 59.0497)
    che vengono arrotondati e matchati con i nomi colonna mXX.XX della pipeline.

    Args:
        excel_path: Percorso al file Excel con colonna 'Massa Protonata (m/z)'
        available_masses: Set di masse disponibili nei dati (es. {'m31.02', 'm59.05', ...})
        tolerance: Tolleranza massima per il matching (default 0.05 Da)

    Returns:
        Tuple di:
        - Lista ordinata di nomi massa matchati (formato mXX.XX)
        - Dizionario massa->nome composto (es. {'m59.05': 'Acetone (C3H6O, 59.0497)'})
    """
    excel_path = Path(excel_path)
    if not excel_path.exists():
        logger.warning(f"File composti non trovato: {excel_path}, uso tutte le masse")
        return sorted(available_masses), {}

    df = pd.read_excel(excel_path)
    col_mz = df.columns[0]  # Prima colonna = Massa Protonata (m/z)

    # Colonne opzionali per nome composto e formula
    col_formula = df.columns[1] if len(df.columns) > 1 else None
    col_composto = df.columns[2] if len(df.columns) > 2 else None

    # Estrai valori numerici dalle masse disponibili: "m59.05" -> 59.05
    avail_values = {}
    for m in available_masses:
        val = extract_mass_value(m)
        if val is not None:
            avail_values[val] = m

    avail_nums = sorted(avail_values.keys())

    matched = []
    mass_to_compound = {}  # Mapping massa_colonna -> nome composto

    for row_idx, raw_val in enumerate(df[col_mz]):
        # Converti il valore Excel in float m/z reale
        # L'Excel puÃ² avere interi (es. 31018 per 31.018) o float
        s = str(raw_val).replace('.', '').replace(',', '')  # Rimuovi separatori migliaia
        # Prova a piazzare il punto decimale per ottenere un m/z nel range PTR-MS [14, 300]
        # Strategia: preferisci il valore m/z piÃ¹ grande nel range (tipico PTR-MS),
        # ma se un candidato piÃ¹ piccolo ha un match molto migliore, usa quello.
        candidates = []
        for pos in range(1, len(s)):
            try:
                candidate = float(s[:pos] + '.' + s[pos:])
            except ValueError:
                continue
            if 14 <= candidate <= 300:
                candidates.append(candidate)

        if not candidates:
            logger.warning(f"  Massa Excel {raw_val}: impossibile determinare m/z, skip")
            continue

        # Per ogni candidato, trova il suo best match e la distanza
        candidate_matches = []
        for target_mz in candidates:
            target_rounded = round(target_mz, 2)
            best_dist_for_cand = float('inf')
            best_match_for_cand = None
            for av in avail_nums:
                dist = abs(av - target_rounded)
                if dist < best_dist_for_cand:
                    best_dist_for_cand = dist
                    best_match_for_cand = av
            candidate_matches.append((target_mz, best_match_for_cand, best_dist_for_cand))

        # Preferisci il candidato con match entro tolleranza.
        # Se piÃ¹ candidati matchano, usa quello con distanza minore.
        # Se nessuno matcha, usa il candidato piÃ¹ grande (valore PTR-MS tipico).
        good_matches = [(mz, match, dist) for mz, match, dist in candidate_matches if dist <= tolerance]

        if good_matches:
            # Scegli il match migliore (distanza minima)
            good_matches.sort(key=lambda x: x[2])
            overall_best_mz, overall_best_match, overall_best_dist = good_matches[0]
        else:
            # Nessun match entro tolleranza: usa il candidato piÃ¹ grande (valore PTR-MS tipico)
            overall_best_mz = candidates[-1]
            overall_best_match = candidate_matches[-1][1]
            overall_best_dist = candidate_matches[-1][2]

        if overall_best_match is not None and overall_best_dist <= tolerance:
            mass_col_name = avail_values[overall_best_match]
            matched.append(mass_col_name)

            # Costruisci nome composto (se piÃ¹ righe matchano la stessa massa,
            # accumula tutti i nomi separati da " / ")
            compound_parts = []
            if col_composto is not None:
                comp = df.iloc[row_idx][col_composto]
                if pd.notna(comp) and str(comp).strip():
                    compound_parts.append(str(comp).strip())
            if col_formula is not None:
                form = df.iloc[row_idx][col_formula]
                if pd.notna(form) and str(form).strip():
                    compound_parts.append(str(form).strip())
            compound_parts.append(f"m/z {overall_best_mz:.4f}")
            new_compound = " | ".join(compound_parts)

            if mass_col_name in mass_to_compound:
                # Accumula se Ã¨ un composto diverso matchato alla stessa massa
                existing = mass_to_compound[mass_col_name]
                if new_compound not in existing:
                    mass_to_compound[mass_col_name] = existing + " / " + new_compound
            else:
                mass_to_compound[mass_col_name] = new_compound

            logger.info(f"  Massa Excel {raw_val} -> m/z {overall_best_mz:.4f} "
                       f"-> match: {mass_col_name} (delta={overall_best_dist:.3f})")
        else:
            # Log con il miglior candidato trovato
            best_cand = candidates[-1] if candidates else 0
            logger.warning(f"  Massa Excel {raw_val} -> m/z candidati {[f'{c:.4f}' for c in candidates]} "
                          f"-> NESSUN match (best delta={overall_best_dist:.3f})")

    # Rimuovi duplicati e ordina
    matched = sorted(set(matched))
    logger.info(f"Masse da Excel per plotting: {len(matched)}/{len(df)} matchate")
    return matched, mass_to_compound


# =============================================================================
# DATACLASS PER STRUTTURE DATI
# =============================================================================
@dataclass
class TesiData:
    """
    Contenitore per tutti i dati relativi a una singola tesi sperimentale.

    Memorizza i dati in ogni fase della pipeline:
    - raw_data: dati grezzi originali
    - medians: mediane calcolate per blocco
    - medians_clean: mediane dopo rimozione anomalie
    - aggregated: dati aggregati tra repliche
    - final: dati finali dopo sottrazione del bianco
    """
    name: str                              # Nome identificativo della tesi (es. "H0", "H1")
    raw_data: List[pd.DataFrame]           # Lista dei DataFrame grezzi per ogni file
    medians: List[pd.DataFrame]            # Lista delle mediane per ogni file
    medians_clean: List[pd.DataFrame]      # Lista delle mediane pulite per ogni file
    aggregated: Optional[pd.DataFrame] = None  # DataFrame aggregato (uno per tesi)
    final: Optional[pd.DataFrame] = None       # DataFrame finale dopo sottrazione blank
    file_names: List[str] = None               # Lista dei nomi dei file sorgente


# =============================================================================
# FUNZIONI DI CARICAMENTO DATI
# =============================================================================
def load_folder_data(
    folder_path: Union[str, Path],
    runs: Optional[List[int]] = None
) -> Tuple[List[pd.DataFrame], List[str]]:
    """
    Carica tutti i file CSV/TXT da una cartella con sottocartelle run{n}.

    Questa funzione gestisce due strutture di cartelle:

    Struttura 1 - Con sottocartelle run:
        folder_path/
            run1/
                stanza1.csv
                stanza2.csv
            run2/
                stanza1.csv
                ...

    Struttura 2 - File diretti:
        folder_path/
            stanza1.csv
            stanza2.csv
            ...

    Args:
        folder_path: Percorso alla cartella contenente i dati
        runs: Lista opzionale di numeri di run da caricare (es. [2, 4]).
              Se None, carica tutte le run disponibili.

    Returns:
        Tuple contenente:
            - Lista di DataFrame con i dati caricati
            - Lista di nomi file corrispondenti

    Raises:
        FileNotFoundError: Se la cartella non esiste
        ValueError: Se non vengono trovati file validi
    """
    # Converte il percorso in oggetto Path per gestione cross-platform
    folder_path = Path(folder_path)

    # Verifica esistenza della cartella
    if not folder_path.exists():
        raise FileNotFoundError(f"Cartella non trovata: {folder_path}")

    # Liste per accumulare i dati caricati
    data_list = []   # DataFrame caricati
    file_names = []  # Nomi file corrispondenti

    # -------------------------------------------------------------------------
    # RICERCA SOTTOCARTELLE RUN
    # -------------------------------------------------------------------------
    # Cerca sottocartelle che matchano il pattern "run" seguito da numeri
    # Esempio: run1, run2, Run01, RUN10, etc.
    run_dirs = sorted([d for d in folder_path.iterdir()
                       if d.is_dir() and re.match(r'run\d+', d.name, re.IGNORECASE)])

    # Filtra per numeri di run richiesti (se specificato)
    if runs is not None and run_dirs:
        runs_set = set(runs)
        run_dirs = [d for d in run_dirs
                    if int(re.search(r'\d+', d.name).group()) in runs_set]
        logger.info(f"Filtro run attivo: {sorted(runs_set)} -> {[d.name for d in run_dirs]}")

    # -------------------------------------------------------------------------
    # CARICAMENTO FILE
    # -------------------------------------------------------------------------
    if run_dirs:
        # CASO 1: Struttura con sottocartelle run
        # Itera su ogni sottocartella run in ordine
        for run_dir in run_dirs:
            # Trova tutti i file CSV e TXT nella sottocartella
            files = sorted(list(run_dir.glob("*.csv")) + list(run_dir.glob("*.txt")))

            # Carica ogni file trovato
            for f in files:
                try:
                    # Usa il parser PTR-MS per caricare e preparare i dati
                    df = load_and_prepare(f)
                    data_list.append(df)
                    # Salva il nome file con il prefisso della cartella run
                    file_names.append(f"{run_dir.name}/{f.name}")
                    logger.info(f"Caricato: {run_dir.name}/{f.name} ({len(df)} righe)")
                except Exception as e:
                    # Log warning ma continua con gli altri file
                    logger.warning(f"Errore caricamento {f}: {e}")
    else:
        # CASO 2: File direttamente nella cartella principale
        # Trova tutti i file CSV e TXT nella cartella
        files = sorted(list(folder_path.glob("*.csv")) + list(folder_path.glob("*.txt")))

        # Carica ogni file trovato
        for f in files:
            try:
                df = load_and_prepare(f)
                data_list.append(df)
                file_names.append(f.name)
                logger.info(f"Caricato: {f.name} ({len(df)} righe)")
            except Exception as e:
                logger.warning(f"Errore caricamento {f}: {e}")

    # -------------------------------------------------------------------------
    # VALIDAZIONE RISULTATI
    # -------------------------------------------------------------------------
    # Verifica che almeno un file sia stato caricato
    if not data_list:
        raise ValueError(f"Nessun file valido trovato in {folder_path}")

    logger.info(f"Caricati {len(data_list)} file da {folder_path}")
    return data_list, file_names


# =============================================================================
# FUNZIONI DI SPLIT TEMPORALE
# =============================================================================
def estimate_days_from_data(df: pd.DataFrame) -> float:
    """
    Stima il numero di giorni coperti dai dati nel DataFrame.

    Calcola la differenza tra il tempo massimo e minimo nella colonna
    RelTime e converte in giorni.

    Args:
        df: DataFrame con colonna 'RelTime' in secondi

    Returns:
        Numero di giorni (float) coperti dai dati
    """
    # Verifica presenza della colonna temporale
    if 'RelTime' not in df.columns:
        return 0

    # Calcola durata totale e converte in giorni
    total_seconds = df['RelTime'].max() - df['RelTime'].min()
    return total_seconds / SECONDS_PER_DAY


def _extract_run_from_name(file_name: str) -> Optional[int]:
    """Estrae il numero di run dal nome file (es. Run4/stanza_01.txt -> 4)."""
    match = re.search(r'run(\d+)', str(file_name), re.IGNORECASE)
    return int(match.group(1)) if match else None


def split_by_run_schedule(
    data_list: List[pd.DataFrame],
    file_names: List[str],
    run_split_hours: Dict[int, float],
    transition_exclude_hours: float = 0.0,
    fallback_split_hours: Optional[float] = None
) -> Tuple[Tuple[List[pd.DataFrame], List[str]], Tuple[List[pd.DataFrame], List[str]]]:
    """
    Divide ogni serie storica in H0/H1 e H2/H3 usando offset specifici per run.

    Inoltre elimina una fascia temporale attorno al cambio (fine H1 + inizio H2/H3)
    per ridurre contaminazioni ambientali durante il passaggio.
    """
    first_part_data: List[pd.DataFrame] = []
    first_part_names: List[str] = []
    second_part_data: List[pd.DataFrame] = []
    second_part_names: List[str] = []

    transition_exclude_seconds = max(0.0, transition_exclude_hours * 3600.0)

    for df, name in zip(data_list, file_names):
        if df.empty or 'RelTime' not in df.columns:
            logger.warning(f"File {name}: dati vuoti o senza RelTime, saltato")
            continue

        run_id = _extract_run_from_name(name)
        split_hours = run_split_hours.get(run_id, fallback_split_hours)

        if split_hours is None:
            logger.warning(
                f"File {name}: run {run_id} senza schedule dedicato e fallback assente, uso punto medio"
            )

        min_time = df['RelTime'].min()
        max_time = df['RelTime'].max()
        total_duration = max_time - min_time

        if split_hours is None:
            split_point = (min_time + max_time) / 2.0
            schedule_label = 'midpoint'
        else:
            split_point = min_time + split_hours * 3600.0
            schedule_label = f'{split_hours:.2f}h'

        if split_point > max_time:
            split_point = (min_time + max_time) / 2.0
            logger.warning(
                f"File {name}: durata ({total_duration/3600:.1f}h) < split ({schedule_label}), uso punto medio"
            )

        first_end = split_point - transition_exclude_seconds
        second_start = split_point + transition_exclude_seconds

        if first_end <= min_time:
            first_end = split_point
        if second_start >= max_time:
            second_start = split_point

        total_days = total_duration / SECONDS_PER_DAY
        logger.info(f"{name}: run={run_id}, durata totale {total_days:.2f} giorni ({total_duration/3600:.1f} ore)")
        logger.info(
            f"  - split_point: {split_point/3600:.1f}h (schedule {schedule_label}), "
            f"esclusione cambio Â±{transition_exclude_hours:.2f}h"
        )

        df_first = df[df['RelTime'] <= first_end].copy()
        if not df_first.empty:
            first_part_data.append(df_first)
            first_part_names.append(name)
            first_duration = estimate_days_from_data(df_first)
            logger.info(
                f"  - Prima parte: {len(df_first)} righe, "
                f"{first_duration:.2f} giorni ({first_duration*24:.1f} ore)"
            )

        df_second = df[df['RelTime'] >= second_start].copy()
        if not df_second.empty:
            df_second = df_second.copy()
            df_second['RelTime'] = df_second['RelTime'] - df_second['RelTime'].min()
            second_part_data.append(df_second)
            second_part_names.append(name)
            second_duration = estimate_days_from_data(df_second)
            logger.info(
                f"  - Seconda parte: {len(df_second)} righe, "
                f"{second_duration:.2f} giorni ({second_duration*24:.1f} ore)"
            )

    logger.info(
        f"Split schedule-run completato: {len(first_part_data)} file prima parte, "
        f"{len(second_part_data)} file seconda parte"
    )

    return (first_part_data, first_part_names), (second_part_data, second_part_names)


def split_acclimation(
    data_list: List[pd.DataFrame],
    file_names: List[str],
    run_acclim_hours: Dict[int, float],
    transition_exclude_hours: float = 0.0,
    fallback_acclim_hours: Optional[float] = None
) -> Tuple[Tuple[List[pd.DataFrame], List[str]], Tuple[List[pd.DataFrame], List[str]]]:
    """
    Separa la fase di acclimatizzazione (giorno 0 -> giorno 1) dalla fase
    post-acclimatizzazione nei dati giÃ  splittati (prima parte).

    Restituisce (pre_data, pre_names), (post_data, post_names) dove:
    - pre = acclimatizzazione
    - post = post-acclimatizzazione
    """
    pre_data: List[pd.DataFrame] = []
    pre_names: List[str] = []
    post_data: List[pd.DataFrame] = []
    post_names: List[str] = []

    transition_exclude_seconds = max(0.0, transition_exclude_hours * 3600.0)

    for df, name in zip(data_list, file_names):
        if df.empty or 'RelTime' not in df.columns:
            logger.warning(f"File {name}: dati vuoti o senza RelTime, saltato (acclim split)")
            continue

        run_id = _extract_run_from_name(name)
        acclim_hours = run_acclim_hours.get(run_id, fallback_acclim_hours)

        if acclim_hours is None:
            logger.warning(
                f"File {name}: run {run_id} senza acclim_hours dedicato e fallback assente, "
                f"nessuna separazione acclimatizzazione"
            )
            post_data.append(df)
            post_names.append(name)
            continue

        min_time = df['RelTime'].min()
        max_time = df['RelTime'].max()
        total_duration = max_time - min_time

        split_point = min_time + acclim_hours * 3600.0

        if split_point > max_time:
            logger.warning(
                f"File {name}: durata ({total_duration/3600:.1f}h) < acclim split "
                f"({acclim_hours:.2f}h), tutto assegnato a post-acclim"
            )
            post_data.append(df)
            post_names.append(name)
            continue

        pre_end = split_point - transition_exclude_seconds
        post_start = split_point + transition_exclude_seconds

        if pre_end <= min_time:
            pre_end = split_point
        if post_start >= max_time:
            post_start = split_point

        logger.info(
            f"{name}: run={run_id}, acclim split a {acclim_hours:.2f}h, "
            f"esclusione Â±{transition_exclude_hours:.2f}h"
        )

        df_pre = df[df['RelTime'] <= pre_end].copy()
        if not df_pre.empty:
            pre_data.append(df_pre)
            pre_names.append(name)
            pre_hours = (df_pre['RelTime'].max() - df_pre['RelTime'].min()) / 3600
            logger.info(f"  - Acclimatizzazione: {len(df_pre)} righe, {pre_hours:.1f} ore")

        df_post = df[df['RelTime'] >= post_start].copy()
        if not df_post.empty:
            df_post = df_post.copy()
            df_post['RelTime'] = df_post['RelTime'] - df_post['RelTime'].min()
            post_data.append(df_post)
            post_names.append(name)
            post_hours = (df_post['RelTime'].max() - df_post['RelTime'].min()) / 3600
            logger.info(f"  - Post-acclimatizzazione: {len(df_post)} righe, {post_hours:.1f} ore")

    logger.info(
        f"Split acclimatizzazione completato: {len(pre_data)} file pre, "
        f"{len(post_data)} file post"
    )

    return (pre_data, pre_names), (post_data, post_names)


def split_by_hours(
    data_list: List[pd.DataFrame],
    file_names: List[str],
    split_hours: float
) -> Tuple[Tuple[List[pd.DataFrame], List[str]], Tuple[List[pd.DataFrame], List[str]]]:
    """
    Divide ogni serie storica di VOC in due parti basate su un punto di split in ore.

    Questa funzione permette di separare i dati pre-trattamento dai dati
    post-trattamento, dividendo le serie temporali in:
    - Prima parte: dall'inizio fino a split_hours
    - Seconda parte: da split_hours fino alla fine

    Esempio con split_hours=70:
        Serie di 118 ore -> [Ora 0-70] e [Ora 70-118]

    Args:
        data_list: Lista di DataFrame con i dati grezzi
        file_names: Lista di nomi file corrispondenti
        split_hours: Numero di ore per il punto di split

    Returns:
        Tupla di due tuple, ciascuna contenente:
            - Lista di DataFrame per quella parte
            - Lista di nomi file corrispondenti
    """
    # Liste per accumulare i risultati dello split
    first_part_data = []   # DataFrame della prima parte
    first_part_names = []  # Nomi file della prima parte
    second_part_data = []  # DataFrame della seconda parte
    second_part_names = [] # Nomi file della seconda parte

    # Converti ore in secondi
    split_seconds = split_hours * 3600

    # -------------------------------------------------------------------------
    # ELABORAZIONE DI OGNI FILE
    # -------------------------------------------------------------------------
    for df, name in zip(data_list, file_names):
        # Salta file vuoti o senza colonna temporale
        if df.empty or 'RelTime' not in df.columns:
            logger.warning(f"File {name}: dati vuoti o senza RelTime, saltato")
            continue

        # -----------------------------------------------------------------
        # CALCOLO LIMITI TEMPORALI
        # -----------------------------------------------------------------
        # Trova estremi temporali della serie
        min_time = df['RelTime'].min()  # Tempo iniziale (secondi)
        max_time = df['RelTime'].max()  # Tempo finale (secondi)
        total_duration = max_time - min_time  # Durata totale

        # Calcola il punto di split
        split_point = min_time + split_seconds

        # -----------------------------------------------------------------
        # GESTIONE FILE CORTI
        # -----------------------------------------------------------------
        # Se il file Ã¨ piÃ¹ corto del punto di split, usa il punto medio
        if split_point > max_time:
            split_point = (min_time + max_time) / 2
            logger.warning(f"File {name}: durata ({total_duration/3600:.1f}h) < split ({split_hours}h), uso punto medio")

        # Log dettagliato per debug
        total_days = total_duration / SECONDS_PER_DAY
        logger.info(f"{name}: durata totale {total_days:.2f} giorni ({total_duration/3600:.1f} ore)")
        logger.info(f"  - split_point: {split_point/3600:.1f}h")

        # -----------------------------------------------------------------
        # SPLIT PRIMA PARTE (prima di split_point)
        # -----------------------------------------------------------------
        # Filtra i dati fino al punto di split
        df_first = df[df['RelTime'] <= split_point].copy()
        if not df_first.empty:
            first_part_data.append(df_first)
            first_part_names.append(name)
            first_duration = estimate_days_from_data(df_first)
            logger.info(f"  - Prima parte: {len(df_first)} righe, "
                       f"{first_duration:.2f} giorni ({first_duration*24:.1f} ore)")

        # -----------------------------------------------------------------
        # SPLIT SECONDA PARTE (dopo split_point)
        # -----------------------------------------------------------------
        # Filtra i dati dal punto di split in poi
        df_second = df[df['RelTime'] >= split_point].copy()
        if not df_second.empty:
            # Ricalcola RelTime per far partire la seconda parte da 0
            # Questo normalizza i tempi per confronti successivi
            df_second = df_second.copy()
            df_second['RelTime'] = df_second['RelTime'] - df_second['RelTime'].min()
            second_part_data.append(df_second)
            second_part_names.append(name)
            second_duration = estimate_days_from_data(df_second)
            logger.info(f"  - Seconda parte: {len(df_second)} righe, "
                       f"{second_duration:.2f} giorni ({second_duration*24:.1f} ore)")

    # -------------------------------------------------------------------------
    # LOG RISULTATI SPLIT
    # -------------------------------------------------------------------------
    logger.info(f"Split completato: {len(first_part_data)} file prima parte, "
               f"{len(second_part_data)} file seconda parte")

    return (first_part_data, first_part_names), (second_part_data, second_part_names)


# =============================================================================
# FUNZIONI DI CALCOLO MEDIANE
# =============================================================================
def compute_medians_from_dataframes(
    data_list: List[pd.DataFrame],
    file_names: List[str],
    min_gap_s_between_block: float = 300.0,
    trim_percent: float = TRIM_PERCENT,
    spike_window: int = 5,
    spike_threshold: float = 3.0
) -> List[pd.DataFrame]:
    """
    Calcola le mediane per blocco per ogni DataFrame nella lista.

    Per ogni blocco di rilevamento VOC:
    1. Rileva i blocchi temporali (separati da gap > min_gap_s_between_block)
    2. Per ogni blocco, rimuove il 20% iniziale e finale (trim_percent=0.2)
       per eliminare dati transitori da attacco/stacco macchinario
    3. Calcola la mediana di ogni massa sui dati rimanenti

    Il risultato Ã¨ 1 valore mediano per ogni massa per ogni blocco temporale
    (tipicamente 1 blocco â‰ˆ 1 ora di misurazioni).

    IMPORTANTE: Il trimming del 20% iniziale e finale Ã¨ fondamentale per
    evitare di includere dati non rappresentativi causati dalle fasi di
    attacco e stacco del macchinario di rilevazione PTR-MS.

    Args:
        data_list: Lista di DataFrame con dati grezzi (serie temporali complete)
        file_names: Lista di nomi file per logging
        min_gap_s_between_block: Gap minimo in secondi per separare blocchi (default: 300s = 5 min)
        trim_percent: Percentuale da rimuovere da inizio/fine blocco (default: 0.2 = 20%)
        spike_window: Finestra per rilevamento spike (default: 5 punti)
        spike_threshold: Soglia per spike in deviazioni standard (default: 3Ïƒ)

    Returns:
        Lista di DataFrame, ognuno contenente le mediane per blocco di un file
    """
    medians_list = []  # Accumula i risultati

    # -------------------------------------------------------------------------
    # ELABORAZIONE DI OGNI FILE
    # -------------------------------------------------------------------------
    for df, name in zip(data_list, file_names):
        # Salta DataFrame vuoti
        if df.empty:
            logger.warning(f"File {name}: DataFrame vuoto, saltato")
            continue

        # -----------------------------------------------------------------
        # RILEVAMENTO BLOCCHI TEMPORALI
        # -----------------------------------------------------------------
        # Un blocco Ã¨ un insieme di misurazioni continue.
        # Due misurazioni appartengono a blocchi diversi se separate da
        # un gap temporale > min_gap_s_between_block secondi.
        blocks = detect_blocks(df, gap_threshold_seconds=min_gap_s_between_block)

        if not blocks:
            logger.warning(f"File {name}: nessun blocco rilevato")
            continue

        # Ottiene informazioni dettagliate su ogni blocco
        block_info = get_block_info(df, blocks)

        # -----------------------------------------------------------------
        # CALCOLO MEDIANE PER OGNI BLOCCO
        # -----------------------------------------------------------------
        results = []  # Accumula le righe del DataFrame risultato

        for block_id, (start_idx, end_idx) in enumerate(blocks):
            bi = block_info[block_id]  # Info del blocco corrente

            # -------------------------------------------------------------
            # PULIZIA DEL BLOCCO (TRIMMING + SPIKE REMOVAL)
            # -------------------------------------------------------------
            # La funzione clean_block:
            # 1. Rimuove il 20% iniziale dei dati (transitori da attacco macchinario)
            # 2. Rimuove il 20% finale dei dati (transitori da stacco macchinario)
            # 3. Rileva e interpola gli spike (valori anomali isolati)
            #
            # Esempio: blocco di 100 campioni -> usa solo campioni 21-80
            df_clean = clean_block(
                df, start_idx, end_idx,
                trim_percent=trim_percent,      # 20% da rimuovere per lato
                spike_window=spike_window,       # Finestra per spike detection
                spike_threshold=spike_threshold, # Soglia 3Ïƒ per spike
                verbose=False
            )

            # Salta blocchi che risultano vuoti dopo la pulizia
            if df_clean is None or df_clean.empty:
                continue

            # -------------------------------------------------------------
            # CALCOLO MEDIANA PER OGNI MASSA (vettorizzato)
            # -------------------------------------------------------------
            mass_cols = get_mass_columns(df_clean)

            # Prepara i dati di output per questo blocco
            row_data = {
                'block_id': block_id,
                'start_time': bi.start_time,
                'end_time': bi.end_time,
                'duration_seconds': bi.duration_seconds,
                'n_samples_original': bi.n_samples,
                'n_samples_after_trim': len(df_clean),
                'source_file': name
            }

            # Calcola la mediana per tutte le colonne massa in un colpo solo
            if mass_cols:
                medians = df_clean[mass_cols].median()
                row_data.update(medians.to_dict())

            results.append(row_data)

        # -----------------------------------------------------------------
        # CREAZIONE DATAFRAME RISULTATO
        # -----------------------------------------------------------------
        if results:
            medians_df = pd.DataFrame(results)

            # Normalizza i nomi delle colonne masse con tolleranza MASS_TOLERANCE
            # Questo assicura che masse quasi-identiche vengano trattate come uniche
            medians_df = normalize_mass_columns(medians_df, tolerance=MASS_TOLERANCE)

            medians_list.append(medians_df)
            logger.info(f"{name}: {len(results)} blocchi processati, "
                       f"{len(get_mass_columns(medians_df))} masse")
        else:
            logger.warning(f"File {name}: nessun blocco valido dopo pulizia")

    return medians_list


def log_pipeline_mass_coverage(stage_name: str, grouped_frames: List[List[pd.DataFrame]], tesi_names: List[str]) -> None:
    """Logga quante masse restano in ogni fase per verificare eventuali perdite."""
    logger.info(f"\n[CHECK MASSE] {stage_name}")
    for idx, frames in enumerate(grouped_frames):
        t_name = tesi_names[idx] if idx < len(tesi_names) else f"Tesi_{idx}"
        if not frames:
            logger.info(f"  - {t_name}: nessun DataFrame")
            continue

        all_masses = set()
        total_rows = 0
        for df in frames:
            if df is None or df.empty:
                continue
            total_rows += len(df)
            all_masses.update(get_mass_columns(df))

        logger.info(
            f"  - {t_name}: {len(frames)} file, {total_rows} righe totali, {len(all_masses)} masse uniche"
        )


def mediana(
    data_tuples: List[Tuple[List[pd.DataFrame], List[str]]],
    min_gap_s_between_block: float = 300.0
) -> List[List[pd.DataFrame]]:
    """
    Applica il calcolo delle mediane a ogni gruppo di dati (tesi).

    Funzione wrapper che itera su tutte le tesi e calcola le mediane
    per ogni file di ogni tesi.

    NOTA: Durante il calcolo delle mediane, il 20% iniziale e finale
    di ogni blocco viene rimosso per eliminare i transitori da
    attacco/stacco del macchinario.

    Args:
        data_tuples: Lista di tuple (data_list, file_names) per ogni tesi
        min_gap_s_between_block: Gap minimo tra blocchi in secondi (default: 300s)

    Returns:
        Lista di liste: per ogni tesi, lista di DataFrame con mediane per file
    """
    result = []

    # Itera su ogni tesi (H0, H1, H2, H3, Blank1, Blank2)
    for data_list, file_names in data_tuples:
        # Calcola mediane per tutti i file di questa tesi
        medians = compute_medians_from_dataframes(
            data_list, file_names,
            min_gap_s_between_block=min_gap_s_between_block,
            trim_percent=TRIM_PERCENT  # 20% trimming per evitare transitori
        )
        result.append(medians)

    return result


# =============================================================================
# FUNZIONI DI RIMOZIONE ANOMALIE (SALTI)
# =============================================================================
def detect_jumps(
    medians_df: pd.DataFrame,
    tau: float = 5.0,
    window_size: int = 7,
    max_glitch_length: int = 3
) -> Dict[str, np.ndarray]:
    """
    Rileva glitch di campionamento con approccio despike basato sulla derivata.

    Approccio:
    1. Calcola le differenze prime d_t = x_t - x_{t-1}
    2. Stima una soglia robusta sulle differenze (mediana + tau * MAD)
    3. Flagga un picco se |d_t| > soglia E entro max_glitch_length campioni
       esiste un d_{t+j} di segno opposto e grande (ritorno)
       E il valore intermedio Ã¨ fuori dal range dei vicini

    Args:
        medians_df: DataFrame con mediane per blocco (una riga per blocco)
        tau: Soglia in unitÃ  di MAD sulle differenze (default: 5.0)
        window_size: Semi-larghezza finestra per il range locale dei vicini (default: 7)
        max_glitch_length: Numero massimo di campioni tra salto e ritorno (default: 3)

    Returns:
        Dizionario {nome_massa: array di indici dei glitch rilevati}
    """
    mass_cols = get_mass_columns(medians_df)
    outliers = {}

    for mass in mass_cols:
        if mass not in medians_df.columns:
            continue

        values = medians_df[mass].astype(float).values
        n = len(values)

        if n < 5:
            continue

        # -----------------------------------------------------------------
        # STEP 1: Calcola le differenze prime d_t = x_t - x_{t-1}
        # -----------------------------------------------------------------
        diffs = np.diff(values)  # lunghezza n-1, diffs[i] = values[i+1] - values[i]

        # -----------------------------------------------------------------
        # STEP 2: Soglia robusta sulle differenze (mediana + tau * MAD)
        # -----------------------------------------------------------------
        med_diff = np.nanmedian(diffs)
        mad_diff = np.nanmedian(np.abs(diffs - med_diff))
        sigma_diff = mad_diff * 1.4826  # fattore per distribuzione normale

        if sigma_diff < 1e-10:
            sigma_diff = np.nanstd(diffs)
        if sigma_diff < 1e-10:
            continue

        threshold = tau * sigma_diff

        # -----------------------------------------------------------------
        # STEP 3: Cerca coppie salto-su / salto-giÃ¹ entro N campioni
        # -----------------------------------------------------------------
        glitch_mask = np.zeros(n, dtype=bool)

        # Pre-calcola maschere per evitare operazioni ripetute nel loop
        abs_diffs = np.abs(diffs)
        sign_diffs = np.sign(diffs)
        big_jump = abs_diffs > threshold
        big_jump_indices = np.where(big_jump)[0]

        for i in big_jump_indices:
            # diffs[i] Ã¨ un salto grande al campione i+1
            # Cerca un ritorno di segno opposto entro max_glitch_length
            for j in range(1, max_glitch_length + 1):
                if i + j >= len(diffs):
                    break

                # Verifica: segno opposto e ampiezza grande
                if sign_diffs[i] != sign_diffs[i + j] and big_jump[i + j]:

                    # Verifica che i valori intermedi siano fuori dal range
                    # dei vicini (prima del salto e dopo il ritorno)
                    idx_before = i
                    idx_after = i + j + 1

                    left_start = max(0, idx_before - window_size)
                    right_end = min(n, idx_after + window_size)

                    # Raccogli vicini "buoni" come slice numpy (no list)
                    left_vals = values[left_start:i + 1]
                    right_vals = values[idx_after:right_end]

                    if len(left_vals) + len(right_vals) < 2:
                        break

                    local_min = min(np.nanmin(left_vals) if len(left_vals) > 0 else np.inf,
                                    np.nanmin(right_vals) if len(right_vals) > 0 else np.inf)
                    local_max = max(np.nanmax(left_vals) if len(left_vals) > 0 else -np.inf,
                                    np.nanmax(right_vals) if len(right_vals) > 0 else -np.inf)

                    # Controlla punti intermedi del glitch con numpy
                    spike_vals = values[i + 1:i + j + 1]
                    if np.any((spike_vals < local_min) | (spike_vals > local_max)):
                        glitch_mask[i + 1:i + j + 1] = True

                    break

        # -----------------------------------------------------------------
        # STEP 4: Gestione salti alla fine della serie (senza ritorno)
        # -----------------------------------------------------------------
        # Controlla gli ultimi max_glitch_length punti per salti senza ritorno
        for i in range(max(0, len(diffs) - max_glitch_length), len(diffs)):
            # Se giÃ  flaggato, salta
            if glitch_mask[i + 1]:
                continue
                
            # Se c'Ã¨ un salto grande
            if big_jump[i]:
                idx_after_jump = i + 1

                # Range locale dei vicini PRIMA del salto
                left_start = max(0, i - window_size)
                neighbor_vals = values[left_start:i + 1]

                if len(neighbor_vals) < 2:
                    continue

                local_min = np.nanmin(neighbor_vals)
                local_max = np.nanmax(neighbor_vals)

                # Controlla se i punti dal salto alla fine sono fuori range
                tail_vals = values[idx_after_jump:n]
                if len(tail_vals) > 0:
                    out_of_range_count = int(np.sum(
                        (tail_vals < local_min) | (tail_vals > local_max)
                    ))
                    if out_of_range_count == len(tail_vals):
                        glitch_mask[idx_after_jump:n] = True

        glitch_indices = np.where(glitch_mask)[0]
        if len(glitch_indices) > 0:
            outliers[mass] = glitch_indices

    return outliers


def replace_jumps(
    medians_df: pd.DataFrame,
    outliers: Dict[str, np.ndarray],
    method: str = "interpolate"
) -> pd.DataFrame:
    """
    Sostituisce i salti anomali rilevati con valori corretti.

    Metodi disponibili:
    - "interpolate": Interpola linearmente usando i valori validi vicini
    - "median": Sostituisce con la mediana dei valori non-outlier

    Args:
        medians_df: DataFrame con le mediane originali
        outliers: Dizionario degli outlier da detect_jumps
        method: Metodo di sostituzione ("interpolate" o "median")

    Returns:
        DataFrame con i valori anomali corretti
    """
    # Crea copia per non modificare l'originale
    df_corrected = medians_df.copy()

    # -------------------------------------------------------------------------
    # CORREZIONE PER OGNI MASSA
    # -------------------------------------------------------------------------
    for mass, indices in outliers.items():
        if mass not in df_corrected.columns:
            continue

        # Estrae i valori correnti
        values = df_corrected[mass].astype(float).values.copy()

        if method == "interpolate":
            # -----------------------------------------------------------------
            # METODO INTERPOLAZIONE
            # -----------------------------------------------------------------
            # Crea maschera dei punti validi (non outlier)
            valid_mask = np.ones(len(values), dtype=bool)
            valid_mask[indices] = False

            # Coordinate e valori dei punti validi
            x_valid = np.where(valid_mask)[0]
            y_valid = values[valid_mask]

            # Interpola i valori degli outlier usando i punti validi
            if len(x_valid) >= 2:
                values[indices] = np.interp(indices, x_valid, y_valid)
            else:
                # Fallback a mediana se non abbastanza punti per interpolare
                values[indices] = np.nanmedian(values[valid_mask])

        elif method == "median":
            # -----------------------------------------------------------------
            # METODO MEDIANA
            # -----------------------------------------------------------------
            # Sostituisce outlier con la mediana dei valori validi
            mask = np.ones(len(values), dtype=bool)
            mask[indices] = False
            values[indices] = np.nanmedian(values[mask])

        # Aggiorna la colonna nel DataFrame
        df_corrected[mass] = values

    return df_corrected


def rimozione_salti(
    medians_list: List[List[pd.DataFrame]],
    tau: float = 5.0,
    window_size: int = 7,
    max_glitch_length: int = 3,
) -> List[List[pd.DataFrame]]:
    """
    Rimuove i salti anomali da tutte le liste di mediane.

    Applica detect_jumps + replace_jumps a ogni DataFrame di mediane
    per ogni file di ogni tesi.

    Args:
        medians_list: Lista di liste di DataFrame mediane (per tesi, per file)
        tolerance: Tolleranza per rilevamento salti (moltiplicatore MAD)

    Returns:
        Lista di liste di DataFrame con salti corretti
    """
    result = []

    # Itera su ogni gruppo (tesi)
    for med_group in medians_list:
        cleaned_group = []

        # Itera su ogni file della tesi
        for medians_df in med_group:
            # Rileva gli outlier
            outliers = detect_jumps(medians_df,tau,window_size,max_glitch_length)
            n_outliers = sum(len(v) for v in outliers.values())

            if n_outliers > 0:
                # Correggi gli outlier trovati
                cleaned_df = replace_jumps(medians_df, outliers)
                logger.info(f"Rimossi {n_outliers} salti anomali")
            else:
                # Nessun outlier, copia originale
                cleaned_df = medians_df.copy()

            cleaned_group.append(cleaned_df)

        result.append(cleaned_group)

    return result


# =============================================================================
# FUNZIONI DI AGGREGAZIONE REPLICHE
# =============================================================================
def synchronize_masses(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Sincronizza le colonne masse tra tutti i DataFrame.

    Se una massa Ã¨ presente in un file e non in un altro, viene aggiunta
    al file mancante con valori 0. Questo Ã¨ necessario per:
    1. Permettere confronti tra file con set di masse leggermente diversi
    2. Aggregare repliche che potrebbero avere masse diverse

    Args:
        dfs: Lista di DataFrame da sincronizzare

    Returns:
        Lista di DataFrame con le stesse colonne massa (ordinate)
    """
    if not dfs:
        return dfs

    # -------------------------------------------------------------------------
    # TROVA TUTTE LE MASSE UNICHE
    # -------------------------------------------------------------------------
    all_masses = set()
    for df in dfs:
        all_masses.update(get_mass_columns(df))

    # Ordina le masse per valore numerico
    all_masses = sorted(all_masses)

    # -------------------------------------------------------------------------
    # SINCRONIZZA OGNI DATAFRAME
    # -------------------------------------------------------------------------
    result = []
    for df in dfs:
        current_masses = set(get_mass_columns(df))
        missing_masses = [m for m in all_masses if m not in current_masses]

        if missing_masses:
            # Crea tutte le colonne mancanti in un colpo solo con pd.concat
            missing_df = pd.DataFrame(
                np.zeros((len(df), len(missing_masses))),
                columns=missing_masses,
                index=df.index,
            )
            df_sync = pd.concat([df, missing_df], axis=1)
        else:
            df_sync = df.copy()

        result.append(df_sync)

    return result


def aggregate(medians_clean_list: List[List[pd.DataFrame]]) -> List[pd.DataFrame]:
    """
    Aggrega tutti i file di ogni tesi, massa per massa.

    Per ogni tesi:
    1. Combina tutti i file (repliche dello stesso esperimento)
    2. Per ogni blocco temporale e ogni massa, calcola:
       - Mediana: valore centrale tra le repliche
       - Q1, Q3: primo e terzo quartile (intervallo interquartile)
       - IQR: range interquartile (Q3 - Q1)

    Il risultato Ã¨ una timeseries per massa con banda di confidenza
    dinamica basata sulla variabilitÃ  tra le repliche.

    Args:
        medians_clean_list: Lista di liste di DataFrame mediane pulite

    Returns:
        Lista di DataFrame aggregati (uno per tesi)
    """
    aggregated_list = []

    # -------------------------------------------------------------------------
    # AGGREGAZIONE PER OGNI TESI
    # -------------------------------------------------------------------------
    for tesi_medians in medians_clean_list:
        # Salta tesi senza dati
        if not tesi_medians:
            aggregated_list.append(pd.DataFrame())
            continue

        # Sincronizza le masse tra i file della stessa tesi
        # (aggiunge masse mancanti con 0)
        synced_dfs = synchronize_masses(tesi_medians)

        # Usa la funzione di aggregazione del modulo ptrms_analysis
        # Questa calcola mediana, Q1, Q3, IQR per ogni massa per ogni blocco
        agg_df, stats = aggregate_replicates(synced_dfs)

        logger.info(f"Aggregati {len(synced_dfs)} file: {stats.get('n_blocks', 0)} blocchi, "
                   f"{stats.get('n_masses', 0)} masse")

        aggregated_list.append(agg_df)

    return aggregated_list


# =============================================================================
# FUNZIONI DI SOTTRAZIONE BIANCO
# =============================================================================
def remove_blank(
    aggregated_list: List[pd.DataFrame],
    blank_list: List[pd.DataFrame],
    blank_mapping: Optional[Dict[int, int]] = None
) -> List[pd.DataFrame]:
    """
    Rimuove il bianco (controllo) da ogni tesi.

    Per ogni tesi, sottrae la timeseries del blank corrispondente.
    Se blank_mapping Ã¨ fornito, usa quello per determinare quale blank
    sottrarre a ogni tesi (tesi_idx -> blank_idx).
    Altrimenti usa la logica di default: prima metÃ  -> Blank1, seconda metÃ  -> Blank2.

    IMPORTANTE: I valori negativi risultanti dalla sottrazione
    vengono automaticamente trasformati in 0.

    Args:
        aggregated_list: Lista di DataFrame aggregati delle tesi
        blank_list: Lista di DataFrame blank aggregati
        blank_mapping: Mapping opzionale tesi_idx -> blank_idx

    Returns:
        Lista di DataFrame finali con blank sottratto
    """
    final_list = []
    n_tesi = len(aggregated_list)
    n_blanks = len(blank_list)

    # -------------------------------------------------------------------------
    # SOTTRAZIONE PER OGNI TESI
    # -------------------------------------------------------------------------
    for i, tesi_df in enumerate(aggregated_list):
        # Salta tesi vuote
        if tesi_df.empty:
            final_list.append(pd.DataFrame())
            continue

        # -----------------------------------------------------------------
        # SELEZIONE BLANK APPROPRIATO
        # -----------------------------------------------------------------
        if blank_mapping is not None:
            blank_idx = blank_mapping.get(i)
            if blank_idx is None:
                logger.warning(f"Tesi {i}: nessun blank nel mapping, skip sottrazione")
                final_list.append(tesi_df.copy())
                continue
        else:
            blank_idx = 0 if i < n_tesi // 2 else min(1, n_blanks - 1)

        # Verifica disponibilitÃ  blank
        if blank_idx >= n_blanks or blank_list[blank_idx].empty:
            logger.warning(f"Tesi {i}: blank {blank_idx} non disponibile, skip sottrazione")
            final_list.append(tesi_df.copy())
            continue

        blank_df = blank_list[blank_idx]

        # -----------------------------------------------------------------
        # APPLICA SOTTRAZIONE
        # -----------------------------------------------------------------
        result_df = subtract_blank_from_aggregated(tesi_df, blank_df)
        final_list.append(result_df)

        logger.info(f"Tesi {i}: sottratto blank {blank_idx}")

    return final_list


def subtract_blank_from_aggregated(
    tesi_df: pd.DataFrame,
    blank_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Sottrae il blank da un DataFrame aggregato, blocco per blocco.

    Per ogni massa:
    1. Allinea i blocchi del blank con quelli della tesi (ripetendo ciclicamente se necessario)
    2. Sottrae blocco per blocco
    3. Clip a 0 i valori negativi

    Se il blank ha meno blocchi della tesi, viene ripetuto ciclicamente.
    Esempio: tesi 59 blocchi, blank 10 blocchi -> blank ripetuto ~6 volte

    Args:
        tesi_df: DataFrame aggregato della tesi (con colonne _median, _q1, _q3)
        blank_df: DataFrame aggregato del blank

    Returns:
        DataFrame con blank sottratto blocco-per-blocco e valori negativi azzerati
    """
    result_df = tesi_df.copy()

    n_tesi_blocks = len(tesi_df)
    n_blank_blocks = len(blank_df)

    # Calcola quante volte ripetere il blank
    if n_blank_blocks > 0:
        n_repeats = int(np.ceil(n_tesi_blocks / n_blank_blocks))
        logger.info(f"  Blank: {n_blank_blocks} blocchi, Tesi: {n_tesi_blocks} blocchi, "
                   f"ripetizioni: {n_repeats}x")
    else:
        logger.warning("Blank vuoto, nessuna sottrazione")
        return result_df

    # -------------------------------------------------------------------------
    # IDENTIFICA LE MASSE NEL FORMATO AGGREGATO
    # -------------------------------------------------------------------------
    mass_cols = set()
    for col in tesi_df.columns:
        if col.endswith('_median'):
            mass_name = col.replace('_median', '')
            mass_cols.add(mass_name)

    # -------------------------------------------------------------------------
    # SOTTRAZIONE BLOCCO-PER-BLOCCO PER OGNI MASSA (vettorizzato)
    # -------------------------------------------------------------------------
    # Identifica colonne comuni per sottrazione batch
    blank_cols_set = set(blank_df.columns)
    cols_to_subtract = []
    for mass in mass_cols:
        if f'{mass}_median' in blank_cols_set:
            for suffix in ('_median', '_q1', '_q3'):
                col = f'{mass}{suffix}'
                if col in result_df.columns:
                    cols_to_subtract.append((col, f'{mass}_median'))

    if cols_to_subtract:
        # Costruisci array blank ripetuto ciclicamente per ogni colonna
        for tesi_col, blank_col in cols_to_subtract:
            blank_values = blank_df[blank_col].values
            blank_repeated = np.tile(blank_values, n_repeats)[:n_tesi_blocks]
            result_df[tesi_col] = np.clip(
                result_df[tesi_col].values - blank_repeated, 0, None
            )

    return result_df


# =============================================================================
# FUNZIONI DI CONFRONTO E METRICHE
# =============================================================================
def compute_comparison_metric(
    series_a: np.ndarray,
    series_b: np.ndarray
) -> Dict[str, float]:
    """
    Calcola la metrica di confronto tra due timeseries.

    Metrica combinata = ABS(correlazione - 1) * NRMSE

    Dove:
    - Correlazione: coefficiente di Pearson [-1, 1]
    - NRMSE: Root Mean Square Error normalizzato [0, 1]

    La metrica combinata:
    - Vale 0 per serie identiche (corr=1, NRMSE=0)
    - Aumenta con la dissimilaritÃ  tra le serie
    - Penalizza sia differenze nei valori (NRMSE) che nell'andamento (1-corr)

    Args:
        series_a: Prima serie temporale (array numpy)
        series_b: Seconda serie temporale (array numpy)

    Returns:
        Dizionario con:
            - 'correlation': coefficiente di Pearson
            - 'nrmse': RMSE normalizzato
            - 'metric': metrica combinata ABS(corr-1)*NRMSE
    """
    # -------------------------------------------------------------------------
    # ALLINEAMENTO E VALIDAZIONE
    # -------------------------------------------------------------------------
    # Usa la lunghezza minima tra le due serie
    n = min(len(series_a), len(series_b))
    if n < 2:
        return {'correlation': 0, 'nrmse': 1, 'metric': 1}

    a = series_a[:n]
    b = series_b[:n]

    # Rimuovi punti con NaN in una delle due serie
    valid_mask = ~(np.isnan(a) | np.isnan(b))
    a = a[valid_mask]
    b = b[valid_mask]

    if len(a) < 2:
        return {'correlation': 0, 'nrmse': 1, 'metric': 1}

    # -------------------------------------------------------------------------
    # CALCOLO CORRELAZIONE DI PEARSON
    # -------------------------------------------------------------------------
    # Verifica che le serie abbiano variabilitÃ  (std > 0)
    if np.std(a) > 0 and np.std(b) > 0:
        correlation = np.corrcoef(a, b)[0, 1]
        if np.isnan(correlation):
            correlation = 0
    else:
        # Serie costanti -> correlazione non definita, assume 0
        correlation = 0

    # -------------------------------------------------------------------------
    # CALCOLO NRMSE (NORMALIZED ROOT MEAN SQUARE ERROR)
    # -------------------------------------------------------------------------
    # RMSE: radice dell'errore quadratico medio
    rmse = np.sqrt(np.mean((a - b) ** 2))

    # Normalizza per il range dei dati
    range_val = np.max(np.concatenate([a, b])) - np.min(np.concatenate([a, b]))
    if range_val > 0:
        nrmse = rmse / range_val
    else:
        nrmse = 0  # Serie costanti e uguali

    # -------------------------------------------------------------------------
    # CALCOLO METRICA COMBINATA
    # -------------------------------------------------------------------------
    # ABS(correlazione - 1) = 0 se perfettamente correlate, 2 se anticorrelate
    # Moltiplicato per NRMSE dÃ  una metrica che:
    # - Ãˆ 0 per serie identiche
    # - Aumenta con dissimilaritÃ  in forma e/o valori
    metric = abs(correlation - 1) * nrmse

    return {
        'correlation': correlation,
        'nrmse': nrmse,
        'metric': metric
    }


def compute_daily_metrics(
    tesi_a_df: pd.DataFrame,
    tesi_b_df: pd.DataFrame,
    mass_name: str,
    hours_per_day: int = 24
) -> List[Dict[str, float]]:
    """
    Calcola le metriche di confronto per intervalli giornalieri.

    Divide le serie in segmenti giornalieri e calcola la metrica
    per ogni giorno, piÃ¹ un valore complessivo.

    Args:
        tesi_a_df: DataFrame aggregato della tesi A
        tesi_b_df: DataFrame aggregato della tesi B
        mass_name: Nome della massa da confrontare (es. "m59.05")
        hours_per_day: Ore per giorno, usato per stimare blocchi/giorno

    Returns:
        Lista di dizionari con metriche per ogni giorno + complessivo
        Ogni dizionario ha: 'day', 'correlation', 'nrmse', 'metric'
    """
    # Costruisce il nome della colonna mediana
    median_col = f'{mass_name}_median'

    # Verifica che la massa sia presente in entrambe le tesi
    if median_col not in tesi_a_df.columns or median_col not in tesi_b_df.columns:
        return []

    # Estrae i valori delle serie
    a_values = tesi_a_df[median_col].values
    b_values = tesi_b_df[median_col].values

    # Allinea le lunghezze
    n = min(len(a_values), len(b_values))
    a_values = a_values[:n]
    b_values = b_values[:n]

    # -------------------------------------------------------------------------
    # STIMA BLOCCHI PER GIORNO
    # -------------------------------------------------------------------------
    # Assume circa 1 blocco per ora (dipende dai dati reali)
    blocks_per_day = hours_per_day
    n_days = max(1, n // blocks_per_day)

    daily_metrics = []

    # -------------------------------------------------------------------------
    # METRICHE PER OGNI GIORNO
    # -------------------------------------------------------------------------
    for day in range(n_days):
        # Calcola indici per questo giorno
        start_idx = day * blocks_per_day
        end_idx = min((day + 1) * blocks_per_day, n)

        if start_idx >= n:
            break

        # Estrae i dati del giorno
        day_a = a_values[start_idx:end_idx]
        day_b = b_values[start_idx:end_idx]

        # Calcola metriche
        metrics = compute_comparison_metric(day_a, day_b)
        metrics['day'] = day + 1  # Giorni 1-indexed
        daily_metrics.append(metrics)

    # -------------------------------------------------------------------------
    # METRICA COMPLESSIVA
    # -------------------------------------------------------------------------
    total_metrics = compute_comparison_metric(a_values, b_values)
    total_metrics['day'] = 'complessivo'
    daily_metrics.append(total_metrics)

    return daily_metrics


def generate_comparison_excel(
    final_list: List[pd.DataFrame],
    tesi_names: List[str],
    confronti: List[Tuple[int, int]],
    output_path: Path,
    masses: Optional[List[str]] = None,
    mass_to_compound: Optional[Dict[str, str]] = None
) -> None:
    """
    Genera un file Excel con le metriche di confronto tra tesi.

    Struttura Excel:
    - Un foglio per ogni coppia di tesi confrontate
    - Righe: masse
    - Colonne: massa, composto (se disponibile), giorno_1, giorno_2, ..., complessivo
    - Valori: metrica ABS(correlazione-1)*NRMSE

    Args:
        final_list: Lista di DataFrame finali (dopo sottrazione blank)
        tesi_names: Nomi delle tesi (es. ['H0', 'H1', 'H2', 'H3'])
        confronti: Lista di tuple (idx_a, idx_b) delle coppie da confrontare
        output_path: Percorso del file Excel di output
        masses: Lista opzionale di masse da includere (None = tutte)
        mass_to_compound: Dizionario opzionale massa->nome composto
    """
    if mass_to_compound is None:
        mass_to_compound = {}

    # -------------------------------------------------------------------------
    # CREAZIONE FILE EXCEL
    # -------------------------------------------------------------------------
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:

        # Itera su ogni coppia di tesi da confrontare
        for idx_a, idx_b in confronti:
            # Verifica validitÃ  indici
            if idx_a >= len(final_list) or idx_b >= len(final_list):
                continue

            tesi_a = final_list[idx_a]
            tesi_b = final_list[idx_b]

            # Salta coppie con dati vuoti
            if tesi_a.empty or tesi_b.empty:
                continue

            # -----------------------------------------------------------------
            # TROVA MASSE COMUNI
            # -----------------------------------------------------------------
            if masses is None:
                # Estrae masse da colonne _median
                masses_a = {col.replace('_median', '') for col in tesi_a.columns
                           if col.endswith('_median')}
                masses_b = {col.replace('_median', '') for col in tesi_b.columns
                           if col.endswith('_median')}
                common_masses = sorted(masses_a & masses_b)
            else:
                common_masses = sorted(masses) if isinstance(masses, set) else masses

            # -----------------------------------------------------------------
            # COSTRUISCI DATI PER IL FOGLIO
            # -----------------------------------------------------------------
            rows = []
            for mass in common_masses:
                # Calcola metriche giornaliere per questa massa
                daily_metrics = compute_daily_metrics(tesi_a, tesi_b, mass)

                # Costruisci riga con massa e metriche per ogni giorno
                row = {'massa': mass}
                # Aggiungi nome composto se disponibile
                if mass in mass_to_compound:
                    row['composto'] = mass_to_compound[mass]
                else:
                    row['composto'] = ''

                for dm in daily_metrics:
                    day_key = f"giorno_{dm['day']}" if dm['day'] != 'complessivo' else 'complessivo'
                    row[day_key] = dm['metric']

                rows.append(row)

            # -----------------------------------------------------------------
            # SCRIVI FOGLIO EXCEL
            # -----------------------------------------------------------------
            if rows:
                sheet_df = pd.DataFrame(rows)
                # Riordina colonne: massa, composto, giorno_1, ..., complessivo
                col_order = ['massa', 'composto'] + [c for c in sheet_df.columns if c not in ('massa', 'composto')]
                sheet_df = sheet_df[col_order]
                # Nome foglio max 31 caratteri (limite Excel)
                sheet_name = f"{tesi_names[idx_a]}_vs_{tesi_names[idx_b]}"[:31]
                sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)

    logger.info(f"Excel confronti salvato: {output_path}")


# =============================================================================
# FUNZIONI DI PLOTTING - CONFRONTO RAW VS MEDIANE
# =============================================================================
def plot_comparison(
    masses: List[str],
    raw_data_list: List[List[pd.DataFrame]],
    medians_list: List[List[pd.DataFrame]],
    tesi_idx: int,
    tesi_name: str,
    output_dir: Path
) -> None:
    """
    Plotta confronto tra dati grezzi e mediane per ogni massa.

    Genera un grafico a due pannelli per ogni massa:
    - Pannello superiore: serie temporali grezze (tutti i file sovrapposti)
    - Pannello inferiore: mediane per blocco (punti)

    Questo permette di visualizzare l'effetto della riduzione dei dati
    da migliaia di punti (grezzi) a decine di mediane (per blocco).

    Args:
        masses: Lista di masse da plottare
        raw_data_list: Lista di liste di DataFrame grezzi (per tesi, per file)
        medians_list: Lista di liste di DataFrame mediane
        tesi_idx: Indice della tesi da plottare
        tesi_name: Nome della tesi per i titoli
        output_dir: Directory dove salvare i grafici PNG
    """
    # Crea directory output se non esiste
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verifica validitÃ  indice
    if tesi_idx >= len(raw_data_list) or tesi_idx >= len(medians_list):
        logger.warning(f"Indice tesi {tesi_idx} fuori range")
        return

    # Estrae i dati della tesi selezionata
    raw_dfs = raw_data_list[tesi_idx]
    median_dfs = medians_list[tesi_idx]

    # -------------------------------------------------------------------------
    # GENERA GRAFICO PER OGNI MASSA
    # -------------------------------------------------------------------------
    for mass in masses:
        # Salta se la massa non esiste in nessun DataFrame o Ã¨ identicamente 0
        has_data = False
        for df in list(raw_dfs) + list(median_dfs):
            if mass in df.columns and not np.all(df[mass].values == 0):
                has_data = True
                break
        if not has_data:
            logger.debug(f"Massa {mass} non trovata o identicamente 0 in {tesi_name}, skip plot")
            continue

        # Crea figura con 2 subplot verticali (thread-safe: usa Figure() diretto)
        fig = plt.Figure(figsize=(14, 10))
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        # -----------------------------------------------------------------
        # PANNELLO 1: DATI GREZZI
        # -----------------------------------------------------------------
        for i, df in enumerate(raw_dfs):
            if mass in df.columns:
                # Converte tempo da secondi a ore per leggibilitÃ 
                ax1.plot(df['RelTime'] / 3600, df[mass], alpha=0.5, label=f'File {i+1}')

        ax1.set_xlabel('Tempo (ore)')
        ax1.set_ylabel(f'{mass} (ppb)')
        ax1.set_title(f'{tesi_name} - {mass} - Dati Grezzi')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # -----------------------------------------------------------------
        # PANNELLO 2: MEDIANE PER BLOCCO
        # -----------------------------------------------------------------
        for i, df in enumerate(median_dfs):
            if mass in df.columns:
                # Usa start_time se disponibile, altrimenti block_id
                if 'start_time' in df.columns:
                    x = df['start_time'].values / 3600
                elif 'block_id' in df.columns:
                    x = df['block_id'].values
                else:
                    x = np.arange(len(df))
                ax2.plot(x, df[mass], 'o-', alpha=0.7, label=f'File {i+1}')

        ax2.set_xlabel('Tempo (ore)')
        ax2.set_ylabel(f'{mass} (ppb)')
        ax2.set_title(f'{tesi_name} - {mass} - Mediane per Blocco')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Salva il grafico (thread-safe)
        fig.tight_layout()
        fig.savefig(output_dir / f'{tesi_name}_{mass}_comparison.png', dpi=100,
                    facecolor='white', edgecolor='none')
        plt.close(fig)

    logger.info(f"Grafici confronto salvati in {output_dir}")


# =============================================================================
# FUNZIONI DI PLOTTING - RIMOZIONE ANOMALIE
# =============================================================================
def plot_anomaly(
    masses: List[str],
    medians_list: List[List[pd.DataFrame]],
    medians_clean_list: List[List[pd.DataFrame]],
    tesi_idx: int,
    tesi_name: str,
    output_dir: Path
) -> None:
    """
    Plotta confronto prima e dopo rimozione anomalie.

    Per ogni massa e ogni file, mostra due pannelli affiancati:
    - Sinistra: serie originale con anomalie (blu)
    - Destra: serie pulita dopo correzione (verde)

    Questo permette di verificare visivamente l'efficacia
    dell'algoritmo di rimozione salti.

    Args:
        masses: Lista di masse da plottare
        medians_list: Lista di liste di DataFrame mediane originali
        medians_clean_list: Lista di liste di DataFrame mediane pulite
        tesi_idx: Indice della tesi
        tesi_name: Nome della tesi per i titoli
        output_dir: Directory output
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verifica validitÃ  indice
    if tesi_idx >= len(medians_list) or tesi_idx >= len(medians_clean_list):
        return

    original_dfs = medians_list[tesi_idx]
    clean_dfs = medians_clean_list[tesi_idx]

    # -------------------------------------------------------------------------
    # GENERA GRAFICO PER OGNI MASSA (tutte le repliche sovrapposte)
    # -------------------------------------------------------------------------
    for mass in masses:
        # Salta se la massa non esiste in nessun DataFrame o Ã¨ identicamente 0
        has_data = False
        for df in list(original_dfs) + list(clean_dfs):
            if mass in df.columns and not np.all(df[mass].values == 0):
                has_data = True
                break
        if not has_data:
            continue

        # Due pannelli affiancati: Prima | Dopo (thread-safe: usa Figure() diretto)
        fig = plt.Figure(figsize=(16, 5))
        ax_before = fig.add_subplot(1, 2, 1)
        ax_after = fig.add_subplot(1, 2, 2)

        for i, (orig_df, clean_df) in enumerate(zip(original_dfs, clean_dfs)):
            if mass not in orig_df.columns:
                continue

            x_orig = orig_df['start_time'].values / 3600 if 'start_time' in orig_df.columns else np.arange(len(orig_df))
            x_clean = clean_df['start_time'].values / 3600 if 'start_time' in clean_df.columns else np.arange(len(clean_df))

            ax_before.plot(x_orig, orig_df[mass].values, '-o', markersize=3,
                          alpha=0.6, label=f'File {i+1}')
            ax_after.plot(x_clean, clean_df[mass].values, '-o', markersize=3,
                         alpha=0.6, label=f'File {i+1}')

        ax_before.set_xlabel('Tempo (ore)')
        ax_before.set_ylabel(f'{mass} (ppb)')
        ax_before.set_title('Prima (con anomalie)')
        ax_before.legend(fontsize=7, ncol=2)
        ax_before.grid(True, alpha=0.3)

        ax_after.set_xlabel('Tempo (ore)')
        ax_after.set_ylabel(f'{mass} (ppb)')
        ax_after.set_title('Dopo (pulito)')
        ax_after.legend(fontsize=7, ncol=2)
        ax_after.grid(True, alpha=0.3)

        fig.suptitle(f'{tesi_name} - {mass} - Rimozione Anomalie', fontsize=12, fontweight='bold')
        fig.tight_layout()
        fig.savefig(output_dir / f'{tesi_name}_{mass}_anomaly.png', dpi=100,
                    facecolor='white', edgecolor='none')
        plt.close(fig)

    logger.info(f"Grafici anomalie salvati in {output_dir}")


# =============================================================================
# FUNZIONI DI PLOTTING - AGGREGAZIONE
# =============================================================================
def plot_aggregated(
    medians_clean_list: List[List[pd.DataFrame]],
    aggregated_list: List[pd.DataFrame],
    masses: List[str],
    tesi_idx: int,
    tesi_name: str,
    output_dir: Path
) -> None:
    """
    Plotta confronto tra repliche sovrapposte e dati aggregati.

    Per ogni massa, mostra due pannelli affiancati:
    - Sinistra: tutte le repliche sovrapposte (per vedere variabilitÃ )
    - Destra: serie aggregata con banda di confidenza IQR

    Questo permette di visualizzare come l'aggregazione cattura
    la tendenza centrale e la variabilitÃ  tra repliche.

    Args:
        medians_clean_list: Mediane pulite per sovrapporre repliche
        aggregated_list: Dati aggregati con statistiche
        masses: Masse da plottare
        tesi_idx: Indice tesi
        tesi_name: Nome tesi per i titoli
        output_dir: Directory output
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if tesi_idx >= len(medians_clean_list) or tesi_idx >= len(aggregated_list):
        return

    clean_dfs = medians_clean_list[tesi_idx]
    agg_df = aggregated_list[tesi_idx]

    if agg_df.empty:
        return

    # -------------------------------------------------------------------------
    # GENERA GRAFICO PER OGNI MASSA
    # -------------------------------------------------------------------------
    for mass in masses:
        # Nomi colonne nel formato aggregato
        median_col = f'{mass}_median'
        q1_col = f'{mass}_q1'
        q3_col = f'{mass}_q3'

        if median_col not in agg_df.columns:
            continue

        # Salta se la colonna aggregata Ã¨ identicamente 0
        if np.all(agg_df[median_col].values == 0):
            logger.debug(f"Massa {mass} identicamente 0 in {tesi_name}, skip plot aggregated")
            continue

        # Thread-safe: usa Figure() diretto
        fig = plt.Figure(figsize=(14, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        # -----------------------------------------------------------------
        # PANNELLO 1: REPLICHE SOVRAPPOSTE
        # -----------------------------------------------------------------
        for i, df in enumerate(clean_dfs):
            if mass in df.columns:
                x = df['block_id'].values if 'block_id' in df.columns else np.arange(len(df))
                ax1.plot(x, df[mass].values, 'o-', alpha=0.6, label=f'Replica {i+1}')

        ax1.set_xlabel('Block ID')
        ax1.set_ylabel(f'{mass} (ppb)')
        ax1.set_title(f'{tesi_name} - Repliche Sovrapposte')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # -----------------------------------------------------------------
        # PANNELLO 2: AGGREGATO CON BANDA DI CONFIDENZA
        # -----------------------------------------------------------------
        x = agg_df['block_id'].values if 'block_id' in agg_df.columns else np.arange(len(agg_df))
        y_median = agg_df[median_col].values

        # Linea mediana
        ax2.plot(x, y_median, 'b-o', markersize=4, label='Mediana')

        # Banda IQR (intervallo interquartile)
        if q1_col in agg_df.columns and q3_col in agg_df.columns:
            y_q1 = agg_df[q1_col].values
            y_q3 = agg_df[q3_col].values
            ax2.fill_between(x, y_q1, y_q3, alpha=0.3, color='blue', label='IQR')

        ax2.set_xlabel('Block ID')
        ax2.set_ylabel(f'{mass} (ppb)')
        ax2.set_title(f'{tesi_name} - Aggregato con Intervallo di Confidenza')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / f'{tesi_name}_{mass}_aggregated.png', dpi=100,
                    facecolor='white', edgecolor='none')
        plt.close(fig)

    logger.info(f"Grafici aggregazione salvati in {output_dir}")


# =============================================================================
# FUNZIONI DI PLOTTING - RIMOZIONE BIANCO
# =============================================================================
def plot_remove_blank(
    final_list: List[pd.DataFrame],
    aggregated_list: List[pd.DataFrame],
    blank_list: List[pd.DataFrame],
    masses: List[str],
    tesi_idx: int,
    tesi_name: str,
    output_dir: Path,
    blank_mapping: Optional[Dict[int, int]] = None
) -> None:
    """
    Plotta confronto prima e dopo rimozione bianco.

    Per ogni massa, mostra TRE pannelli:
    - Sinistra: dati aggregati originali (con contributo blank)
    - Centro: timeseries del Blank che viene sottratto
    - Destra: dati finali (dopo sottrazione blank)

    I valori negativi risultanti sono stati giÃ  azzerati.

    Args:
        final_list: Dati finali (dopo sottrazione blank)
        aggregated_list: Dati aggregati (prima della sottrazione)
        blank_list: Lista DataFrame blank aggregati
        masses: Masse da plottare
        tesi_idx: Indice tesi
        tesi_name: Nome tesi
        output_dir: Directory output
        blank_mapping: Mapping opzionale tesi_idx -> blank_idx
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if tesi_idx >= len(final_list) or tesi_idx >= len(aggregated_list):
        return

    final_df = final_list[tesi_idx]
    agg_df = aggregated_list[tesi_idx]

    if final_df.empty or agg_df.empty:
        return

    # Seleziona il blank appropriato
    if blank_mapping is not None:
        blank_idx = blank_mapping.get(tesi_idx, 0)
    else:
        n_tesi = len(aggregated_list)
        blank_idx = 0 if tesi_idx < n_tesi // 2 else min(1, len(blank_list) - 1)
    blank_df = blank_list[blank_idx] if blank_idx < len(blank_list) else pd.DataFrame()

    # -------------------------------------------------------------------------
    # GENERA GRAFICO PER OGNI MASSA
    # -------------------------------------------------------------------------
    for mass in masses:
        median_col = f'{mass}_median'
        q1_col = f'{mass}_q1'
        q3_col = f'{mass}_q3'

        if median_col not in agg_df.columns or median_col not in final_df.columns:
            continue

        # Salta se entrambe le colonne sono identicamente 0
        if (np.all(agg_df[median_col].values == 0)
                and np.all(final_df[median_col].values == 0)):
            logger.debug(f"Massa {mass} identicamente 0 in {tesi_name}, skip plot blank removal")
            continue

        # Crea figura con 3 subplot (thread-safe: usa Figure() diretto)
        fig = plt.Figure(figsize=(18, 5))
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        # Prepara assi x
        x_agg = agg_df['block_id'].values if 'block_id' in agg_df.columns else np.arange(len(agg_df))
        x_final = final_df['block_id'].values if 'block_id' in final_df.columns else np.arange(len(final_df))

        # -----------------------------------------------------------------
        # PANNELLO 1: PRIMA (CON BLANK)
        # -----------------------------------------------------------------
        ax1.plot(x_agg, agg_df[median_col].values, 'b-o', markersize=4, label='Mediana')
        if q1_col in agg_df.columns and q3_col in agg_df.columns:
            ax1.fill_between(x_agg, agg_df[q1_col].values, agg_df[q3_col].values,
                           alpha=0.3, color='blue', label='IQR')
        ax1.set_xlabel('Block ID')
        ax1.set_ylabel(f'{mass} (ppb)')
        ax1.set_title(f'{tesi_name} - Prima (con Blank)')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # -----------------------------------------------------------------
        # PANNELLO 2: BLANK (CENTRO)
        # -----------------------------------------------------------------
        if not blank_df.empty and median_col in blank_df.columns:
            x_blank = blank_df['block_id'].values if 'block_id' in blank_df.columns else np.arange(len(blank_df))
            ax2.plot(x_blank, blank_df[median_col].values, 'r-o', markersize=4, label='Blank Mediana')
            if q1_col in blank_df.columns and q3_col in blank_df.columns:
                ax2.fill_between(x_blank, blank_df[q1_col].values, blank_df[q3_col].values,
                               alpha=0.3, color='red', label='Blank IQR')
            # Mostra la media del blank che viene sottratta
            blank_mean = np.nanmean(blank_df[median_col].values)
            ax2.axhline(y=blank_mean, color='darkred', linestyle='--', linewidth=2,
                       label=f'Media sottratta: {blank_mean:.2f}')
        ax2.set_xlabel('Block ID')
        ax2.set_ylabel(f'{mass} (ppb)')
        blank_name = f'Blank{blank_idx + 1}'
        ax2.set_title(f'{blank_name} - Timeseries Sottratta')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # -----------------------------------------------------------------
        # PANNELLO 3: DOPO (SENZA BLANK)
        # -----------------------------------------------------------------
        ax3.plot(x_final, final_df[median_col].values, 'g-o', markersize=4, label='Mediana')
        if q1_col in final_df.columns and q3_col in final_df.columns:
            ax3.fill_between(x_final, final_df[q1_col].values, final_df[q3_col].values,
                           alpha=0.3, color='green', label='IQR')
        ax3.set_xlabel('Block ID')
        ax3.set_ylabel(f'{mass} (ppb)')
        ax3.set_title(f'{tesi_name} - Dopo (Blank Sottratto)')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / f'{tesi_name}_{mass}_blank_removal.png', dpi=100,
                    facecolor='white', edgecolor='none')
        plt.close(fig)

    logger.info(f"Grafici rimozione blank salvati in {output_dir}")


# =============================================================================
# FUNZIONI DI PLOTTING - CONFRONTO FINALE TRA TESI
# =============================================================================
def _day_boundary_indices(df: pd.DataFrame) -> List[float]:
    """
    Ritorna le posizioni x (in indice di blocco) dove inizia ogni nuovo giorno.

    Usa la colonna 'start_time' (secondi) per determinare i confini di giorno.
    Se 'start_time' non Ã¨ presente, ritorna lista vuota.
    """
    if 'start_time' not in df.columns or df.empty:
        return []

    times_days = df['start_time'].values / 86400.0  # Converti in giorni
    x_vals = np.arange(len(df), dtype=float)

    boundaries = []
    current_day = int(times_days[0])
    for i, t in enumerate(times_days):
        day = int(t)
        if day > current_day:
            # Posiziona la linea a metÃ  tra il blocco precedente e quello corrente
            boundaries.append((x_vals[i - 1] + x_vals[i]) / 2.0)
            current_day = day

    return boundaries


def plot_comparison_final(
    final_list: List[pd.DataFrame],
    masses: List[str],
    confronti: List[Tuple[int, int]],
    tesi_names: List[str],
    output_dir: Path,
    metric_threshold: float = 0.2,
    mass_to_compound: Optional[Dict[str, str]] = None,
    sequential_pairs: Optional[List[Tuple[int, int]]] = None
) -> None:
    """
    Plotta confronto finale tra coppie di tesi.

    Per ogni coppia e ogni massa in cui ALMENO UNA delle metriche giornaliere
    (o la complessiva) supera metric_threshold, genera un grafico con:

    - Layout SOVRAPPOSTO (default): le due serie sullo stesso asse, una per tesi.
      Usato per confronti tra tesi della stessa fase (es. H0 vs H1, H2 vs H3).

    - Layout SEQUENZIALE (affiancato, due pannelli): tesi A a sinistra, tesi B
      a destra. Usato per confronti pre vs post (es. H0_pre vs H0).
      Le coppie in layout sequenziale si specificano in `sequential_pairs`.

    In entrambi i casi:
    - Bande IQR colorate per ogni tesi
    - Linee rosse verticali ai cambi di giorno
    - Box con metriche (correlazione, NRMSE, metrica combinata, max giorno)

    La soglia usa la stessa logica dell'Excel: compute_daily_metrics, scatta
    se max(giorno_1, ..., complessivo) > metric_threshold.

    Args:
        final_list: Lista di DataFrame finali (dopo sottrazione blank)
        masses: Masse da considerare (tipicamente list_mass = tutte)
        confronti: Lista di tuple (idx_a, idx_b) delle coppie
        tesi_names: Nomi delle tesi per legenda
        output_dir: Directory output
        metric_threshold: Soglia (default 0.2)
        mass_to_compound: Dizionario opzionale massa->nome composto per il titolo
        sequential_pairs: Coppie da plottare in layout affiancato (pre vs post).
                          Se None, tutte le coppie usano il layout sovrapposto.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if mass_to_compound is None:
        mass_to_compound = {}
    if sequential_pairs is None:
        sequential_pairs = []
    # Normalizza in set di tuple ordinate per confronto rapido
    _seq_set = set(tuple(p) for p in sequential_pairs)

    # Palette colori per le due tesi
    color_a = 'steelblue'
    color_b = 'tomato'

    # -------------------------------------------------------------------------
    # GENERA GRAFICI PER OGNI COPPIA DI TESI
    # -------------------------------------------------------------------------
    for idx_a, idx_b in confronti:
        # Verifica validitÃ  indici
        if idx_a >= len(final_list) or idx_b >= len(final_list):
            continue

        tesi_a = final_list[idx_a]
        tesi_b = final_list[idx_b]

        if tesi_a.empty or tesi_b.empty:
            continue

        # Nomi per legenda
        name_a = tesi_names[idx_a] if idx_a < len(tesi_names) else f'Tesi_{idx_a}'
        name_b = tesi_names[idx_b] if idx_b < len(tesi_names) else f'Tesi_{idx_b}'

        skipped = 0
        plotted = 0

        # -----------------------------------------------------------------
        # GENERA GRAFICO PER OGNI MASSA
        # -----------------------------------------------------------------
        for mass in masses:
            median_col = f'{mass}_median'
            q1_col = f'{mass}_q1'
            q3_col = f'{mass}_q3'

            if median_col not in tesi_a.columns or median_col not in tesi_b.columns:
                continue

            y_a = tesi_a[median_col].values
            y_b = tesi_b[median_col].values

            # Salta se entrambe le tesi hanno valori identicamente 0
            if np.all(y_a == 0) and np.all(y_b == 0):
                logger.debug(f"Massa {mass} identicamente 0 in {name_a} vs {name_b}, skip plot")
                skipped += 1
                continue

            # -----------------------------------------------------------------
            # FILTRO SOGLIA: stessa logica dell'Excel
            # Calcola metriche giornaliere + complessiva; plotta solo se
            # almeno UNO dei valori supera metric_threshold.
            # -----------------------------------------------------------------
            daily_metrics = compute_daily_metrics(tesi_a, tesi_b, mass)
            if not daily_metrics:
                skipped += 1
                continue
            max_metric = max(dm['metric'] for dm in daily_metrics)
            if max_metric <= metric_threshold:
                logger.debug(
                    f"Massa {mass} max_metric {max_metric:.4f} <= soglia {metric_threshold}, skip"
                )
                skipped += 1
                continue
            # Metrica complessiva per il box nel grafico
            overall_metrics = next(
                (dm for dm in daily_metrics if dm.get('day') == 'complessivo'),
                daily_metrics[-1]
            )

            # -----------------------------------------------------------------
            # INDICI X e confini di giorno per ciascuna tesi
            # -----------------------------------------------------------------
            x_a = np.arange(len(tesi_a), dtype=float)
            x_b = np.arange(len(tesi_b), dtype=float)

            day_bounds_a = _day_boundary_indices(tesi_a)
            day_bounds_b = _day_boundary_indices(tesi_b)

            compound_label = mass_to_compound.get(mass, '')
            title_str = (f'Confronto {name_a} vs {name_b} - {mass}'
                         + (f'\n{compound_label}' if compound_label else ''))
            metrics_text = (
                f"Corr: {overall_metrics['correlation']:.3f}\n"
                f"NRMSE: {overall_metrics['nrmse']:.3f}\n"
                f"Metric: {overall_metrics['metric']:.4f}\n"
                f"Max giorno: {max_metric:.4f}"
            )

            use_sequential = (idx_a, idx_b) in _seq_set

            if use_sequential:
                # -------------------------------------------------------------
                # LAYOUT SEQUENZIALE (affiancato): pre | post
                # Usato per confronti pre vs post (es. H0_pre vs H0)
                # -------------------------------------------------------------
                fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 5),
                                                  sharey=False, constrained_layout=True)

                ax_a.plot(x_a, y_a, '-o', color=color_a, markersize=4, alpha=0.85)
                if q1_col in tesi_a.columns and q3_col in tesi_a.columns:
                    ax_a.fill_between(x_a, tesi_a[q1_col].values, tesi_a[q3_col].values,
                                      alpha=0.25, color=color_a)
                for xv in day_bounds_a:
                    ax_a.axvline(x=xv, color='red', linewidth=1.2, linestyle='--', alpha=0.8)
                ax_a.set_xlabel('Block index')
                ax_a.set_ylabel(f'{mass} (ppb)')
                ax_a.set_title(name_a)
                ax_a.grid(True, alpha=0.3)

                ax_b.plot(x_b, y_b, '-s', color=color_b, markersize=4, alpha=0.85)
                if q1_col in tesi_b.columns and q3_col in tesi_b.columns:
                    ax_b.fill_between(x_b, tesi_b[q1_col].values, tesi_b[q3_col].values,
                                      alpha=0.25, color=color_b)
                for xv in day_bounds_b:
                    ax_b.axvline(x=xv, color='red', linewidth=1.2, linestyle='--', alpha=0.8)
                ax_b.set_xlabel('Block index')
                ax_b.set_ylabel(f'{mass} (ppb)')
                ax_b.set_title(name_b)
                ax_b.grid(True, alpha=0.3)

                fig.suptitle(title_str, fontsize=12, fontweight='bold')
                ax_b.text(0.98, 0.98, metrics_text,
                          transform=ax_b.transAxes, fontsize=9,
                          verticalalignment='top', horizontalalignment='right',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

            else:
                # -------------------------------------------------------------
                # LAYOUT SOVRAPPOSTO: entrambe le serie sullo stesso asse
                # Usato per confronti tra tesi della stessa fase (H0 vs H1, ecc.)
                # -------------------------------------------------------------
                fig = plt.Figure(figsize=(12, 6))
                ax = fig.add_subplot(1, 1, 1)

                ax.plot(x_a, y_a, '-o', color=color_a, markersize=4,
                        label=name_a, alpha=0.85)
                if q1_col in tesi_a.columns and q3_col in tesi_a.columns:
                    ax.fill_between(x_a, tesi_a[q1_col].values, tesi_a[q3_col].values,
                                    alpha=0.2, color=color_a)

                ax.plot(x_b, y_b, '-s', color=color_b, markersize=4,
                        label=name_b, alpha=0.85)
                if q1_col in tesi_b.columns and q3_col in tesi_b.columns:
                    ax.fill_between(x_b, tesi_b[q1_col].values, tesi_b[q3_col].values,
                                    alpha=0.2, color=color_b)

                # Linee rosse verticali per i giorni (usa i confini di tesi_a
                # come riferimento; se le serie hanno la stessa durata coincidono)
                for xv in day_bounds_a:
                    ax.axvline(x=xv, color='red', linewidth=1.2, linestyle='--', alpha=0.8)

                ax.set_xlabel('Block index')
                ax.set_ylabel(f'{mass} (ppb)')
                ax.set_title(title_str)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.text(0.02, 0.98, metrics_text,
                        transform=ax.transAxes, fontsize=9,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                fig.tight_layout()

            fig.savefig(output_dir / f'{name_a}_vs_{name_b}_{mass}_comparison.png', dpi=100,
                        facecolor='white', edgecolor='none')
            plt.close(fig)
            plotted += 1

        logger.info(
            f"Coppia {name_a} vs {name_b}: {plotted} grafici salvati "
            f"(soglia={metric_threshold}), {skipped} masse saltate"
        )

    logger.info(f"Grafici confronto finale salvati in {output_dir}")


# =============================================================================
# MAIN - ESECUZIONE PIPELINE
# =============================================================================
def main():
    """
    Main della pipeline di analisi VOC.

    Esegue tutti gli step della pipeline in sequenza:
    1. Caricamento dati
    2. Split temporale
    3. Calcolo mediane
    4. Rimozione anomalie
    5. Aggregazione
    6. Sottrazione bianco
    7. Confronto tra tesi
    """

    # =========================================================================
    # CONFIGURAZIONE - MODIFICA QUESTI PATH PER I TUOI DATI
    # =========================================================================
    # Path delle cartelle contenenti i dati
    # Ogni cartella puÃ² avere sottocartelle run1/, run2/, etc.
    # oppure file direttamente al suo interno
    H0_path = Path("H0")      # Dati tesi H0
    H1_path = Path("H1")      # Dati tesi H1
    Blank_path = Path("Blank") # Dati bianco/controllo

    # Directory per i file di output (grafici, Excel)
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # FILTRO RUN (opzionale)
    # -------------------------------------------------------------------------
    # Specifica quali run caricare per H0, H1 e Blank.
    # None = carica tutte le run disponibili nella cartella.
    # Esempio: [2, 4] = carica solo run2 e run4
    H0_runs: Optional[List[int]] =  None   # es. [2, 4, 5, 6]
    H1_runs: Optional[List[int]] =  None   # es. [2, 4, 5, 6]
    Blank_runs: Optional[List[int]] = None  # es. [2, 4, 5, 6]

    # Soglia metrica per stampare i grafici nella sezione 5 (final comparison).
    # Solo le masse con metrica di differenza > PLOT_THRESHOLD vengono plottate.
    # Imposta a 0.0 per plottare tutte le masse.
    PLOT_THRESHOLD: float = 0.2

    # -------------------------------------------------------------------------
    # PARAMETRI SPLIT TEMPORALE
    # -------------------------------------------------------------------------
    fallback_split_hours = 70  # Fallback se una run non ha orario dedicato

    # Config run schedule da file opzionale (es. config/run_schedule.xlsx)
    split_hours_map = RUN_SPLIT_HOURS.copy()
    acclim_hours_map = RUN_ACCLIM_HOURS.copy()
    start_datetimes_map = RUN_START_DATETIMES.copy()
    transition_exclude_hours = TRANSITION_EXCLUDE_HOURS

    run_config_path = RUN_CONFIG_PATH if RUN_CONFIG_PATH.exists() else Path("config/run_schedule.csv")
    cfg_split, cfg_acclim, cfg_start, cfg_transition = load_run_schedule_config(run_config_path)
    if cfg_split:
        split_hours_map.update(cfg_split)
    if cfg_acclim:
        acclim_hours_map.update(cfg_acclim)
    if cfg_start:
        start_datetimes_map.update(cfg_start)
    if cfg_transition is not None:
        transition_exclude_hours = float(cfg_transition)

    # -------------------------------------------------------------------------
    # PARAMETRI ANALISI
    # -------------------------------------------------------------------------
    # Gap minimo in secondi per considerare due misurazioni come blocchi separati
    min_gap_s_between_block = 300  # 5 minuti

    # Tolleranza per il rilevamento dei salti anomali (moltiplicatore MAD)
    tau: float = 5.0
    window_size: int = 7
    max_glitch_length: int = 3

    # -------------------------------------------------------------------------
    # SELEZIONE MASSE DA ANALIZZARE
    # -------------------------------------------------------------------------
    # None = usa tutte le masse comuni
    # Oppure specifica una lista: ["m59.05", "m69.07", "m79.05"]
    list_mass = None #["m59.05", "m69.07", "m79.05"]

    # File Excel con le masse da monitorare (solo per i grafici)
    # Se None o file non trovato, plotta tutte le masse
    composti_excel_path = Path("composti da monitorare.xlsx")

    # =========================================================================
    # STEP 1: CARICAMENTO DATI
    # =========================================================================
    logger.info("=" * 70)
    logger.info("CARICAMENTO DATI")
    logger.info("=" * 70)

    # Carica i file della tesi H0
    logger.info(f"\nCaricamento H0 da {H0_path}..." + (f" (run: {H0_runs})" if H0_runs else ""))
    H0_data, H0_names = load_folder_data(H0_path, runs=H0_runs)

    # Carica i file della tesi H1
    logger.info(f"\nCaricamento H1 da {H1_path}..." + (f" (run: {H1_runs})" if H1_runs else ""))
    H1_data, H1_names = load_folder_data(H1_path, runs=H1_runs)

    # Carica i file del bianco (controllo)
    logger.info(f"\nCaricamento Blank da {Blank_path}..." + (f" (run: {Blank_runs})" if Blank_runs else ""))
    Blank_data, Blank_names = load_folder_data(Blank_path, runs=Blank_runs)

    # =========================================================================
    # STEP 2: SPLIT TEMPORALE
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("SPLIT TEMPORALE")
    logger.info(f"  - Schedule run->split (ore): {split_hours_map}")
    logger.info(f"  - Schedule run->acclim (ore): {acclim_hours_map}")
    logger.info(f"  - Fallback split: {fallback_split_hours} ore")
    logger.info(f"  - Esclusione cambio: ±{transition_exclude_hours} ore")
    logger.info("=" * 70)

    # --- Split H0/H1 -> prima parte (H0_full/H1_full) e seconda parte (H2/H3) ---
    (H0_full_data, H0_full_names), (H2_data, H2_names) = split_by_run_schedule(
        H0_data,
        H0_names,
        split_hours_map,
        transition_exclude_hours=transition_exclude_hours,
        fallback_split_hours=fallback_split_hours,
    )

    (H1_full_data, H1_full_names), (H3_data, H3_names) = split_by_run_schedule(
        H1_data,
        H1_names,
        split_hours_map,
        transition_exclude_hours=transition_exclude_hours,
        fallback_split_hours=fallback_split_hours,
    )

    (Blank1_full_data, Blank1_full_names), (Blank2_data, Blank2_names) = split_by_run_schedule(
        Blank_data,
        Blank_names,
        split_hours_map,
        transition_exclude_hours=transition_exclude_hours,
        fallback_split_hours=fallback_split_hours,
    )

    # --- Split acclimatizzazione: separa giorno 0 (pre) da giorno 1+ (post) ---
    (H0_pre_data, H0_pre_names), (H0_data, H0_names) = split_acclimation(
        H0_full_data,
        H0_full_names,
        acclim_hours_map,
        transition_exclude_hours=transition_exclude_hours,
    )

    (H1_pre_data, H1_pre_names), (H1_data, H1_names) = split_acclimation(
        H1_full_data,
        H1_full_names,
        acclim_hours_map,
        transition_exclude_hours=transition_exclude_hours,
    )

    (Blank_pre_data, Blank_pre_names), (Blank1_data, Blank1_names) = split_acclimation(
        Blank1_full_data,
        Blank1_full_names,
        acclim_hours_map,
        transition_exclude_hours=transition_exclude_hours,
    )

    # Organizza tutti i dati in una lista ordinata
    # Ordine: [H0_pre, H0, H1_pre, H1, H2, H3, Blank_pre, Blank1, Blank2]
    List_data = [
        (H0_pre_data, H0_pre_names),
        (H0_data, H0_names),
        (H1_pre_data, H1_pre_names),
        (H1_data, H1_names),
        (H2_data, H2_names),
        (H3_data, H3_names),
        (Blank_pre_data, Blank_pre_names),
        (Blank1_data, Blank1_names),
        (Blank2_data, Blank2_names),
    ]
    tesi_names = ['H0_pre', 'H0', 'H1_pre', 'H1', 'H2', 'H3', 'Blank_pre', 'Blank1', 'Blank2']

    # =========================================================================
    # STEP 3: CALCOLO MEDIANE
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("CALCOLO MEDIANE PER BLOCCO")
    logger.info(f"  - Gap minimo tra blocchi: {min_gap_s_between_block}s")
    logger.info(f"  - Trimming: {TRIM_PERCENT*100:.0f}% iniziale + {TRIM_PERCENT*100:.0f}% finale per blocco")
    logger.info("=" * 70)

    # Calcola le mediane per ogni blocco di ogni file
    # Il 20% iniziale e finale di ogni blocco viene rimosso
    # per eliminare transitori da attacco/stacco macchinario
    List_med = mediana(List_data, min_gap_s_between_block=min_gap_s_between_block)
    log_pipeline_mass_coverage("dopo mediane", List_med, tesi_names)

    # -------------------------------------------------------------------------
    # DETERMINA MASSE (TUTTE PER EXCEL, FILTRATE PER PLOT)
    # -------------------------------------------------------------------------
    if list_mass is None:
        # Trova tutte le masse presenti nei dati
        all_masses = set()
        for med_group in List_med:
            for df in med_group:
                all_masses.update(get_mass_columns(df))
        list_mass = all_masses

    # Carica masse da monitorare dall'Excel per i grafici
    mass_to_compound = {}  # Mapping massa -> nome composto (per Excel)
    if composti_excel_path is not None:
        plot_masses, mass_to_compound = load_plot_masses(composti_excel_path, list_mass)
    else:
        plot_masses = sorted(list_mass)

    logger.info(f"Masse totali per Excel: {len(list_mass)}")
    logger.info(f"Masse selezionate per plotting: {len(plot_masses)} -> {plot_masses}")
    if mass_to_compound:
        logger.info(f"Composti identificati: {len(mass_to_compound)}")
        for m, c in sorted(mass_to_compound.items()):
            logger.info(f"  {m} -> {c}")

    # -------------------------------------------------------------------------
    # PLOT: CONFRONTO RAW VS MEDIANE (parallelizzato)
    # -------------------------------------------------------------------------
    raw_data_for_plot = [d[0] for d in List_data]
    n_workers = min(6, multiprocessing.cpu_count() or 4)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i, tesi_name in enumerate(tesi_names[:6]):
            futures.append(executor.submit(
                plot_comparison,
                plot_masses, raw_data_for_plot, List_med,
                i, tesi_name, output_dir / "01_comparison_raw_median"
            ))
        for f in as_completed(futures):
            f.result()  # Propaga eccezioni

    # =========================================================================
    # STEP 4: RIMOZIONE ANOMALIE/SALTI
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("RIMOZIONE ANOMALIE")
    logger.info(f"  - Tolleranza: {tau} (moltiplicatore MAD)")
    logger.info("=" * 70)

    # Rileva e corregge i salti anomali nelle serie di mediane
    List_med_clean = rimozione_salti(List_med,tau,window_size,max_glitch_length)
    log_pipeline_mass_coverage("dopo rimozione salti", List_med_clean, tesi_names)

    # -------------------------------------------------------------------------
    # PLOT: PRIMA/DOPO RIMOZIONE ANOMALIE (parallelizzato)
    # -------------------------------------------------------------------------
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i, tesi_name in enumerate(tesi_names[:6]):
            futures.append(executor.submit(
                plot_anomaly,
                plot_masses, List_med, List_med_clean,
                i, tesi_name, output_dir / "02_anomaly_removal"
            ))
        for f in as_completed(futures):
            f.result()

    # =========================================================================
    # STEP 5: AGGREGAZIONE REPLICHE
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("AGGREGAZIONE REPLICHE")
    logger.info("  - Calcolo mediana e IQR tra repliche")
    logger.info("=" * 70)

    # Aggrega i file di ogni tesi calcolando mediana e intervalli
    List_aggregated = aggregate(List_med_clean)

    # -------------------------------------------------------------------------
    # PLOT: REPLICHE VS AGGREGATO (parallelizzato)
    # -------------------------------------------------------------------------
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i, tesi_name in enumerate(tesi_names[:6]):
            futures.append(executor.submit(
                plot_aggregated,
                List_med_clean, List_aggregated, plot_masses,
                i, tesi_name, output_dir / "03_aggregation"
            ))
        for f in as_completed(futures):
            f.result()

    # =========================================================================
    # STEP 6: SOTTRAZIONE BIANCO
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("SOTTRAZIONE BIANCO")
    logger.info("  - H0_pre, H0, H1_pre, H1 usano Blank1")
    logger.info("  - H2, H3 usano Blank2")
    logger.info("  - Sottrazione blocco-per-blocco (blank ripetuto ciclicamente)")
    logger.info("  - Valori negativi -> 0")
    logger.info("=" * 70)

    # Separa tesi e blank dalla lista aggregata
    # Ordine: [H0_pre, H0, H1_pre, H1, H2, H3, Blank_pre, Blank1, Blank2]
    tesi_aggregated = List_aggregated[:6]   # H0_pre, H0, H1_pre, H1, H2, H3
    blank_aggregated = List_aggregated[6:]  # Blank_pre, Blank1, Blank2

    # Mapping tesi_idx -> blank_idx:
    # H0_pre(0), H0(1), H1_pre(2), H1(3) -> Blank1 (indice 1)
    # H2(4), H3(5) -> Blank2 (indice 2)
    blank_mapping = {0: 1, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2}

    # Sottrae il bianco appropriato da ogni tesi
    List_final = remove_blank(tesi_aggregated, blank_aggregated, blank_mapping=blank_mapping)

    # -------------------------------------------------------------------------
    # PLOT: PRIMA/DOPO SOTTRAZIONE BIANCO (parallelizzato)
    # -------------------------------------------------------------------------
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i, tesi_name in enumerate(tesi_names[:6]):
            futures.append(executor.submit(
                plot_remove_blank,
                List_final, tesi_aggregated, blank_aggregated,
                plot_masses, i, tesi_name,
                output_dir / "04_blank_removal",
                blank_mapping,
            ))
        for f in as_completed(futures):
            f.result()

    # =========================================================================
    # STEP 7: CONFRONTO TRA TESI
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("CONFRONTO TRA TESI")
    logger.info("  - Metrica: ABS(correlazione-1) * NRMSE")
    logger.info("  - NOTA: Confronti solo tra tesi della stessa durata")
    logger.info("=" * 70)

    # Definisce le coppie di tesi da confrontare
    # IMPORTANTE: Confrontiamo solo tesi della stessa durata temporale:
    # Indici: H0_pre=0, H0=1, H1_pre=2, H1=3, H2=4, H3=5
    # - H0_pre vs H0: acclimatizzazione vs post-acclimatizzazione
    # - H0 vs H1: entrambi post-acclim, stessa durata (~3 giorni)
    # - H2 vs H3: entrambi ultimi 2 giorni
    List_confronti = [
        (0, 1),  # H0_pre vs H0 (acclimatizzazione vs post-acclim)
        (1, 3),  # H0 vs H1 (entrambi post-acclim)
        (4, 5),  # H2 vs H3 (entrambi ultimi 2 giorni)
    ]

    # -------------------------------------------------------------------------
    # GENERA EXCEL CON METRICHE
    # -------------------------------------------------------------------------
    generate_comparison_excel(
        List_final,
        tesi_names[:6],
        List_confronti,
        output_dir / "comparison_metrics.xlsx",
        masses=list_mass,
        mass_to_compound=mass_to_compound
    )

    # -------------------------------------------------------------------------
    # PLOT: CONFRONTI FINALI
    # Usa list_mass (tutte le masse) per il plot, non solo plot_masses.
    # Il filtro per soglia Ã¨ applicato internamente in plot_comparison_final.
    #
    # sequential_pairs: coppie PRE vs POST -> layout affiancato (due pannelli).
    # Le altre coppie (stesso periodo) -> layout sovrapposto come prima.
    # Indici: H0_pre=0, H0=1, H1_pre=2, H1=3, H2=4, H3=5
    # -------------------------------------------------------------------------
    sequential_pairs = [
        (0, 1),  # H0_pre vs H0 -> affiancato
        (2, 3),  # H1_pre vs H1 -> affiancato (se presente in List_confronti)
    ]
    plot_comparison_final(
        List_final,
        sorted(list_mass),
        List_confronti,
        tesi_names[:6],
        output_dir / "05_final_comparison",
        metric_threshold=PLOT_THRESHOLD,
        mass_to_compound=mass_to_compound,
        sequential_pairs=sequential_pairs
    )

    # =========================================================================
    # STEP 8: ANALISI PCA
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("ANALISI PCA")
    logger.info("  - PCA singola per ogni tesi (loadings, biplot, traiettoria)")
    logger.info("  - PCA congiunta multi-tesi (traiettorie 2D/3D, distanze)")
    logger.info("  - PCA per fascia oraria (Mattina/Pomeriggio/Notte)")
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # Costruisce run_start_map: datetime REALE di inizio FASE per ogni tesi.
    #
    # ATTENZIONE: questi datetime devono corrispondere al momento in cui il
    # blocco_id=0 di List_final[i] e' stato misurato, NON al datetime grezzo
    # di inizio run. Questo perche' List_final non ha la colonna start_time
    # (viene persa nell'aggregazione), quindi la PCA stima rel_time come
    # block_id * avg_block_duration_s. Se il datetime e' sbagliato, la fascia
    # oraria e' sbagliata.
    #
    # H0_pre: il blocco_id=0 corrisponde all'inizio della run
    # H0:     il blocco_id=0 corrisponde a run_start + acclim_hours + exclusion_hours
    # H1_pre: uguale a H0_pre ma per H1_runs
    # H1:     uguale a H0 ma per H1_runs
    # H2:     il blocco_id=0 corrisponde a run_start + split_hours + exclusion_hours
    # H3:     uguale a H2 ma per H1_runs
    # -------------------------------------------------------------------------
    def _get_run_id(runs_filter):
        """Ritorna il run_id per il set di run usato (prende il primo)."""
        return runs_filter[0] if runs_filter and len(runs_filter) > 0 else None

    def _get_phase_start(run_id, phase_offset_hours):
        """
        Ritorna il datetime di inizio fase aggiustato.
        phase_offset_hours = acclim_hours + exclusion per H0/H1,
                           = split_hours + exclusion per H2/H3
        """
        if run_id is None:
            return None
        base_dt = start_datetimes_map.get(run_id)
        if base_dt is None:
            return None
        return base_dt + timedelta(hours=phase_offset_hours)

    h0_run_id = _get_run_id(H0_runs)
    h1_run_id = _get_run_id(H1_runs)

    # H0_pre/H1_pre: blocco 0 = inizio run (nessun offset)
    # H0/H1 post-acclim: blocco 0 = run_start + acclim_hours + TRANSITION_EXCLUDE_HOURS
    # H2/H3: blocco 0 = run_start + split_hours + TRANSITION_EXCLUDE_HOURS
    h0_acclim_h = acclim_hours_map.get(h0_run_id, 22.0) if h0_run_id else 22.0
    h1_acclim_h = acclim_hours_map.get(h1_run_id, 22.0) if h1_run_id else 22.0
    h0_split_h  = split_hours_map.get(h0_run_id, 90.0)  if h0_run_id else 90.0
    h1_split_h  = split_hours_map.get(h1_run_id, 90.0)  if h1_run_id else 90.0
    excl_h = transition_exclude_hours

    run_start_map = {
        'H0_pre': _get_phase_start(h0_run_id, 0.0),
        'H0':     _get_phase_start(h0_run_id, h0_acclim_h + excl_h),
        'H1_pre': _get_phase_start(h1_run_id, 0.0),
        'H1':     _get_phase_start(h1_run_id, h1_acclim_h + excl_h),
        'H2':     _get_phase_start(h0_run_id, h0_split_h + excl_h),
        'H3':     _get_phase_start(h1_run_id, h1_split_h + excl_h),
    }

    logger.info("  run_start_map (datetime inizio FASE per ogni tesi):")
    for t, dt in run_start_map.items():
        logger.info(f"    {t}: {dt}")

    # -------------------------------------------------------------------------
    # Calcola avg_block_duration_s per ogni tesi come:
    #   durata_fase_ore * 3600 / n_blocchi
    # Usa n_blocchi da List_final (aggregato) e le durate calcolate dagli offset.
    #
    # Per H0/H1: durata = split_hours - acclim_hours - 2*exclusion_hours
    # Per H2/H3: stimata come (end_of_run - split_point) = durata rimanente.
    #   Purtroppo non abbiamo il datetime di fine run preciso, ma List_med_clean
    #   contiene start_time (RelTime grezzo), quindi calcoliamo dal max RelTime
    #   dell'ultima replica disponibile.
    # -------------------------------------------------------------------------
    tesi_idx_map = {name: idx for idx, name in enumerate(tesi_names[:6])}

    def _calc_avg_duration(tesi_name, phase_duration_h):
        """Calcola avg_block_duration_s dalla durata della fase e dal numero di blocchi."""
        idx = tesi_idx_map.get(tesi_name)
        if idx is None or idx >= len(List_final):
            return None
        df = List_final[idx]
        if df is None or df.empty:
            return None
        n_blocks = len(df)
        if n_blocks < 1:
            return None
        return (phase_duration_h * 3600.0) / n_blocks

    # Durata H0/H1 post-acclim = split_hours - acclim_hours - 2*exclusion
    # (periodo dalla fine dell'acclim al punto di split, con esclusione su entrambi i lati)
    # Nota: l'esclusione e' applicata SOLO all'inizio (dopo acclim) e SOLO alla fine (prima di split)
    # quindi la durata effettiva e' split_hours - acclim_hours - 2 * excl_h
    h0_phase_h = h0_split_h - h0_acclim_h - 2.0 * excl_h
    h1_phase_h = h1_split_h - h1_acclim_h - 2.0 * excl_h

    # Per H2/H3: stima la durata dalla prima data di H2/H3 (start = split+excl)
    # al massimo RelTime grezzo nelle repliche (non disponibile direttamente qui).
    # Usiamo List_med_clean se disponibile (ha start_time), altrimenti stimiamo
    # come ~(70% del tempo rimanente dopo lo split, tipicamente ~23 ore per run6).
    # Metodo robusto: per ogni tesi, se il df finale ha un numero ragionevole di blocchi,
    # stima la durata dal rapporto n_blocchi_H2 / n_blocchi_H0 * h0_phase_h
    def _estimate_h23_duration_h(tesi_name_h23, tesi_name_h01, h01_phase_h):
        """
        Stima la durata H2/H3 come proporzione rispetto a H0/H1.
        Se H0 ha 66 blocchi e H2 ha 23 blocchi, e H0 dura 68h,
        allora H2 dura ~68h * (23/66) * (h2_phase / h0_phase correction) ~24h.
        Piu' accurato: usa il max start_time da List_med_clean se disponibile.
        """
        # Prima scelta: usa start_time da List_med_clean (ha RelTime grezzo)
        idx_h23 = tesi_idx_map.get(tesi_name_h23)
        if idx_h23 is not None and idx_h23 < len(List_med_clean):
            med_dfs = List_med_clean[idx_h23]
            if isinstance(med_dfs, list):
                # List_med_clean[i] puo' essere una lista di df per ogni replica
                all_dfs = [d for d in med_dfs if d is not None and not d.empty
                           and 'start_time' in d.columns]
            elif isinstance(med_dfs, pd.DataFrame) and not med_dfs.empty:
                all_dfs = [med_dfs] if 'start_time' in med_dfs.columns else []
            else:
                all_dfs = []

            if all_dfs:
                max_rel_time = max(float(d['start_time'].max()) for d in all_dfs)
                # max_rel_time e' il RelTime del ULTIMO blocco H2/H3 dopo il reset a 0
                # durata reale = max_rel_time + un blocco approssimativo
                idx_h01 = tesi_idx_map.get(tesi_name_h01)
                n_h01 = len(List_final[idx_h01]) if idx_h01 is not None else 1
                approx_block = (h01_phase_h * 3600.0) / max(n_h01, 1)
                return (max_rel_time + approx_block) / 3600.0

        # Seconda scelta: proporzione blocchi
        idx_h23 = tesi_idx_map.get(tesi_name_h23)
        idx_h01 = tesi_idx_map.get(tesi_name_h01)
        if idx_h23 is None or idx_h01 is None:
            return h01_phase_h * 0.35  # fallback: H2/H3 dura ~35% di H0/H1
        n_h23 = len(List_final[idx_h23]) if idx_h23 < len(List_final) else 1
        n_h01 = len(List_final[idx_h01]) if idx_h01 < len(List_final) else 1
        if n_h01 == 0:
            return h01_phase_h * 0.35
        # Se H0 ha 66 blocchi in 68h, ogni blocco = 68/66 h
        # H2 con 23 blocchi ha durata = 23 * (68/66) = 23.7h
        return n_h23 * (h01_phase_h / n_h01)

    h2_phase_h = _estimate_h23_duration_h('H2', 'H0', h0_phase_h)
    h3_phase_h = _estimate_h23_duration_h('H3', 'H1', h1_phase_h)

    avg_block_durations = {
        'H0_pre': _calc_avg_duration('H0_pre', h0_acclim_h),
        'H0':     _calc_avg_duration('H0', h0_phase_h),
        'H1_pre': _calc_avg_duration('H1_pre', h1_acclim_h),
        'H1':     _calc_avg_duration('H1', h1_phase_h),
        'H2':     _calc_avg_duration('H2', h2_phase_h),
        'H3':     _calc_avg_duration('H3', h3_phase_h),
    }
    # Rimuovi None e valori non ragionevoli
    avg_block_durations = {k: v for k, v in avg_block_durations.items()
                           if v is not None and v > 60.0}

    logger.info("  avg_block_durations (secondi per blocco aggregato):")
    for t, d in avg_block_durations.items():
        logger.info(f"    {t}: {d:.0f}s = {d/3600:.2f}h per blocco")

    run_all_pca(
        List_final=List_final,
        tesi_names=tesi_names[:6],   # H0_pre, H0, H1_pre, H1, H2, H3
        output_base_dir=output_dir,
        run_start_datetimes=run_start_map,
        mass_to_compound=mass_to_compound,
        tesi_for_joint=['H0', 'H1', 'H2', 'H3'],  # Esclude _pre dall'analisi congiunta
        avg_block_durations=avg_block_durations,
    )

    # =========================================================================
    # CONCLUSIONE
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETATA")
    logger.info(f"Output salvati in: {output_dir}")
    logger.info("  - 01_comparison_raw_median/: Grafici raw vs mediane")
    logger.info("  - 02_anomaly_removal/: Grafici rimozione anomalie")
    logger.info("  - 03_aggregation/: Grafici aggregazione repliche")
    logger.info("  - 04_blank_removal/: Grafici sottrazione bianco")
    logger.info("  - 05_final_comparison/: Grafici confronto tra tesi")
    logger.info("  - 06_pca_analysis/:  Analisi PCA (singola, congiunta, per slot)")
    logger.info("  - comparison_metrics.xlsx: Metriche di confronto")
    logger.info("=" * 70)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()






