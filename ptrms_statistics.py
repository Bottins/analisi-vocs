"""
PTR-MS Statistics Module
========================

Modulo per il calcolo delle statistiche sui dati PTR-MS.

Funzionalita principali:
- Calcolo mediana per blocco temporale
- Generazione DataFrame con mediane di tutti i blocchi
- Export in formato CSV

Autore: PHD-VOCS Project
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from ptrms_parser import load_and_prepare
from ptrms_blocks import detect_blocks, get_block_info, BlockInfo
from ptrms_cleaning import clean_block

# Configurazione logging
logger = logging.getLogger(__name__)


def compute_block_median(df_block: pd.DataFrame) -> pd.Series:
    """
    Calcola la mediana colonna per colonna di un blocco di dati.

    La mediana e' preferita alla media per i dati PTR-MS perche'
    e' piu' robusta agli outlier residui.

    Args:
        df_block: DataFrame con i dati del blocco (gia' pulito)

    Returns:
        pd.Series: Serie con le mediane per ogni colonna numerica

    Example:
        >>> df_clean = clean_block(df, start, end)
        >>> medians = compute_block_median(df_clean)
        >>> print(medians['m59.05'])
        6.54
    """
    logger.debug(f"Calcolo mediana per blocco con {len(df_block)} righe")

    # Seleziona solo colonne numeriche
    numeric_cols = df_block.select_dtypes(include=[np.number]).columns
    df_numeric = df_block[numeric_cols]

    # Calcola la mediana
    medians = df_numeric.median()

    logger.debug(f"Mediane calcolate per {len(medians)} colonne")

    return medians


def compute_file_medians(
    filepath: str | Path,
    gap_threshold_seconds: float = 300.0,
    trim_percent: float = 0.2,
    spike_window: int = 5,
    spike_threshold: float = 3.0,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Elabora un file CSV PTR-MS e calcola le mediane per ogni blocco.

    Pipeline completa:
    1. Carica e prepara i dati (parse + normalizzazione)
    2. Rileva i blocchi temporali
    3. Per ogni blocco: pulizia (trim + spike removal) + mediana
    4. Assembla il DataFrame finale con metadata

    Args:
        filepath: Percorso al file CSV PTR-MS
        gap_threshold_seconds: Soglia per rilevamento blocchi (default: 300s)
        trim_percent: Percentuale trimming per lato (default: 0.2)
        spike_window: Finestra rilevamento spike (default: 5)
        spike_threshold: Soglia spike in sigma (default: 3.0)
        verbose: Se True, mostra log dettagliati (default: False)

    Returns:
        pd.DataFrame: DataFrame con una riga per blocco, contenente:
            - block_id: identificatore del blocco
            - start_time: RelTime di inizio blocco (originale)
            - end_time: RelTime di fine blocco (originale)
            - duration_seconds: durata del blocco originale
            - n_samples_original: numero campioni originali
            - n_samples_after_trim: numero campioni dopo pulizia
            - Colonne masse: mediane per ogni massa

    Raises:
        FileNotFoundError: Se il file non esiste

    Example:
        >>> medians_df = compute_file_medians("stanza_01.h5_export.csv")
        >>> print(f"Elaborati {len(medians_df)} blocchi")
        >>> print(medians_df[['block_id', 'duration_seconds', 'm59.05']].head())
    """
    filepath = Path(filepath)

    logger.debug(f"Elaborazione file: {filepath.name}")

    # Step 1: Carica e prepara i dati
    df = load_and_prepare(filepath)

    # Step 2: Rileva i blocchi
    blocks = detect_blocks(df, gap_threshold_seconds=gap_threshold_seconds)
    block_info = get_block_info(df, blocks)

    if verbose:
        logger.info(f"File {filepath.name}: {len(blocks)} blocchi rilevati")

    # Step 3: Per ogni blocco, pulisci e calcola mediana
    results = []
    skipped_blocks = 0

    for block_id, (start_idx, end_idx) in enumerate(blocks):
        bi = block_info[block_id]

        # Pulisci il blocco
        df_clean = clean_block(
            df, start_idx, end_idx,
            trim_percent=trim_percent,
            spike_window=spike_window,
            spike_threshold=spike_threshold,
            verbose=False  # Non loggare ogni singolo blocco
        )

        if df_clean is None:
            skipped_blocks += 1
            continue

        # Calcola mediane
        medians = compute_block_median(df_clean)

        # Prepara riga risultato con metadata
        row_data = {
            'block_id': block_id,
            'start_time': bi.start_time,
            'end_time': bi.end_time,
            'duration_seconds': bi.duration_seconds,
            'n_samples_original': bi.n_samples,
            'n_samples_after_trim': len(df_clean),
        }

        # Aggiungi le mediane delle masse (escludi AbsTime e RelTime)
        for col in medians.index:
            if col not in ['AbsTime', 'RelTime']:
                row_data[col] = medians[col]

        results.append(row_data)

    if len(results) == 0:
        logger.warning(f"Nessun blocco valido in {filepath.name}")
        return pd.DataFrame()

    medians_df = pd.DataFrame(results)

    # Riordina colonne: metadata prima, poi masse in ordine
    metadata_cols = ['block_id', 'start_time', 'end_time', 'duration_seconds',
                    'n_samples_original', 'n_samples_after_trim']
    mass_cols = sorted([c for c in medians_df.columns if c not in metadata_cols])
    medians_df = medians_df[metadata_cols + mass_cols]

    logger.debug(f"{filepath.name}: {len(results)} blocchi elaborati, {skipped_blocks} scartati, {len(mass_cols)} masse")

    return medians_df


def export_medians_csv(
    medians_df: pd.DataFrame,
    output_path: str | Path,
    include_metadata: bool = True
) -> Path:
    """
    Esporta il DataFrame delle mediane in formato CSV.

    Args:
        medians_df: DataFrame con le mediane (output di compute_file_medians)
        output_path: Percorso del file CSV di output
        include_metadata: Se True, include le colonne metadata (default: True)

    Returns:
        Path: Percorso del file salvato

    Example:
        >>> medians_df = compute_file_medians("data.csv")
        >>> output = export_medians_csv(medians_df, "medians_output.csv")
        >>> print(f"Salvato: {output}")
    """
    output_path = Path(output_path)

    logger.info(f"Esportazione mediane in: {output_path}")

    # Se richiesto, rimuovi metadata
    if not include_metadata:
        metadata_cols = ['block_id', 'start_time', 'end_time', 'duration_seconds',
                        'n_samples_original', 'n_samples_after_trim']
        export_df = medians_df.drop(columns=[c for c in metadata_cols if c in medians_df.columns])
    else:
        export_df = medians_df

    # Salva CSV
    export_df.to_csv(output_path, index=False, float_format='%.6f')

    logger.info(f"File salvato: {len(export_df)} righe, {len(export_df.columns)} colonne")

    return output_path


def get_processing_summary(medians_df: pd.DataFrame) -> dict:
    """
    Genera un riepilogo dell'elaborazione per la UI.

    Args:
        medians_df: DataFrame con le mediane

    Returns:
        dict: Dizionario con statistiche riassuntive
    """
    if medians_df.empty:
        return {
            'n_blocks': 0,
            'total_samples_original': 0,
            'total_samples_processed': 0,
            'n_masses': 0,
            'total_duration_hours': 0
        }

    metadata_cols = ['block_id', 'start_time', 'end_time', 'duration_seconds',
                    'n_samples_original', 'n_samples_after_trim']
    mass_cols = [c for c in medians_df.columns if c not in metadata_cols]

    return {
        'n_blocks': len(medians_df),
        'total_samples_original': int(medians_df['n_samples_original'].sum()),
        'total_samples_processed': int(medians_df['n_samples_after_trim'].sum()),
        'n_masses': len(mass_cols),
        'total_duration_hours': medians_df['duration_seconds'].sum() / 3600,
        'avg_block_duration_seconds': medians_df['duration_seconds'].mean(),
        'avg_samples_per_block': medians_df['n_samples_after_trim'].mean()
    }


# =============================================================================
# Test del modulo
# =============================================================================
if __name__ == "__main__":
    """
    Test del modulo con il file di esempio.
    Eseguire con: python ptrms_statistics.py
    """

    # Configura logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Percorso file di test
    test_file = Path(__file__).parent / "Run" / "stanza_01.h5_export.csv"
    output_file = Path(__file__).parent / "output_medians_stanza_01.csv"

    print("\n" + "=" * 70)
    print("TEST MODULO ptrms_statistics.py")
    print("=" * 70)

    if not test_file.exists():
        print(f"File di test non trovato: {test_file}")
        exit(1)

    # Test 1: compute_file_medians
    print("\n--- Test compute_file_medians ---")
    medians_df = compute_file_medians(test_file)

    print(f"\nRisultato:")
    print(f"  Shape: {medians_df.shape}")
    print(f"  Colonne metadata: block_id, start_time, end_time, duration_seconds, ...")
    print(f"  Colonne masse: {len([c for c in medians_df.columns if c.startswith('m')])}")

    # Mostra prime righe
    print(f"\nPrime 5 righe (colonne selezionate):")
    display_cols = ['block_id', 'duration_seconds', 'n_samples_original',
                   'n_samples_after_trim', 'm59.05', 'm69.08']
    display_cols = [c for c in display_cols if c in medians_df.columns]
    print(medians_df[display_cols].head().to_string())

    # Test 2: export_medians_csv
    print("\n--- Test export_medians_csv ---")
    output = export_medians_csv(medians_df, output_file)
    print(f"File salvato: {output}")

    # Verifica file
    df_reload = pd.read_csv(output)
    print(f"Verifica rilettura: {df_reload.shape}")

    # Test 3: get_processing_summary
    print("\n--- Test get_processing_summary ---")
    summary = get_processing_summary(medians_df)
    print("Riepilogo elaborazione:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("TEST COMPLETATO CON SUCCESSO!")
    print("=" * 70)
