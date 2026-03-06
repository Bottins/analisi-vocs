"""
PTR-MS Data Parser Module
=========================

Modulo per il parsing e la preparazione dei dati PTR-MS (Proton Transfer Reaction
Mass Spectrometry) esportati in formato CSV.

Il modulo gestisce:
- Lettura di file CSV con dati PTR-MS
- Filtraggio delle colonne di concentrazione (ppb)
- Normalizzazione dei nomi delle masse
- Unificazione di colonne con masse simili

Autore: PHD-VOCS Project
"""

import re
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# Configurazione logging per la finestra di log dell'applicazione web
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_csv(filepath: str | Path) -> pd.DataFrame:
    """
    Legge un file CSV PTR-MS e filtra solo le colonne di concentrazione (ppb).

    Questa funzione:
    1. Legge il file CSV (tab-separated)
    2. Mantiene le colonne temporali (AbsTime, RelTime)
    3. Filtra SOLO le colonne che contengono "(Conc)" nel nome (concentrazioni in ppb)
    4. Esclude le colonne "(Conc ug)" che rappresentano concentrazioni in microgrammi

    Args:
        filepath: Percorso al file CSV da leggere (stringa o Path)

    Returns:
        pd.DataFrame: DataFrame con colonne temporali e colonne di concentrazione (ppb)

    Raises:
        FileNotFoundError: Se il file non esiste
        pd.errors.EmptyDataError: Se il file e' vuoto
        ValueError: Se mancano le colonne temporali richieste

    Example:
        >>> df = parse_csv("stanza_01.h5_export.csv")
        >>> print(df.columns.tolist()[:5])
        ['AbsTime', 'RelTime', 'm17.027 (B_m17.027) (Conc)', ...]
    """
    filepath = Path(filepath)

    # Verifica esistenza file
    if not filepath.exists():
        error_msg = f"File non trovato: {filepath}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.debug(f"Lettura file CSV: {filepath}")

    try:
        # Legge il CSV (tab-separated per i file PTR-MS esportati)
        df = pd.read_csv(filepath, sep='\t')
        logger.debug(f"File letto: {len(df)} righe, {len(df.columns)} colonne")
    except pd.errors.EmptyDataError:
        error_msg = f"Il file e' vuoto: {filepath}"
        logger.error(error_msg)
        raise
    except Exception as e:
        error_msg = f"Errore nella lettura del file: {e}"
        logger.error(error_msg)
        raise

    # Colonne temporali da mantenere
    time_columns = ['AbsTime', 'RelTime']

    # Verifica che le colonne temporali esistano
    missing_time_cols = [col for col in time_columns if col not in df.columns]
    if missing_time_cols:
        error_msg = f"Colonne temporali mancanti: {missing_time_cols}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Filtra le colonne di concentrazione (ppb)
    # IMPORTANTE: "(Conc)" ma NON "(Conc ug)" - usiamo regex per essere precisi
    conc_columns = [
        col for col in df.columns
        if "(Conc)" in col and "(Conc ug)" not in col
    ]

    logger.debug(f"Colonne temporali: {time_columns}, concentrazione: {len(conc_columns)}")

    if len(conc_columns) == 0:
        logger.warning("Nessuna colonna di concentrazione trovata nel file!")

    # Seleziona solo le colonne rilevanti
    selected_columns = time_columns + conc_columns
    df_filtered = df[selected_columns].copy()

    logger.debug(f"DataFrame filtrato: {len(df_filtered)} righe, {len(df_filtered.columns)} colonne")

    return df_filtered


def normalize_mass_name(column_name: str) -> str:
    """
    Estrae e normalizza il numero della massa dal nome della colonna PTR-MS.

    Questa funzione:
    1. Estrae il numero della massa dal formato "mXX.XXX (B_mXX.XXX) (Conc)"
    2. Arrotonda il valore a 2 decimali
    3. Ritorna il nome normalizzato come "mXX.XX"

    Args:
        column_name: Nome della colonna nel formato PTR-MS
                    Es: "m66.025 (B_m66.025) (Conc)"

    Returns:
        str: Nome normalizzato della massa, es: "m66.02"

    Raises:
        ValueError: Se il nome della colonna non corrisponde al formato atteso

    Example:
        >>> normalize_mass_name("m66.025 (B_m66.025) (Conc)")
        'm66.03'
        >>> normalize_mass_name("m108.962 (B_m108.962) (Conc)")
        'm108.96'
    """
    # Pattern regex per estrarre il numero della massa
    # Cerca "m" seguito da numeri e punto decimale all'inizio della stringa
    pattern = r'^m(\d+\.\d+)'

    match = re.match(pattern, column_name)

    if not match:
        error_msg = f"Formato colonna non valido: '{column_name}'. Atteso formato 'mXX.XXX (...)'"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Estrae il valore numerico della massa
    mass_value = float(match.group(1))

    # Arrotonda a 2 decimali
    mass_rounded = round(mass_value, 2)

    # Costruisce il nome normalizzato
    # Usa format per garantire sempre 2 decimali (es: 66.00 invece di 66.0)
    normalized_name = f"m{mass_rounded:.2f}"

    logger.debug(f"Normalizzazione: '{column_name}' -> '{normalized_name}'")

    return normalized_name


def unify_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizza i nomi delle colonne masse e unifica colonne duplicate.

    Questa funzione:
    1. Identifica le colonne temporali (AbsTime, RelTime) e le preserva
    2. Applica normalize_mass_name a tutte le colonne masse
    3. Se dopo la normalizzazione esistono colonne duplicate (es. m66.025 e m66.029
       diventano entrambe m66.03), le fonde calcolando la media dei valori

    Args:
        df: DataFrame con colonne masse nel formato originale PTR-MS

    Returns:
        pd.DataFrame: DataFrame con colonne normalizzate e duplicate unificate

    Example:
        >>> df = parse_csv("data.csv")
        >>> df_unified = unify_columns(df)
        >>> print(df_unified.columns.tolist()[:5])
        ['AbsTime', 'RelTime', 'm17.03', 'm18.01', 'm18.03']
    """
    logger.debug("Normalizzazione e unificazione colonne...")

    # Separa colonne temporali dalle colonne masse
    time_columns = ['AbsTime', 'RelTime']
    time_cols_present = [col for col in time_columns if col in df.columns]
    mass_columns = [col for col in df.columns if col not in time_columns]

    logger.debug(f"Colonne temporali: {len(time_cols_present)}, masse: {len(mass_columns)}")

    # Crea un dizionario per raggruppare colonne con lo stesso nome normalizzato
    # Chiave: nome normalizzato, Valore: lista di nomi colonne originali
    normalized_groups: dict[str, list[str]] = {}

    for col in mass_columns:
        try:
            normalized_name = normalize_mass_name(col)
            if normalized_name not in normalized_groups:
                normalized_groups[normalized_name] = []
            normalized_groups[normalized_name].append(col)
        except ValueError as e:
            # Se una colonna non puo' essere normalizzata, logga l'errore e salta
            logger.warning(f"Colonna saltata: {e}")
            continue

    # Conta quante colonne verranno unificate
    duplicates_found = {k: v for k, v in normalized_groups.items() if len(v) > 1}
    if duplicates_found:
        logger.debug(f"Colonne duplicate trovate: {len(duplicates_found)}")

    # Costruisce il nuovo DataFrame
    # Inizia con le colonne temporali
    result_data = {col: df[col].values for col in time_cols_present}

    # Aggiunge le colonne masse normalizzate
    for normalized_name in sorted(normalized_groups.keys()):
        original_columns = normalized_groups[normalized_name]

        if len(original_columns) == 1:
            # Nessun duplicato: usa direttamente i valori
            result_data[normalized_name] = df[original_columns[0]].values
        else:
            # Duplicati trovati: calcola la media
            values_to_average = df[original_columns].values
            result_data[normalized_name] = np.mean(values_to_average, axis=1)
            logger.debug(f"Media calcolata per {normalized_name} da {len(original_columns)} colonne")

    # Crea il DataFrame risultante
    df_unified = pd.DataFrame(result_data)

    logger.debug(f"Unificazione: {len(df.columns)} -> {len(df_unified.columns)} colonne")

    return df_unified


def load_and_prepare(filepath: str | Path) -> pd.DataFrame:
    """
    Carica un file CSV PTR-MS e lo prepara per l'analisi.

    Funzione principale che esegue l'intera pipeline di preparazione dati:
    1. Legge il file CSV
    2. Filtra solo le colonne di concentrazione (ppb)
    3. Normalizza i nomi delle masse
    4. Unifica eventuali colonne duplicate

    Questa e' la funzione consigliata per l'uso standard del modulo.

    Args:
        filepath: Percorso al file CSV PTR-MS

    Returns:
        pd.DataFrame: DataFrame pronto per l'analisi con:
            - Colonna 'AbsTime': timestamp assoluto
            - Colonna 'RelTime': tempo relativo in secondi
            - Colonne masse normalizzate: 'mXX.XX' con valori in ppb

    Raises:
        FileNotFoundError: Se il file non esiste
        ValueError: Se il file non contiene i dati attesi

    Example:
        >>> df = load_and_prepare("stanza_01.h5_export.csv")
        >>> print(f"Dati caricati: {df.shape[0]} misurazioni, {df.shape[1]-2} masse")
        >>> print(f"Durata: {df['RelTime'].max():.1f} secondi")
    """
    logger.debug(f"Preparazione file: {filepath}")

    # Step 1: Parse CSV
    df_parsed = parse_csv(filepath)

    # Step 2: Unifica colonne (include normalizzazione)
    df_prepared = unify_columns(df_parsed)

    # Riepilogo finale
    n_masses = len(df_prepared.columns) - 2  # Esclude AbsTime e RelTime
    duration = df_prepared['RelTime'].max() - df_prepared['RelTime'].min()

    logger.debug(f"Preparazione completata: {len(df_prepared)} righe, {n_masses} masse, {duration:.1f}s")

    return df_prepared


# =============================================================================
# Test del modulo
# =============================================================================
if __name__ == "__main__":
    """
    Test del modulo con il file di esempio.
    Eseguire con: python ptrms_parser.py
    """
    import sys

    # Percorso file di test
    test_file = Path(__file__).parent / "Run" / "stanza_01.h5_export.csv"

    print("\n" + "=" * 70)
    print("TEST MODULO ptrms_parser.py")
    print("=" * 70)

    # Test 1: normalize_mass_name
    print("\n--- Test normalize_mass_name ---")
    test_cases = [
        ("m66.025 (B_m66.025) (Conc)", "m66.03"),  # Python round() arrotonda 0.025 a 0.03
        ("m17.027 (B_m17.027) (Conc)", "m17.03"),
        ("m108.962 (B_m108.962) (Conc)", "m108.96"),
        ("m19.432 (B_m19.432) (Conc)", "m19.43"),
    ]

    all_passed = True
    for input_name, expected in test_cases:
        result = normalize_mass_name(input_name)
        status = "OK" if result == expected else "FAIL"
        if result != expected:
            all_passed = False
        print(f"  {status}: '{input_name}' -> '{result}' (atteso: '{expected}')")

    if all_passed:
        print("  Tutti i test normalize_mass_name passati!")

    # Test 2: Pipeline completa
    print("\n--- Test load_and_prepare ---")
    if test_file.exists():
        try:
            df = load_and_prepare(test_file)

            print(f"\n  Risultato:")
            print(f"    Shape: {df.shape}")
            print(f"    Colonne (prime 10): {list(df.columns[:10])}")
            print(f"    Tipi dati: {df.dtypes.value_counts().to_dict()}")
            print(f"\n  Prime 3 righe:")
            print(df.head(3).to_string())

            print("\n  TEST COMPLETATO CON SUCCESSO!")

        except Exception as e:
            print(f"  ERRORE: {e}")
            sys.exit(1)
    else:
        print(f"  File di test non trovato: {test_file}")
        print("  Salta test pipeline completa")

    print("\n" + "=" * 70)
