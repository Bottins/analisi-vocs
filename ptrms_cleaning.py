"""
PTR-MS Data Cleaning Module
===========================

Modulo per la pulizia e il preprocessing dei dati PTR-MS.

Gestisce:
- Trimming dei transitori (primo/ultimo 20% di ogni blocco)
- Rilevamento di spike/anomalie con finestra mobile
- Interpolazione cubica per correggere i valori anomali

Autore: PHD-VOCS Project
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np
from scipy import interpolate

# Configurazione logging
logger = logging.getLogger(__name__)


def trim_block(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    trim_percent: float = 0.2
) -> Optional[tuple[int, int]]:
    """
    Rimuove il primo e ultimo X% delle righe di un blocco temporale.

    Il trimming rimuove i dati transitori all'inizio e alla fine di ogni
    blocco di acquisizione, che possono essere contaminati o instabili.

    Args:
        df: DataFrame completo con i dati PTR-MS
        start_idx: Indice di inizio del blocco nel DataFrame
        end_idx: Indice di fine del blocco nel DataFrame (incluso)
        trim_percent: Percentuale da rimuovere da ogni estremita (default: 0.2 = 20%)

    Returns:
        Optional[tuple[int, int]]: Tupla (new_start, new_end) con gli indici del
                                   blocco trimmato, oppure None se il blocco
                                   risultante ha meno di 10 righe

    Example:
        >>> # Blocco di 100 righe (indici 0-99)
        >>> new_start, new_end = trim_block(df, 0, 99, trim_percent=0.2)
        >>> # Rimuove 20 righe all'inizio e 20 alla fine
        >>> print(f"Nuovo blocco: {new_start}-{new_end}")
        Nuovo blocco: 20-79
    """
    # Calcola la dimensione del blocco
    block_size = end_idx - start_idx + 1

    logger.debug(f"Trimming blocco idx {start_idx}-{end_idx} ({block_size} righe), "
                f"trim={trim_percent*100:.0f}%")

    # Calcola quante righe rimuovere da ogni lato
    trim_rows = int(block_size * trim_percent)

    # Calcola i nuovi indici
    new_start = start_idx + trim_rows
    new_end = end_idx - trim_rows

    # Calcola la nuova dimensione
    new_size = new_end - new_start + 1

    # Verifica che il blocco risultante sia abbastanza grande
    min_rows = 5
    if new_size < min_rows:
        logger.warning(f"Blocco troppo piccolo dopo trimming: {new_size} righe "
                      f"(minimo: {min_rows}). Blocco scartato.")
        return None

    logger.debug(f"Trimming: {block_size} -> {new_size} righe")

    return (new_start, new_end)


def detect_spikes(
    series: pd.Series,
    window: int = 5,
    threshold_std: float = 3.0
) -> np.ndarray:
    """
    Identifica gli spike (outlier) in una serie temporale usando finestra mobile.

    Un valore e' considerato spike se la sua deviazione dalla media locale
    supera una soglia basata sulla deviazione standard locale.

    Criterio: |valore - media_locale| > threshold_std * std_locale

    Args:
        series: Serie pandas con i valori da analizzare
        window: Dimensione della finestra mobile per calcolare media e std locali
                (default: 5 punti)
        threshold_std: Moltiplicatore della deviazione standard per la soglia
                      (default: 3.0, cioe' 3 sigma)

    Returns:
        np.ndarray: Array di indici (posizioni nella serie) dei valori anomali

    Example:
        >>> series = pd.Series([1, 2, 1, 100, 2, 1, 2])  # 100 e' uno spike
        >>> spike_indices = detect_spikes(series, window=3, threshold_std=2.0)
        >>> print(spike_indices)
        [3]
    """
    if len(series) < window:
        logger.debug(f"Serie troppo corta ({len(series)}) per finestra {window}, "
                    "nessuno spike rilevato")
        return np.array([], dtype=int)

    # Usa rolling di pandas per calcolo vettorizzato di media e std locale.
    # Il rolling include il punto corrente, quindi usiamo min_periods=2
    # e poi correggiamo per escludere il punto corrente dalla statistica.
    values = series.values.astype(float)
    n = len(values)
    s = pd.Series(values)

    # Rolling centrato (include il punto corrente)
    roll = s.rolling(window=window, center=True, min_periods=2)
    roll_sum = roll.sum()
    roll_count = roll.count()
    roll_sq_sum = (s ** 2).rolling(window=window, center=True, min_periods=2).sum()

    # Correggi media e varianza escludendo il punto corrente
    # mean_excl = (sum_window - x_i) / (count - 1)
    count_excl = roll_count - 1
    sum_excl = roll_sum - values
    mean_excl = np.where(count_excl >= 2, sum_excl / count_excl, np.nan)

    # var_excl = (sum_sq_excl / (count-1)) - mean_excl^2
    sq_sum_excl = roll_sq_sum - values ** 2
    var_excl = np.where(count_excl >= 2,
                        sq_sum_excl / count_excl - mean_excl ** 2,
                        np.nan)
    # Clip varianza negativa (arrotondamento numerico)
    var_excl = np.clip(var_excl, 0, None)
    std_excl = np.sqrt(var_excl)
    std_excl = np.where(std_excl < 1e-10, 1e-10, std_excl)

    # Deviazione dal punto corrente rispetto alla media locale
    deviation = np.abs(values - mean_excl)
    is_spike = (deviation > threshold_std * std_excl) & (count_excl >= 2)

    spike_indices = np.where(is_spike)[0]

    if len(spike_indices) > 0:
        logger.debug(f"Rilevati {len(spike_indices)} spike nella serie")

    return spike_indices


def detect_spikes_dataframe(
    df: pd.DataFrame,
    window: int = 5,
    threshold_std: float = 3.0,
    exclude_columns: Optional[list[str]] = None,
    verbose: bool = False
) -> dict[str, np.ndarray]:
    """
    Applica detect_spikes a tutte le colonne numeriche di un DataFrame.

    Args:
        df: DataFrame con i dati da analizzare
        window: Dimensione della finestra mobile
        threshold_std: Soglia in deviazioni standard
        exclude_columns: Lista di colonne da escludere dall'analisi
                        (default: ['AbsTime', 'RelTime'])
        verbose: Se True, mostra log dettagliati

    Returns:
        dict[str, np.ndarray]: Dizionario con chiave = nome colonna,
                               valore = array di indici degli spike

    Example:
        >>> spikes = detect_spikes_dataframe(df)
        >>> for col, indices in spikes.items():
        ...     if len(indices) > 0:
        ...         print(f"{col}: {len(indices)} spike")
    """
    if exclude_columns is None:
        exclude_columns = ['AbsTime', 'RelTime']

    spike_indices: dict[str, np.ndarray] = {}

    # Seleziona solo colonne numeriche
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_check = [c for c in numeric_cols if c not in exclude_columns]

    if not cols_to_check:
        return spike_indices

    # Vettorizzazione: calcola rolling su tutte le colonne in una volta
    # invece di chiamare detect_spikes una per una
    values_matrix = df[cols_to_check].values.astype(float)
    n_rows, n_cols = values_matrix.shape

    if n_rows < window:
        return {col: np.array([], dtype=int) for col in cols_to_check}

    s_df = pd.DataFrame(values_matrix, columns=cols_to_check)
    roll = s_df.rolling(window=window, center=True, min_periods=2)
    roll_sum = roll.sum().values
    roll_count = roll.count().values
    roll_sq_sum = (s_df ** 2).rolling(window=window, center=True, min_periods=2).sum().values

    # Correggi media e varianza escludendo il punto corrente
    count_excl = roll_count - 1
    sum_excl = roll_sum - values_matrix
    valid = count_excl >= 2
    mean_excl = np.where(valid, sum_excl / count_excl, np.nan)
    sq_sum_excl = roll_sq_sum - values_matrix ** 2
    var_excl = np.where(valid, sq_sum_excl / count_excl - mean_excl ** 2, np.nan)
    var_excl = np.clip(var_excl, 0, None)
    std_excl = np.sqrt(var_excl)
    std_excl = np.where(std_excl < 1e-10, 1e-10, std_excl)

    deviation = np.abs(values_matrix - mean_excl)
    is_spike_matrix = (deviation > threshold_std * std_excl) & valid

    total_spikes = 0
    for j, col in enumerate(cols_to_check):
        indices = np.where(is_spike_matrix[:, j])[0]
        spike_indices[col] = indices
        total_spikes += len(indices)

    if verbose:
        logger.info(f"Spike rilevati: {total_spikes} su {len(cols_to_check)} colonne")

    return spike_indices


def interpolate_spikes(
    df: pd.DataFrame,
    spike_indices_per_column: dict[str, np.ndarray],
    verbose: bool = False
) -> pd.DataFrame:
    """
    Sostituisce gli spike con valori interpolati usando interpolazione cubica.

    Per ogni spike identificato:
    1. Usa i valori circostanti (non-spike) come punti di riferimento
    2. Applica interpolazione cubica (scipy) per ricostruire il valore
    3. Se l'interpolazione cubica fallisce, usa interpolazione lineare

    Args:
        df: DataFrame originale con i dati
        spike_indices_per_column: Dizionario con indici degli spike per colonna
                                  (output di detect_spikes_dataframe)

    Returns:
        pd.DataFrame: DataFrame con gli spike corretti tramite interpolazione

    Note:
        - Gli spike agli estremi del DataFrame vengono corretti con
          interpolazione lineare o nearest-neighbor
        - Il DataFrame originale non viene modificato (ritorna una copia)

    Example:
        >>> spikes = detect_spikes_dataframe(df)
        >>> df_clean = interpolate_spikes(df, spikes)
    """
    # Crea una copia per non modificare l'originale
    df_corrected = df.copy()

    total_corrected = 0

    for col, spike_indices in spike_indices_per_column.items():
        if len(spike_indices) == 0:
            continue

        if col not in df_corrected.columns:
            logger.warning(f"Colonna '{col}' non trovata nel DataFrame, saltata")
            continue

        values = df_corrected[col].values.astype(float)
        n = len(values)

        # Crea una maschera per i punti validi (non-spike)
        valid_mask = np.ones(n, dtype=bool)
        valid_mask[spike_indices] = False

        # Indici dei punti validi
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) < 2:
            logger.warning(f"Colonna '{col}': troppi pochi punti validi per interpolare")
            continue

        # Valori validi per l'interpolazione
        valid_values = values[valid_mask]

        try:
            # Prova interpolazione cubica
            interp_func = interpolate.interp1d(
                valid_indices,
                valid_values,
                kind='cubic',
                bounds_error=False,
                fill_value='extrapolate'
            )

            # Correggi gli spike
            corrected_values = interp_func(spike_indices)

            # Verifica che i valori interpolati siano ragionevoli
            # (non negativi per concentrazioni, entro range plausibile)
            min_valid = np.min(valid_values)
            max_valid = np.max(valid_values)
            range_margin = (max_valid - min_valid) * 0.5

            # Clip dei valori interpolati per evitare extrapolazioni estreme
            corrected_values = np.clip(
                corrected_values,
                min_valid - range_margin,
                max_valid + range_margin
            )

            # Applica le correzioni
            values[spike_indices] = corrected_values
            df_corrected[col] = values

            total_corrected += len(spike_indices)

            logger.debug(f"Colonna '{col}': {len(spike_indices)} spike interpolati")

        except Exception as e:
            # Fallback a interpolazione lineare
            logger.warning(f"Interpolazione cubica fallita per '{col}': {e}. "
                          "Uso interpolazione lineare.")
            try:
                interp_func = interpolate.interp1d(
                    valid_indices,
                    valid_values,
                    kind='linear',
                    bounds_error=False,
                    fill_value='extrapolate'
                )
                corrected_values = interp_func(spike_indices)
                values[spike_indices] = corrected_values
                df_corrected[col] = values
                total_corrected += len(spike_indices)
            except Exception as e2:
                logger.error(f"Anche interpolazione lineare fallita per '{col}': {e2}")

    if verbose:
        logger.info(f"Interpolazione completata: {total_corrected} valori corretti")

    return df_corrected


def clean_block(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    trim_percent: float = 0.2,
    spike_window: int = 5,
    spike_threshold: float = 3.0,
    verbose: bool = False
) -> Optional[pd.DataFrame]:
    """
    Applica la pipeline completa di pulizia a un blocco temporale.

    Pipeline di pulizia:
    1. Trimming: rimuove primo/ultimo X% del blocco (transitori)
    2. Spike detection: identifica outlier con finestra mobile
    3. Spike interpolation: corregge outlier con interpolazione cubica

    Args:
        df: DataFrame completo con i dati PTR-MS
        start_idx: Indice di inizio del blocco
        end_idx: Indice di fine del blocco (incluso)
        trim_percent: Percentuale da rimuovere per il trimming (default: 0.2)
        spike_window: Dimensione finestra per rilevamento spike (default: 5)
        spike_threshold: Soglia in sigma per spike (default: 3.0)
        verbose: Se True, mostra log dettagliati (default: False)

    Returns:
        Optional[pd.DataFrame]: DataFrame del blocco pulito, oppure None se
                                il blocco e' troppo piccolo dopo il trimming

    Example:
        >>> blocks = detect_blocks(df)
        >>> start, end = blocks[0]
        >>> df_clean = clean_block(df, start, end)
        >>> if df_clean is not None:
        ...     print(f"Blocco pulito: {len(df_clean)} righe")
    """
    block_id = f"{start_idx}-{end_idx}"
    original_size = end_idx - start_idx + 1

    if verbose:
        logger.info(f"Pulizia blocco idx {block_id} ({original_size} righe)")

    # Step 1: Trimming
    trim_result = trim_block(df, start_idx, end_idx, trim_percent=trim_percent)

    if trim_result is None:
        logger.debug(f"Blocco {block_id} scartato (troppo piccolo dopo trimming)")
        return None

    new_start, new_end = trim_result

    # Estrai il blocco trimmato
    df_block = df.iloc[new_start:new_end + 1].copy()
    df_block = df_block.reset_index(drop=True)

    # Step 2: Rilevamento spike
    spike_indices = detect_spikes_dataframe(
        df_block,
        window=spike_window,
        threshold_std=spike_threshold,
        verbose=verbose
    )

    # Conta totale spike
    total_spikes = sum(len(v) for v in spike_indices.values())

    # Step 3: Interpolazione spike
    if total_spikes > 0:
        df_block = interpolate_spikes(df_block, spike_indices, verbose=verbose)

    logger.debug(f"Blocco {block_id}: {original_size} -> {len(df_block)} righe, {total_spikes} spike corretti")

    return df_block


def clean_all_blocks(
    df: pd.DataFrame,
    blocks: list[tuple[int, int]],
    trim_percent: float = 0.2,
    spike_window: int = 5,
    spike_threshold: float = 3.0
) -> list[pd.DataFrame]:
    """
    Applica la pulizia a tutti i blocchi e ritorna la lista dei blocchi puliti.

    Args:
        df: DataFrame completo con i dati PTR-MS
        blocks: Lista di tuple (start_idx, end_idx) dei blocchi
        trim_percent: Percentuale trimming
        spike_window: Finestra rilevamento spike
        spike_threshold: Soglia spike

    Returns:
        list[pd.DataFrame]: Lista dei blocchi puliti (esclude blocchi troppo piccoli)
    """
    logger.info(f"Pulizia di {len(blocks)} blocchi...")

    cleaned_blocks = []
    skipped = 0

    for i, (start, end) in enumerate(blocks):
        logger.info(f"\n--- Blocco {i+1}/{len(blocks)} ---")

        df_clean = clean_block(
            df, start, end,
            trim_percent=trim_percent,
            spike_window=spike_window,
            spike_threshold=spike_threshold
        )

        if df_clean is not None:
            cleaned_blocks.append(df_clean)
        else:
            skipped += 1

    logger.info(f"\nPulizia completata: {len(cleaned_blocks)} blocchi validi, "
               f"{skipped} scartati")

    return cleaned_blocks


# =============================================================================
# Test del modulo
# =============================================================================
if __name__ == "__main__":
    """
    Test del modulo con il file di esempio.
    Eseguire con: python ptrms_cleaning.py
    """
    from pathlib import Path
    from ptrms_parser import load_and_prepare
    from ptrms_blocks import detect_blocks, get_block_info

    # Configura logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Percorso file di test
    test_file = Path(__file__).parent / "Run" / "stanza_01.h5_export.csv"

    print("\n" + "=" * 70)
    print("TEST MODULO ptrms_cleaning.py")
    print("=" * 70)

    if not test_file.exists():
        print(f"File di test non trovato: {test_file}")
        exit(1)

    # Carica i dati
    print("\n--- Caricamento dati ---")
    df = load_and_prepare(test_file)
    print(f"Dati caricati: {len(df)} righe, {len(df.columns)} colonne")

    # Rileva blocchi
    print("\n--- Rilevamento blocchi ---")
    blocks = detect_blocks(df, gap_threshold_seconds=300)
    print(f"Blocchi rilevati: {len(blocks)}")

    # Test 1: trim_block
    print("\n" + "=" * 70)
    print("TEST 1: trim_block")
    print("=" * 70)

    start, end = blocks[0]
    print(f"Blocco originale: idx {start}-{end} ({end-start+1} righe)")

    result = trim_block(df, start, end, trim_percent=0.2)
    if result:
        new_start, new_end = result
        print(f"Blocco trimmato: idx {new_start}-{new_end} ({new_end-new_start+1} righe)")
    else:
        print("Blocco scartato (troppo piccolo)")

    # Test 2: detect_spikes su singola colonna
    print("\n" + "=" * 70)
    print("TEST 2: detect_spikes (singola colonna)")
    print("=" * 70)

    # Crea serie con spike artificiale per test
    test_series = pd.Series([1.0, 1.1, 1.0, 0.9, 100.0, 1.1, 1.0, 0.9, 1.0])
    print(f"Serie di test: {test_series.tolist()}")

    spikes = detect_spikes(test_series, window=3, threshold_std=2.0)
    print(f"Spike rilevati agli indici: {spikes}")
    assert 4 in spikes, "Lo spike a indice 4 (valore 100) dovrebbe essere rilevato"
    print("OK: spike artificiale rilevato correttamente")

    # Test 3: detect_spikes_dataframe
    print("\n" + "=" * 70)
    print("TEST 3: detect_spikes_dataframe")
    print("=" * 70)

    # Usa primo blocco (trimmato) per il test
    if result:
        df_test_block = df.iloc[new_start:new_end+1].copy().reset_index(drop=True)
        print(f"Analisi blocco: {len(df_test_block)} righe")

        all_spikes = detect_spikes_dataframe(df_test_block, window=5, threshold_std=3.0)

        n_cols_with_spikes = sum(1 for v in all_spikes.values() if len(v) > 0)
        total_spikes = sum(len(v) for v in all_spikes.values())
        print(f"Colonne con spike: {n_cols_with_spikes}")
        print(f"Totale spike: {total_spikes}")

    # Test 4: interpolate_spikes
    print("\n" + "=" * 70)
    print("TEST 4: interpolate_spikes")
    print("=" * 70)

    # Crea DataFrame di test con spike
    df_spike_test = pd.DataFrame({
        'RelTime': range(10),
        'mass1': [1.0, 1.1, 1.0, 0.9, 50.0, 1.1, 1.0, 0.9, 1.0, 1.1],  # spike a idx 4
        'mass2': [2.0, 2.1, 2.0, 2.1, 2.0, 200.0, 2.1, 2.0, 2.1, 2.0]  # spike a idx 5
    })

    print("DataFrame con spike artificiali:")
    print(f"  mass1[4] = {df_spike_test['mass1'].iloc[4]} (spike)")
    print(f"  mass2[5] = {df_spike_test['mass2'].iloc[5]} (spike)")

    # Rileva e correggi
    spikes_test = detect_spikes_dataframe(df_spike_test, window=3, threshold_std=2.0)
    df_corrected = interpolate_spikes(df_spike_test, spikes_test)

    print("Dopo interpolazione:")
    print(f"  mass1[4] = {df_corrected['mass1'].iloc[4]:.2f} (corretto)")
    print(f"  mass2[5] = {df_corrected['mass2'].iloc[5]:.2f} (corretto)")

    # Verifica che i valori siano ragionevoli
    assert df_corrected['mass1'].iloc[4] < 5.0, "mass1[4] dovrebbe essere ~1.0"
    assert df_corrected['mass2'].iloc[5] < 5.0, "mass2[5] dovrebbe essere ~2.0"
    print("OK: spike corretti con valori ragionevoli")

    # Test 5: clean_block completo
    print("\n" + "=" * 70)
    print("TEST 5: clean_block (pipeline completa)")
    print("=" * 70)

    start, end = blocks[0]
    df_cleaned = clean_block(df, start, end, trim_percent=0.2)

    if df_cleaned is not None:
        print(f"\nRisultato finale:")
        print(f"  Blocco originale: {end-start+1} righe")
        print(f"  Blocco pulito: {len(df_cleaned)} righe")
        print(f"  Colonne: {list(df_cleaned.columns[:5])} ...")
        print("\nPrime 3 righe del blocco pulito:")
        print(df_cleaned.head(3).to_string())
    else:
        print("Blocco scartato")

    print("\n" + "=" * 70)
    print("TEST COMPLETATO CON SUCCESSO!")
    print("=" * 70)
