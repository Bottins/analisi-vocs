"""
PTR-MS Analysis Module
======================

Modulo per l'analisi avanzata dei dati PTR-MS.

Funzionalita:
- Sottrazione dell'ambiente (controllo/camera vuota)
- Aggregazione di repliche multiple
- Confronto tra trattamenti

Autore: PHD-VOCS Project
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

# Import funzioni comuni
from ptrms_common import (
    to_python_type,
    safe_mean,
    safe_std,
    extract_mass_value,
)

# Configurazione logging
logger = logging.getLogger(__name__)


def normalize_mass_columns(df: pd.DataFrame, tolerance: float = 0.01) -> pd.DataFrame:
    """
    Normalizza i nomi delle colonne masse arrotondando a 2 decimali.

    Questo permette il matching tra file con leggere differenze nei nomi masse.

    Args:
        df: DataFrame con colonne masse
        tolerance: Tolleranza per l'arrotondamento (default: 0.01)

    Returns:
        pd.DataFrame: DataFrame con colonne masse normalizzate
    """
    new_columns = {}
    mass_columns = {}

    for col in df.columns:
        mass_val = extract_mass_value(col)
        if mass_val is not None:
            # Arrotonda a 2 decimali
            rounded = round(mass_val, 2)
            new_name = f"m{rounded:.2f}"

            if new_name in mass_columns:
                # Colonna duplicata dopo normalizzazione - media
                mass_columns[new_name].append(col)
            else:
                mass_columns[new_name] = [col]
                new_columns[col] = new_name
        else:
            new_columns[col] = col

    # Crea nuovo DataFrame con colonne normalizzate
    result_data = {}

    # Prima le colonne non-massa
    for col in df.columns:
        if col not in new_columns or new_columns[col] == col:
            if extract_mass_value(col) is None:
                result_data[col] = df[col].values

    # Poi le colonne massa (con eventuale media dei duplicati)
    for new_name, orig_cols in mass_columns.items():
        if len(orig_cols) == 1:
            result_data[new_name] = df[orig_cols[0]].values
        else:
            result_data[new_name] = df[orig_cols].mean(axis=1).values
            logger.debug(f"Media di {len(orig_cols)} colonne per {new_name}")

    return pd.DataFrame(result_data)


def subtract_environment(
    tesi_df: pd.DataFrame,
    ambiente_df: pd.DataFrame,
    clip_negative: bool = True,
    replicate_blank: bool = False
) -> tuple[pd.DataFrame, dict]:
    """
    Sottrae i valori dell'ambiente (controllo) dalla tesi (campione).

    Per analizzare i VOCs specifici di un campione, si sottrae l'effetto
    della camera vuota o del controllo.

    Args:
        tesi_df: DataFrame con le mediane della tesi (campione)
        ambiente_df: DataFrame con le mediane dell'ambiente (controllo)
        clip_negative: Se True, i valori negativi vengono sostituiti con 0
        replicate_blank: Se True, replica ciclicamente il blank per coprire tutti i blocchi della tesi

    Returns:
        tuple[pd.DataFrame, dict]: Tupla contenente:
            - DataFrame con i valori sottratti
            - Dizionario con statistiche dell'operazione

    Note:
        - Le colonne presenti solo in un file vengono gestite con 0 nell'altro
        - La tolleranza sui nomi masse e' di 0.01 (gia' normalizzati)
        - Le colonne metadata (block_id, ecc.) vengono preservate dalla tesi
        - Se replicate_blank=True e il blank ha meno blocchi della tesi,
          il blank viene replicato ciclicamente per coprire tutti i blocchi

    Example:
        >>> result_df, stats = subtract_environment(tesi_medians, ambiente_medians)
        >>> print(f"Colonne con differenza > 0: {stats['n_positive_cols']}")
    """
    logger.info("=" * 50)
    logger.info("SOTTRAZIONE AMBIENTE")
    logger.info("=" * 50)

    # Normalizza entrambi i DataFrame
    logger.info("Normalizzazione colonne masse...")
    tesi_norm = normalize_mass_columns(tesi_df.copy())
    ambiente_norm = normalize_mass_columns(ambiente_df.copy())

    # Identifica colonne metadata vs masse
    metadata_cols = ['block_id', 'start_time', 'end_time', 'duration_seconds',
                    'n_samples_original', 'n_samples_after_trim']

    tesi_mass_cols = set(c for c in tesi_norm.columns if extract_mass_value(c) is not None)
    ambiente_mass_cols = set(c for c in ambiente_norm.columns if extract_mass_value(c) is not None)

    # Trova tutte le masse uniche
    all_mass_cols = tesi_mass_cols | ambiente_mass_cols
    only_in_tesi = tesi_mass_cols - ambiente_mass_cols
    only_in_ambiente = ambiente_mass_cols - tesi_mass_cols
    common_cols = tesi_mass_cols & ambiente_mass_cols

    logger.info(f"Masse in tesi: {len(tesi_mass_cols)}")
    logger.info(f"Masse in ambiente: {len(ambiente_mass_cols)}")
    logger.info(f"Masse comuni: {len(common_cols)}")
    logger.info(f"Solo in tesi: {len(only_in_tesi)}")
    logger.info(f"Solo in ambiente: {len(only_in_ambiente)}")

    # Prepara risultato - mantieni metadata dalla tesi
    result_data = {}
    for col in metadata_cols:
        if col in tesi_norm.columns:
            result_data[col] = tesi_norm[col].values

    # Modalita' sottrazione: replicazione ciclica o media
    n_tesi_blocks = len(tesi_norm)
    n_ambiente_blocks = len(ambiente_norm)
    use_cyclic_replication = replicate_blank and n_ambiente_blocks < n_tesi_blocks

    if use_cyclic_replication:
        logger.info(f"Blank piccolo: {n_ambiente_blocks} blocchi -> replicato ciclicamente per {n_tesi_blocks} blocchi tesi")
        # Non calcolare la media, useremo i valori blocco per blocco
        ambiente_means = None
    elif n_ambiente_blocks > 1:
        ambiente_means = ambiente_norm[list(ambiente_mass_cols)].mean()
        logger.info(f"Ambiente con {n_ambiente_blocks} blocchi - uso media")
    else:
        ambiente_means = ambiente_norm[list(ambiente_mass_cols)].iloc[0]

    # Sottrazione per ogni massa
    n_negative_cells = 0
    n_positive_cols = 0
    significant_cols = []
    n_replications = 0

    for mass_col in sorted(all_mass_cols):
        if mass_col in tesi_mass_cols:
            tesi_values = tesi_norm[mass_col].values
        else:
            tesi_values = np.zeros(n_tesi_blocks)

        if mass_col in ambiente_mass_cols:
            if use_cyclic_replication:
                # Sottrazione blocco per blocco con replicazione ciclica
                ambiente_block_values = ambiente_norm[mass_col].values
                ambiente_values = np.array([
                    ambiente_block_values[i % n_ambiente_blocks]
                    for i in range(n_tesi_blocks)
                ])
                n_replications = n_tesi_blocks // n_ambiente_blocks
            else:
                # Sottrazione con valore medio (comportamento originale)
                ambiente_values = np.full(n_tesi_blocks, ambiente_means[mass_col])
        else:
            ambiente_values = np.zeros(n_tesi_blocks)

        # Sottrazione
        result_values = tesi_values - ambiente_values

        # Conta negativi prima del clip
        negatives = np.sum(result_values < 0)
        n_negative_cells += negatives

        # Clip se richiesto
        if clip_negative:
            result_values = np.clip(result_values, 0, None)

        result_data[mass_col] = result_values

        # Controlla se c'e' differenza significativa
        if np.any(result_values > 0):
            n_positive_cols += 1
            if np.mean(result_values) > 0.1:  # Soglia arbitraria per "significativo"
                significant_cols.append(mass_col)

    result_df = pd.DataFrame(result_data)

    # Statistiche
    stats = {
        'n_tesi_masses': len(tesi_mass_cols),
        'n_ambiente_masses': len(ambiente_mass_cols),
        'n_common_masses': len(common_cols),
        'n_only_tesi': len(only_in_tesi),
        'n_only_ambiente': len(only_in_ambiente),
        'n_total_masses': len(all_mass_cols),
        'n_negative_cells_clipped': n_negative_cells,
        'n_positive_cols': n_positive_cols,
        'significant_cols': significant_cols[:20],  # Top 20
        'n_rows': len(result_df),
        'blank_replicated': use_cyclic_replication,
        'n_tesi_blocks': n_tesi_blocks,
        'n_ambiente_blocks': n_ambiente_blocks,
        'n_replications': n_replications if use_cyclic_replication else 0
    }

    logger.info(f"Celle negative (clipped a 0): {n_negative_cells}")
    logger.info(f"Colonne con valori > 0: {n_positive_cols}")
    logger.info(f"Colonne significative: {len(significant_cols)}")
    if use_cyclic_replication:
        logger.info(f"Blank replicato {n_replications}x per coprire tutti i blocchi")
    logger.info("=" * 50)

    return result_df, stats


def aggregate_replicates(
    dfs: list[pd.DataFrame],
    file_names: Optional[list[str]] = None
) -> tuple[pd.DataFrame, dict]:
    """
    Aggrega multiple repliche mantenendo la struttura temporale (blocco per blocco).

    Per ogni blocco temporale, calcola mediana e IQR tra le N repliche.
    Il risultato e' una SERIE TEMPORALE con banda di confidenza, non un singolo valore.

    Args:
        dfs: Lista di DataFrame con le mediane delle repliche (output di compute_file_medians)
        file_names: Lista opzionale di nomi file per logging

    Returns:
        tuple[pd.DataFrame, dict]: Tupla contenente:
            - DataFrame con serie temporale: block_id, e per ogni massa le colonne
              {mass}_median, {mass}_q1, {mass}_q3, {mass}_iqr
            - Dizionario con statistiche dell'operazione

    Note:
        - I blocchi sono allineati per indice (blocco 0 di replica 1 = blocco 0 di replica 2)
        - Se le repliche hanno numeri diversi di blocchi, usa il minimo comune
        - Le masse vengono unificate con tolleranza 0.01

    Example:
        >>> agg_df, stats = aggregate_replicates([rep1_df, rep2_df, rep3_df])
        >>> # agg_df ha colonne: block_id, m59.05_median, m59.05_q1, m59.05_q3, ...
        >>> # Ogni riga e' un blocco temporale
    """
    logger.info("=" * 50)
    logger.info("AGGREGAZIONE REPLICHE (SERIE TEMPORALE)")
    logger.info(f"Numero repliche: {len(dfs)}")
    logger.info("=" * 50)

    if len(dfs) == 0:
        logger.error("Nessun DataFrame fornito")
        return pd.DataFrame(), {'error': 'No data'}

    # Normalizza tutti i DataFrame
    normalized_dfs = []
    n_blocks_list = []
    for i, df in enumerate(dfs):
        norm_df = normalize_mass_columns(df.copy())
        normalized_dfs.append(norm_df)
        n_blocks_list.append(len(norm_df))
        if file_names:
            logger.info(f"  Replica {i+1}: {file_names[i]} - {len(norm_df)} blocchi")
        else:
            logger.info(f"  Replica {i+1}: {len(norm_df)} blocchi")

    # Usa il numero minimo di blocchi tra tutte le repliche
    min_blocks = min(n_blocks_list)
    max_blocks = max(n_blocks_list)
    if min_blocks != max_blocks:
        logger.warning(f"Repliche con numero diverso di blocchi ({min_blocks}-{max_blocks}). Uso {min_blocks} blocchi.")

    # Identifica tutte le masse uniche
    all_mass_cols = set()
    for df in normalized_dfs:
        mass_cols = [c for c in df.columns if extract_mass_value(c) is not None]
        all_mass_cols.update(mass_cols)

    all_mass_cols = sorted(all_mass_cols)
    logger.info(f"Masse totali uniche: {len(all_mass_cols)}")

    # Aggregazione vettorizzata: per ogni massa, costruisci un array 3D
    # (n_repliche x min_blocks) e calcola mediana/percentili lungo asse 0.
    result_data = {'block_id': np.arange(min_blocks)}

    for mass_col in all_mass_cols:
        # Raccogli i valori di tutte le repliche in un array 2D
        replica_values = []
        for df in normalized_dfs:
            if mass_col in df.columns:
                vals = df[mass_col].values[:min_blocks].astype(float)
            else:
                vals = np.full(min_blocks, np.nan)
            replica_values.append(vals)

        # shape: (n_repliche, min_blocks)
        arr = np.array(replica_values)

        # Conta i non-NaN per ogni blocco
        counts = np.sum(~np.isnan(arr), axis=0)

        # Calcola statistiche ignorando NaN lungo asse 0 (repliche)
        with np.errstate(all='ignore'):
            medians = np.nanmedian(arr, axis=0)
            q1s = np.nanpercentile(arr, 25, axis=0)
            q3s = np.nanpercentile(arr, 75, axis=0)
            iqrs = q3s - q1s

        # Dove non ci sono valori validi, metti NaN
        no_data = counts == 0
        medians[no_data] = np.nan
        q1s[no_data] = np.nan
        q3s[no_data] = np.nan
        iqrs[no_data] = np.nan

        result_data[f'{mass_col}_median'] = medians
        result_data[f'{mass_col}_q1'] = q1s
        result_data[f'{mass_col}_q3'] = q3s
        result_data[f'{mass_col}_iqr'] = iqrs
        result_data[f'{mass_col}_n'] = counts.astype(float)

    agg_df = pd.DataFrame(result_data)

    # Calcola statistiche riassuntive
    stats = {
        'n_replicates': len(dfs),
        'n_blocks': min_blocks,
        'n_masses': len(all_mass_cols),
        'masses': all_mass_cols,
        'blocks_per_replica': n_blocks_list
    }

    logger.info(f"Blocchi aggregati: {min_blocks}")
    logger.info(f"Masse tracciate: {len(all_mass_cols)}")
    logger.info("=" * 50)

    return agg_df, stats


def get_mass_timeseries(agg_df: pd.DataFrame, mass_name: str) -> dict:
    """
    Estrae la serie temporale per una specifica massa dal DataFrame aggregato.

    Args:
        agg_df: DataFrame aggregato (output di aggregate_replicates)
        mass_name: Nome della massa (es. "m59.05")

    Returns:
        dict: Dizionario con:
            - blocks: lista di block_id
            - median: lista di valori mediana
            - q1: lista di valori q1
            - q3: lista di valori q3
            - iqr: lista di valori iqr
    """
    median_col = f'{mass_name}_median'
    q1_col = f'{mass_name}_q1'
    q3_col = f'{mass_name}_q3'
    iqr_col = f'{mass_name}_iqr'

    if median_col not in agg_df.columns:
        return None

    return {
        'blocks': agg_df['block_id'].tolist(),
        'median': agg_df[median_col].tolist(),
        'q1': agg_df[q1_col].tolist() if q1_col in agg_df.columns else [],
        'q3': agg_df[q3_col].tolist() if q3_col in agg_df.columns else [],
        'iqr': agg_df[iqr_col].tolist() if iqr_col in agg_df.columns else []
    }


def prepare_chart_data(agg_df: pd.DataFrame, top_n: int = 30, stats: dict = None) -> dict:
    """
    Prepara i dati per il grafico Plotly delle masse aggregate.

    Per il nuovo formato time-series, prepara dati per grafici con curve e bande.

    Args:
        agg_df: DataFrame aggregato (output di aggregate_replicates)
        top_n: Numero di masse da includere (le top per mediana media)
        stats: Statistiche dall'aggregazione (contiene lista masse)

    Returns:
        dict: Dati formattati per Plotly.js con serie temporali
    """
    if agg_df.empty:
        return {'type': 'empty'}

    # Identifica le masse disponibili
    mass_cols = set()
    for col in agg_df.columns:
        if col.endswith('_median'):
            mass_name = col.replace('_median', '')
            mass_cols.add(mass_name)

    if not mass_cols:
        return {'type': 'empty'}

    # Calcola la mediana media per ogni massa (per ranking)
    mass_rankings = []
    for mass in mass_cols:
        median_col = f'{mass}_median'
        if median_col in agg_df.columns:
            avg_median = agg_df[median_col].mean()
            mass_rankings.append((mass, avg_median))

    # Ordina per valore e prendi le top N
    mass_rankings.sort(key=lambda x: x[1], reverse=True)
    top_masses = [m[0] for m in mass_rankings[:top_n]]

    # Prepara dati per ogni massa
    blocks = agg_df['block_id'].tolist()
    series_data = {}

    for mass in top_masses:
        ts = get_mass_timeseries(agg_df, mass)
        if ts:
            series_data[mass] = ts

    return {
        'type': 'timeseries',
        'blocks': blocks,
        'series': series_data,
        'top_masses': top_masses
    }


def compare_treatments(
    tesi_a_df: pd.DataFrame,
    tesi_b_df: pd.DataFrame,
    significance_threshold: float = 0.0
) -> tuple[pd.DataFrame, dict]:
    """
    Confronta due tesi/trattamenti e identifica differenze significative.

    Supporta sia il vecchio formato (colonna 'mass') che il nuovo formato
    time-series (colonne {mass}_median, {mass}_q1, etc.).

    Per il formato time-series:
    - Confronta i valori PUNTO PER PUNTO (blocco per blocco)
    - Calcola metriche di errore puntuale (RMSE, MAE)
    - Calcola metriche di errore sull'andamento (trend correlation)

    Args:
        tesi_a_df: DataFrame aggregato della Tesi A
        tesi_b_df: DataFrame aggregato della Tesi B
        significance_threshold: Soglia minima per considerare una differenza significativa

    Returns:
        tuple[pd.DataFrame, dict]: Tupla contenente:
            - DataFrame con il confronto per ogni massa
            - Dizionario con statistiche riassuntive e dati timeseries
    """
    logger.info("=" * 50)
    logger.info("CONFRONTO TRA TESI")
    logger.info("=" * 50)

    def extract_timeseries_data(df: pd.DataFrame, mass_name: str) -> dict:
        """Estrae la serie temporale completa per una massa."""
        median_col = f'{mass_name}_median'
        q1_col = f'{mass_name}_q1'
        q3_col = f'{mass_name}_q3'
        iqr_col = f'{mass_name}_iqr'

        if median_col not in df.columns:
            return None

        medians = df[median_col].values
        q1s = df[q1_col].values if q1_col in df.columns else medians.copy()
        q3s = df[q3_col].values if q3_col in df.columns else medians.copy()
        iqrs = df[iqr_col].values if iqr_col in df.columns else np.zeros_like(medians)

        # Sostituisci NaN con 0 per evitare problemi
        medians = np.nan_to_num(medians, nan=0.0)
        q1s = np.nan_to_num(q1s, nan=0.0)
        q3s = np.nan_to_num(q3s, nan=0.0)
        iqrs = np.nan_to_num(iqrs, nan=0.0)

        return {
            'medians': medians,
            'q1': q1s,
            'q3': q3s,
            'iqr': iqrs,
            'n_blocks': len(medians)
        }

    def extract_masses_from_timeseries(df: pd.DataFrame) -> dict:
        """Estrae dati aggregati per ogni massa dal formato time-series."""
        masses_data = {}
        for col in df.columns:
            if col.endswith('_median'):
                mass_name = col.replace('_median', '')
                ts_data = extract_timeseries_data(df, mass_name)
                if ts_data is None:
                    continue

                medians = ts_data['medians']
                if len(medians) == 0 or np.all(medians == 0):
                    continue

                # Calcola statistiche aggregate (media attraverso i blocchi)
                masses_data[mass_name] = {
                    'median': safe_mean(medians),
                    'median_of_medians': float(np.median(medians)),
                    'min': float(np.min(medians)),
                    'max': float(np.max(medians)),
                    'std': safe_std(medians),
                    'q1': safe_mean(ts_data['q1']),
                    'q3': safe_mean(ts_data['q3']),
                    'iqr': safe_mean(ts_data['iqr']),
                    'n_blocks': ts_data['n_blocks'],
                    'timeseries': ts_data  # Conserva i dati completi per confronto puntuale
                }
        return masses_data

    def extract_masses_from_old_format(df: pd.DataFrame) -> dict:
        """Estrae dati per ogni massa dal vecchio formato (colonna 'mass')."""
        masses_data = {}
        for _, row in df.iterrows():
            mass_name = row['mass']
            median_val = float(row.get('median', 0))
            masses_data[mass_name] = {
                'median': median_val,
                'min': float(row.get('min', 0)),
                'max': float(row.get('max', 0)),
                'std': float(row.get('std', 0)),
                'q1': float(row.get('q1', median_val)),
                'q3': float(row.get('q3', median_val)),
                'iqr': float(row.get('iqr', 0)),
                'n_blocks': 1,
                'timeseries': None
            }
        return masses_data

    def compute_pointwise_metrics(ts_a: dict, ts_b: dict) -> dict:
        """
        Calcola metriche di errore puntuale tra due serie temporali.

        IMPORTANTE: Le metriche vengono calcolate su valori NORMALIZZATI 0-1
        per permettere confronti equi tra masse con scale diverse.

        Metriche calcolate:
        - RMSE: Root Mean Square Error (su valori normalizzati 0-1)
        - MAE: Mean Absolute Error (su valori normalizzati 0-1)
        - Max Error: Errore massimo puntuale (su valori normalizzati 0-1)
        - Pointwise errors: Array degli errori per ogni blocco (valori reali)
        """
        if ts_a is None or ts_b is None:
            return None

        medians_a = ts_a['medians']
        medians_b = ts_b['medians']

        # Allinea le lunghezze (usa il minimo)
        n_points = min(len(medians_a), len(medians_b))
        if n_points == 0:
            return None

        a = medians_a[:n_points]
        b = medians_b[:n_points]

        # === NORMALIZZAZIONE 0-1 per metriche di errore ===
        # Usa min/max combinato di entrambe le serie per normalizzare
        all_values = np.concatenate([a, b])
        min_val = np.min(all_values)
        max_val = np.max(all_values)
        range_val = max_val - min_val

        if range_val > 0:
            a_norm = (a - min_val) / range_val
            b_norm = (b - min_val) / range_val
        else:
            # Valori costanti, nessuna differenza
            a_norm = np.zeros_like(a)
            b_norm = np.zeros_like(b)

        # Errori puntuali su valori normalizzati
        errors_norm = a_norm - b_norm
        abs_errors_norm = np.abs(errors_norm)

        # RMSE su valori normalizzati
        rmse = float(np.sqrt(np.mean(errors_norm ** 2)))
        # MAE su valori normalizzati
        mae = float(np.mean(abs_errors_norm))
        # Max error su valori normalizzati
        max_error = float(np.max(abs_errors_norm))
        max_error_idx = int(np.argmax(abs_errors_norm))

        # Errore relativo medio (percentuale) - su valori reali
        errors_real = a - b
        abs_errors_real = np.abs(errors_real)
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_val = (np.abs(a) + np.abs(b)) / 2
            rel_errors = np.where(mean_val > 0, abs_errors_real / mean_val * 100, 0)
            mape = float(np.mean(rel_errors))

        return {
            'rmse': rmse,  # Su valori normalizzati 0-1
            'mae': mae,    # Su valori normalizzati 0-1
            'max_error': max_error,  # Su valori normalizzati 0-1
            'max_error_block': max_error_idx,
            'mape': mape,  # Mean Absolute Percentage Error (valori reali)
            'n_points': n_points,
            'pointwise_errors': [to_python_type(e) for e in errors_real],  # Errori reali per visualizzazione
            'normalization_range': to_python_type(range_val)
        }

    def compute_trend_metrics(ts_a: dict, ts_b: dict) -> dict:
        """
        Calcola metriche sull'andamento (trend) tra due serie temporali.

        IMPORTANTE: Le metriche vengono calcolate su valori NORMALIZZATI 0-1
        per permettere confronti equi tra masse con scale diverse.

        Metriche calcolate:
        - Correlation: Correlazione tra le due serie (su valori normalizzati)
        - Trend similarity: Similarita degli andamenti (derivate)
        - Phase shift: Eventuale sfasamento temporale
        """
        if ts_a is None or ts_b is None:
            return None

        medians_a = ts_a['medians']
        medians_b = ts_b['medians']

        n_points = min(len(medians_a), len(medians_b))
        if n_points < 3:  # Servono almeno 3 punti per trend
            return None

        a = medians_a[:n_points]
        b = medians_b[:n_points]

        # === NORMALIZZAZIONE 0-1 per metriche di correlazione ===
        all_values = np.concatenate([a, b])
        min_val = np.min(all_values)
        max_val = np.max(all_values)
        range_val = max_val - min_val

        if range_val > 0:
            a_norm = (a - min_val) / range_val
            b_norm = (b - min_val) / range_val
        else:
            a_norm = np.zeros_like(a)
            b_norm = np.zeros_like(b)

        # Correlazione di Pearson su valori normalizzati
        if np.std(a_norm) > 0 and np.std(b_norm) > 0:
            correlation = float(np.corrcoef(a_norm, b_norm)[0, 1])
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0

        # Similarita dei trend (derivate prime) - su valori normalizzati
        diff_a = np.diff(a_norm)
        diff_b = np.diff(b_norm)

        if len(diff_a) > 0 and np.std(diff_a) > 0 and np.std(diff_b) > 0:
            trend_correlation = float(np.corrcoef(diff_a, diff_b)[0, 1])
            if np.isnan(trend_correlation):
                trend_correlation = 0.0
        else:
            trend_correlation = 0.0

        # Direzione concorde: quante volte entrambe salgono o scendono insieme
        same_direction = np.sum((diff_a > 0) == (diff_b > 0)) / len(diff_a) * 100 if len(diff_a) > 0 else 0

        # Cross-correlation per trovare eventuale sfasamento (su valori normalizzati)
        if len(a_norm) >= 5:
            from numpy import correlate
            norm_a_z = (a_norm - np.mean(a_norm)) / (np.std(a_norm) + 1e-10)
            norm_b_z = (b_norm - np.mean(b_norm)) / (np.std(b_norm) + 1e-10)
            xcorr = correlate(norm_a_z, norm_b_z, mode='full')
            lag = int(np.argmax(xcorr) - len(a_norm) + 1)
        else:
            lag = 0

        return {
            'correlation': correlation,
            'trend_correlation': trend_correlation,
            'same_direction_pct': float(same_direction),
            'phase_lag': lag,
            'n_points': n_points
        }

    # Determina il formato e estrai i dati
    is_timeseries_a = any(col.endswith('_median') for col in tesi_a_df.columns)
    is_timeseries_b = any(col.endswith('_median') for col in tesi_b_df.columns)

    if is_timeseries_a:
        logger.info("Tesi A: formato time-series")
        a_data = extract_masses_from_timeseries(tesi_a_df)
    elif 'mass' in tesi_a_df.columns:
        logger.info("Tesi A: formato legacy")
        a_data = extract_masses_from_old_format(tesi_a_df)
    else:
        raise ValueError("Tesi A: formato non riconosciuto")

    if is_timeseries_b:
        logger.info("Tesi B: formato time-series")
        b_data = extract_masses_from_timeseries(tesi_b_df)
    elif 'mass' in tesi_b_df.columns:
        logger.info("Tesi B: formato legacy")
        b_data = extract_masses_from_old_format(tesi_b_df)
    else:
        raise ValueError("Tesi B: formato non riconosciuto")

    masses_a = set(a_data.keys())
    masses_b = set(b_data.keys())

    all_masses = masses_a | masses_b
    common_masses = masses_a & masses_b
    only_in_a = masses_a - masses_b
    only_in_b = masses_b - masses_a

    logger.info(f"Masse in Tesi A: {len(masses_a)}")
    logger.info(f"Masse in Tesi B: {len(masses_b)}")
    logger.info(f"Masse comuni: {len(common_masses)}")
    logger.info(f"Solo in A: {len(only_in_a)}")
    logger.info(f"Solo in B: {len(only_in_b)}")

    # Confronta ogni massa
    comparison_results = []
    timeseries_comparison = {}  # Dati per grafici sovrapposti
    n_significant = 0
    significant_masses = []

    for mass in sorted(all_masses):
        a_info = a_data.get(mass, {})
        b_info = b_data.get(mass, {})

        # Valori di default se mancanti
        a_median = a_info.get('median', 0)
        b_median = b_info.get('median', 0)
        a_iqr = a_info.get('iqr', 0)
        b_iqr = b_info.get('iqr', 0)
        a_q1 = a_info.get('q1', a_median)
        a_q3 = a_info.get('q3', a_median)
        b_q1 = b_info.get('q1', b_median)
        b_q3 = b_info.get('q3', b_median)
        a_std = a_info.get('std', 0)
        b_std = b_info.get('std', 0)

        # Calcola metriche di confronto base
        diff = a_median - b_median
        abs_diff = abs(diff)

        # Calcola effect size (Cohen's d approximation)
        pooled_std = np.sqrt((a_std**2 + b_std**2) / 2) if (a_std > 0 or b_std > 0) else 1
        effect_size = abs_diff / pooled_std if pooled_std > 0 else 0

        # Verifica overlap degli intervalli IQR
        iqr_overlap = not (a_q3 < b_q1 or b_q3 < a_q1)

        # === NUOVE METRICHE: confronto puntuale e trend ===
        ts_a = a_info.get('timeseries')
        ts_b = b_info.get('timeseries')

        pointwise_metrics = compute_pointwise_metrics(ts_a, ts_b)
        trend_metrics = compute_trend_metrics(ts_a, ts_b)

        # Estrai metriche puntuali
        rmse = pointwise_metrics['rmse'] if pointwise_metrics else 0
        mae = pointwise_metrics['mae'] if pointwise_metrics else abs_diff
        mape = pointwise_metrics['mape'] if pointwise_metrics else 0

        # Estrai metriche trend
        correlation = trend_metrics['correlation'] if trend_metrics else 0
        trend_corr = trend_metrics['trend_correlation'] if trend_metrics else 0

        # === SCORE COMBINATO PER RANKING ===
        # Combina RMSE (errore sull'evoluzione, 0-1) e correlazione (similarita, -1 a 1)
        # Score alto = differenza significativa: RMSE alto + correlazione bassa
        # La correlazione viene trasformata: (1 - |corr|) = 0 se perfettamente correlate, 1 se non correlate
        # ranking_score = RMSE + (1 - |correlazione|)
        # Range: 0 (identiche) a 2 (massima differenza)
        ranking_score = rmse + (1 - abs(correlation))

        # Determina significativita (aggiornata con nuove metriche)
        is_significant = (
            abs_diff > significance_threshold and
            (not iqr_overlap or effect_size > 0.5 or rmse > significance_threshold)
        )

        if is_significant:
            n_significant += 1
            significant_masses.append(mass)

        in_a = mass in masses_a
        in_b = mass in masses_b
        mass_value = extract_mass_value(mass) or 0

        # Risultato per questa massa
        result = {
            'mass': mass,
            'mass_value': to_python_type(mass_value),
            'median_a': to_python_type(a_median),
            'median_b': to_python_type(b_median),
            'difference': to_python_type(diff),
            'abs_difference': to_python_type(abs_diff),
            'iqr_a': to_python_type(a_iqr),
            'iqr_b': to_python_type(b_iqr),
            'q1_a': to_python_type(a_q1),
            'q3_a': to_python_type(a_q3),
            'q1_b': to_python_type(b_q1),
            'q3_b': to_python_type(b_q3),
            'effect_size': to_python_type(effect_size),
            'iqr_overlap': bool(iqr_overlap),
            'is_significant': bool(is_significant),
            'in_a': bool(in_a),
            'in_b': bool(in_b),
            'higher_in': 'A' if diff > 0 else ('B' if diff < 0 else 'equal'),
            # Nuove metriche puntuali
            'rmse': to_python_type(rmse),
            'mae': to_python_type(mae),
            'mape': to_python_type(mape),
            # Nuove metriche trend
            'correlation': to_python_type(correlation),
            'trend_correlation': to_python_type(trend_corr),
            # Score combinato per ranking (RMSE + 1-|correlazione|)
            'ranking_score': to_python_type(ranking_score)
        }
        comparison_results.append(result)

        # Prepara dati timeseries per grafici sovrapposti (solo per masse comuni)
        if ts_a is not None and ts_b is not None:
            n_blocks = min(ts_a['n_blocks'], ts_b['n_blocks'])
            timeseries_comparison[mass] = {
                'blocks': list(range(n_blocks)),
                'a': {
                    'median': [to_python_type(v) for v in ts_a['medians'][:n_blocks]],
                    'q1': [to_python_type(v) for v in ts_a['q1'][:n_blocks]],
                    'q3': [to_python_type(v) for v in ts_a['q3'][:n_blocks]]
                },
                'b': {
                    'median': [to_python_type(v) for v in ts_b['medians'][:n_blocks]],
                    'q1': [to_python_type(v) for v in ts_b['q1'][:n_blocks]],
                    'q3': [to_python_type(v) for v in ts_b['q3'][:n_blocks]]
                },
                'pointwise_errors': pointwise_metrics['pointwise_errors'] if pointwise_metrics else [],
                'metrics': {
                    'rmse': to_python_type(rmse),
                    'mae': to_python_type(mae),
                    'correlation': to_python_type(correlation),
                    'trend_correlation': to_python_type(trend_corr)
                }
            }

    # Crea DataFrame risultato
    comparison_df = pd.DataFrame(comparison_results)
    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values('mass_value').reset_index(drop=True)

    # Statistiche riassuntive (con conversione tipi sicura)
    stats = {
        'n_masses_a': to_python_type(len(masses_a)),
        'n_masses_b': to_python_type(len(masses_b)),
        'n_common': to_python_type(len(common_masses)),
        'n_only_a': to_python_type(len(only_in_a)),
        'n_only_b': to_python_type(len(only_in_b)),
        'n_total': to_python_type(len(all_masses)),
        'n_significant': to_python_type(n_significant),
        'pct_significant': to_python_type((n_significant / len(all_masses) * 100) if all_masses else 0),
        'significant_masses': significant_masses[:20],
        # Classifica basata su ranking_score (RMSE + 1-|correlazione|) invece di differenza mediane
        'top_diff_masses': comparison_df.nlargest(10, 'ranking_score')['mass'].tolist() if not comparison_df.empty and 'ranking_score' in comparison_df.columns else [],
        'avg_abs_diff': to_python_type(comparison_df['abs_difference'].mean() if not comparison_df.empty else 0),
        'max_abs_diff': to_python_type(comparison_df['abs_difference'].max() if not comparison_df.empty else 0),
        # Nuove statistiche aggregate
        'avg_rmse': to_python_type(comparison_df['rmse'].mean() if not comparison_df.empty and 'rmse' in comparison_df.columns else 0),
        'avg_correlation': to_python_type(comparison_df['correlation'].mean() if not comparison_df.empty and 'correlation' in comparison_df.columns else 0),
        'avg_ranking_score': to_python_type(comparison_df['ranking_score'].mean() if not comparison_df.empty and 'ranking_score' in comparison_df.columns else 0),
        # Dati timeseries per grafici sovrapposti
        'timeseries_data': timeseries_comparison
    }

    logger.info(f"Masse significativamente diverse: {n_significant}/{len(all_masses)} "
               f"({stats['pct_significant']:.1f}%)")
    logger.info(f"Top masse per ranking (RMSE+1-|corr|): {stats['top_diff_masses'][:5]}")
    logger.info(f"RMSE medio (normalizzato): {stats['avg_rmse']:.4f}")
    logger.info(f"Correlazione media: {stats['avg_correlation']:.3f}")
    logger.info(f"Ranking score medio: {stats['avg_ranking_score']:.4f}")
    logger.info("=" * 50)

    return comparison_df, stats


def prepare_comparison_chart_data(
    comparison_df: pd.DataFrame,
    top_n: int = 20,
    sort_by: str = 'ranking_score',
    timeseries_data: dict = None
) -> dict:
    """
    Prepara i dati per il grafico comparativo Plotly.

    Genera due tipi di visualizzazione:
    1. Grafico a barre affiancate (confronto mediane aggregate)
    2. Grafici timeseries sovrapposti con bande IQR (per ogni massa)

    Args:
        comparison_df: DataFrame di confronto (output di compare_treatments)
        top_n: Numero di masse da includere
        sort_by: Criterio di ordinamento ('ranking_score', 'abs_difference', 'rmse', 'correlation')
                 Default 'ranking_score' = RMSE + (1 - |correlazione|)
        timeseries_data: Dizionario con i dati timeseries per ogni massa

    Returns:
        dict: Dati formattati per grafici Plotly
    """
    if comparison_df.empty:
        return {
            'type': 'empty',
            'bar_chart': {'labels': [], 'a_medians': [], 'b_medians': []},
            'timeseries_charts': {},
            'error_metrics': {}
        }

    # Seleziona top N per il criterio specificato
    if sort_by in comparison_df.columns:
        top_df = comparison_df.nlargest(top_n, sort_by).copy()
    else:
        top_df = comparison_df.head(top_n).copy()

    # Ordina per valore massa per visualizzazione
    top_df = top_df.sort_values('mass_value')

    # 1. Dati per grafico a barre affiancate
    bar_chart_data = {
        'labels': [to_python_type(v) for v in top_df['mass'].tolist()],
        'a_medians': [to_python_type(v) for v in top_df['median_a'].tolist()],
        'b_medians': [to_python_type(v) for v in top_df['median_b'].tolist()],
        'a_q1': [to_python_type(v) for v in top_df['q1_a'].tolist()],
        'a_q3': [to_python_type(v) for v in top_df['q3_a'].tolist()],
        'b_q1': [to_python_type(v) for v in top_df['q1_b'].tolist()],
        'b_q3': [to_python_type(v) for v in top_df['q3_b'].tolist()],
        'a_iqr': [to_python_type(v) for v in top_df['iqr_a'].tolist()],
        'b_iqr': [to_python_type(v) for v in top_df['iqr_b'].tolist()],
        'significant': [to_python_type(v) for v in top_df['is_significant'].tolist()],
        'differences': [to_python_type(v) for v in top_df['difference'].tolist()]
    }

    # 2. Dati per grafici timeseries sovrapposti
    timeseries_charts = {}
    if timeseries_data:
        for mass in top_df['mass'].tolist():
            if mass in timeseries_data:
                ts = timeseries_data[mass]
                timeseries_charts[mass] = {
                    'blocks': ts['blocks'],
                    'series_a': {
                        'name': 'Tesi A',
                        'median': ts['a']['median'],
                        'q1': ts['a']['q1'],
                        'q3': ts['a']['q3'],
                        'color': 'rgba(31, 119, 180, 0.8)',
                        'fill_color': 'rgba(31, 119, 180, 0.2)'
                    },
                    'series_b': {
                        'name': 'Tesi B',
                        'median': ts['b']['median'],
                        'q1': ts['b']['q1'],
                        'q3': ts['b']['q3'],
                        'color': 'rgba(255, 127, 14, 0.8)',
                        'fill_color': 'rgba(255, 127, 14, 0.2)'
                    },
                    'error_band': ts['pointwise_errors'],
                    'metrics': ts['metrics']
                }

    # 3. Metriche di errore aggregate
    error_metrics = {}
    if 'rmse' in top_df.columns:
        error_metrics = {
            'rmse_data': {
                'labels': [to_python_type(v) for v in top_df['mass'].tolist()],
                'values': [to_python_type(v) for v in top_df['rmse'].tolist()]
            },
            'mae_data': {
                'labels': [to_python_type(v) for v in top_df['mass'].tolist()],
                'values': [to_python_type(v) for v in top_df['mae'].tolist()]
            },
            'correlation_data': {
                'labels': [to_python_type(v) for v in top_df['mass'].tolist()],
                'values': [to_python_type(v) for v in top_df['correlation'].tolist()]
            }
        }

    return {
        'type': 'comparison',
        'bar_chart': bar_chart_data,
        'timeseries_charts': timeseries_charts,
        'error_metrics': error_metrics,
        # Backward compatibility
        'labels': bar_chart_data['labels'],
        'a_medians': bar_chart_data['a_medians'],
        'b_medians': bar_chart_data['b_medians'],
        'a_q1': bar_chart_data['a_q1'],
        'a_q3': bar_chart_data['a_q3'],
        'b_q1': bar_chart_data['b_q1'],
        'b_q3': bar_chart_data['b_q3'],
        'a_iqr': bar_chart_data['a_iqr'],
        'b_iqr': bar_chart_data['b_iqr'],
        'significant': bar_chart_data['significant'],
        'differences': bar_chart_data['differences']
    }


# =============================================================================
# Test del modulo
# =============================================================================
if __name__ == "__main__":
    """
    Test del modulo con dati sintetici.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("\n" + "=" * 70)
    print("TEST MODULO ptrms_analysis.py")
    print("=" * 70)

    # Crea dati di test
    print("\n--- Creazione dati di test ---")

    # Tesi con 3 blocchi
    tesi_df = pd.DataFrame({
        'block_id': [0, 1, 2],
        'duration_seconds': [230, 230, 230],
        'm59.05': [10.5, 11.2, 10.8],
        'm69.07': [5.3, 5.1, 5.5],
        'm79.00': [2.1, 2.3, 2.0],  # Solo in tesi
    })

    # Ambiente con 2 blocchi
    ambiente_df = pd.DataFrame({
        'block_id': [0, 1],
        'duration_seconds': [230, 230],
        'm59.05': [3.2, 3.4],
        'm69.07': [1.1, 1.0],
        'm89.00': [0.5, 0.6],  # Solo in ambiente
    })

    print(f"Tesi: {tesi_df.shape}")
    print(f"Ambiente: {ambiente_df.shape}")

    # Test 1: subtract_environment
    print("\n--- Test subtract_environment ---")
    result_df, stats = subtract_environment(tesi_df, ambiente_df)
    print(f"Risultato: {result_df.shape}")
    print(f"Stats: {stats}")
    print("\nRisultato:")
    print(result_df.to_string())

    # Verifica
    assert 'm79.00' in result_df.columns, "m79.00 (solo tesi) dovrebbe essere presente"
    assert 'm89.00' in result_df.columns, "m89.00 (solo ambiente) dovrebbe essere presente"
    assert result_df['m59.05'].iloc[0] == pytest.approx(10.5 - 3.3, rel=0.1), "Sottrazione errata"
    print("\nOK: sottrazione ambiente funziona correttamente")

    # Test 2: aggregate_replicates
    print("\n--- Test aggregate_replicates ---")

    # Crea 3 repliche
    rep1 = pd.DataFrame({
        'm59.05': [10.0, 10.5],
        'm69.07': [5.0, 5.2],
    })
    rep2 = pd.DataFrame({
        'm59.05': [11.0, 11.5],
        'm69.07': [5.5, 5.3],
    })
    rep3 = pd.DataFrame({
        'm59.05': [10.5, 10.8],
        'm69.07': [5.2, 5.4],
        'm79.00': [2.0, 2.1],  # Solo in rep3
    })

    agg_df, stats = aggregate_replicates([rep1, rep2, rep3], ['rep1', 'rep2', 'rep3'])
    print(f"Aggregato: {agg_df.shape}")
    print(f"Stats: {stats}")
    print("\nRisultato:")
    print(agg_df.to_string())

    # Test 3: prepare_chart_data
    print("\n--- Test prepare_chart_data ---")
    chart_data = prepare_chart_data(agg_df, top_n=5)
    print(f"Chart data: {chart_data}")

    print("\n" + "=" * 70)
    print("TEST COMPLETATO CON SUCCESSO!")
    print("=" * 70)
