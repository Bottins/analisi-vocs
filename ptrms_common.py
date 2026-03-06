"""
PTR-MS Common Utilities
=======================

Modulo con utility comuni condivise da tutti i moduli PTR-MS.

Contiene:
- Funzioni di conversione tipi per JSON
- Utility statistiche sicure
- Funzioni per estrazione colonne masse
- Funzioni di formattazione

Autore: PHD-VOCS Project
"""

import re
import logging
from typing import Any, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Conversione Tipi per JSON
# =============================================================================
def to_python_type(val: Any) -> Any:
    """
    Converte tipi numpy/pandas a tipi Python nativi per serializzazione JSON.

    Args:
        val: Valore da convertire (può essere numpy.float64, numpy.int64, etc.)

    Returns:
        Valore convertito in tipo Python nativo (float, int, bool, etc.)
    """
    if val is None:
        return None
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    if isinstance(val, (np.floating, np.float64, np.float32)):
        if np.isnan(val) or np.isinf(val):
            return None  # JSON non supporta NaN/Inf
        return float(val)
    if isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, np.ndarray):
        return [to_python_type(v) for v in val]
    if isinstance(val, (list, tuple)):
        return [to_python_type(v) for v in val]
    if isinstance(val, dict):
        return {k: to_python_type(v) for k, v in val.items()}
    if isinstance(val, pd.Timestamp):
        return val.isoformat()
    return val


# =============================================================================
# Utility Statistiche Sicure
# =============================================================================
def safe_mean(arr) -> float:
    """Calcola la media in modo sicuro, gestendo array vuoti e NaN."""
    if arr is None or len(arr) == 0:
        return 0.0
    result = np.nanmean(arr)
    return 0.0 if np.isnan(result) else float(result)


def safe_std(arr) -> float:
    """Calcola la deviazione standard in modo sicuro."""
    if arr is None or len(arr) < 2:
        return 0.0
    result = np.nanstd(arr)
    return 0.0 if np.isnan(result) else float(result)


def safe_median(arr) -> float:
    """Calcola la mediana in modo sicuro."""
    if arr is None or len(arr) == 0:
        return 0.0
    result = np.nanmedian(arr)
    return 0.0 if np.isnan(result) else float(result)


def safe_percentile(arr, percentile: float) -> float:
    """Calcola un percentile in modo sicuro."""
    if arr is None or len(arr) == 0:
        return 0.0
    result = np.nanpercentile(arr, percentile)
    return 0.0 if np.isnan(result) else float(result)


# =============================================================================
# Funzioni per Colonne Masse
# =============================================================================
def get_mass_columns(df: pd.DataFrame) -> List[str]:
    """
    Estrae le colonne che rappresentano masse (formato mXX.XX).

    Args:
        df: DataFrame con colonne

    Returns:
        Lista di nomi colonne che rappresentano masse
    """
    return [c for c in df.columns if re.match(r'^m\d+\.?\d*$', c)]


def extract_mass_value(column_name: str) -> Optional[float]:
    """
    Estrae il valore numerico della massa dal nome colonna.

    Args:
        column_name: Nome colonna (es. "m59.05" o "m59.055")

    Returns:
        float: Valore numerico della massa, o None se non e' una colonna massa
    """
    match = re.match(r'^m(\d+\.?\d*)$', column_name)
    if match:
        return float(match.group(1))
    return None


def get_top_masses(df: pd.DataFrame, n: int = 10) -> List[str]:
    """
    Trova le N masse con valori medi piu alti.

    Args:
        df: DataFrame con colonne masse
        n: Numero di masse da restituire

    Returns:
        Lista dei nomi delle top N masse
    """
    mass_cols = get_mass_columns(df)
    if not mass_cols:
        return []

    means = {}
    for col in mass_cols:
        values = df[col].astype(float).values
        valid_values = values[~np.isnan(values)]
        if len(valid_values) > 0:
            means[col] = np.mean(valid_values)

    sorted_masses = sorted(means.items(), key=lambda x: x[1], reverse=True)
    return [m[0] for m in sorted_masses[:n]]


def sort_mass_columns(columns: List[str]) -> List[str]:
    """
    Ordina le colonne masse per valore numerico.

    Args:
        columns: Lista di nomi colonne

    Returns:
        Lista ordinata per valore massa
    """
    def get_mass_val(col):
        val = extract_mass_value(col)
        return val if val is not None else float('inf')

    return sorted(columns, key=get_mass_val)


# =============================================================================
# Funzioni di Formattazione
# =============================================================================
def fmt_val(val, fmt_str: str = ".4f") -> str:
    """
    Formatta un valore numerico per visualizzazione.

    Args:
        val: Valore da formattare
        fmt_str: Stringa formato (es. ".4f", ".2e")

    Returns:
        Stringa formattata o "N/A" se non valido
    """
    if val is None or val == 'N/A' or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    try:
        return f"{float(val):{fmt_str}}"
    except (ValueError, TypeError):
        return str(val)


def fmt_duration(seconds: float) -> str:
    """
    Formatta una durata in secondi in formato leggibile.

    Args:
        seconds: Durata in secondi

    Returns:
        Stringa formattata (es. "2h 30m 15s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def fmt_size(bytes_size: int) -> str:
    """
    Formatta una dimensione in bytes in formato leggibile.

    Args:
        bytes_size: Dimensione in bytes

    Returns:
        Stringa formattata (es. "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f} TB"


# =============================================================================
# Metadata Columns
# =============================================================================
METADATA_COLUMNS = [
    'block_id',
    'start_time',
    'end_time',
    'duration_seconds',
    'n_samples_original',
    'n_samples_after_trim',
    'source_file',
    'AbsTime',
    'RelTime'
]


def get_metadata_columns(df: pd.DataFrame) -> List[str]:
    """Restituisce le colonne metadata presenti nel DataFrame."""
    return [c for c in df.columns if c in METADATA_COLUMNS]


def get_data_columns(df: pd.DataFrame) -> List[str]:
    """Restituisce le colonne dati (non metadata) del DataFrame."""
    metadata = set(METADATA_COLUMNS)
    return [c for c in df.columns if c not in metadata]
