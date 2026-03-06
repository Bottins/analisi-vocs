"""
PTR-MS Temporal Blocks Detection Module
========================================

Modulo per il rilevamento e l'analisi dei blocchi temporali nei dati PTR-MS.

I dati PTR-MS possono essere:
- Una registrazione continua (es. 1 ora di acquisizione continua)
- Piu blocchi separati (es. 10 minuti ogni ora)

Questo modulo identifica automaticamente i blocchi temporali basandosi
sui gap tra misurazioni consecutive.

Autore: PHD-VOCS Project
"""

import logging
from typing import Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np

# Configurazione logging
logger = logging.getLogger(__name__)


@dataclass
class BlockInfo:
    """
    Informazioni su un singolo blocco temporale.

    Attributes:
        block_id: Identificatore del blocco (0-indexed)
        start_idx: Indice di inizio nel DataFrame
        end_idx: Indice di fine nel DataFrame (incluso)
        n_samples: Numero di campioni nel blocco
        duration_seconds: Durata del blocco in secondi
        start_time: RelTime del primo campione
        end_time: RelTime dell'ultimo campione
        start_abstime: AbsTime del primo campione (se disponibile)
        end_abstime: AbsTime dell'ultimo campione (se disponibile)
    """
    block_id: int
    start_idx: int
    end_idx: int
    n_samples: int
    duration_seconds: float
    start_time: float
    end_time: float
    start_abstime: Optional[float] = None
    end_abstime: Optional[float] = None


def detect_blocks(
    df: pd.DataFrame,
    gap_threshold_seconds: float = 300.0,
    time_column: str = 'RelTime'
) -> list[tuple[int, int]]:
    """
    Rileva i blocchi temporali continui nei dati PTR-MS.

    Analizza la colonna temporale e identifica i "gap" (interruzioni)
    maggiori della soglia specificata. Ogni segmento continuo tra due
    gap costituisce un blocco.

    Args:
        df: DataFrame con i dati PTR-MS (deve contenere la colonna temporale)
        gap_threshold_seconds: Soglia in secondi per considerare un gap
                              come separatore di blocchi (default: 300s = 5 minuti)
        time_column: Nome della colonna temporale da usare (default: 'RelTime')

    Returns:
        list[tuple[int, int]]: Lista di tuple (start_idx, end_idx) per ogni blocco.
                               Gli indici sono inclusivi (end_idx e' l'ultimo elemento del blocco).

    Raises:
        ValueError: Se la colonna temporale non esiste nel DataFrame
        ValueError: Se il DataFrame e' vuoto

    Example:
        >>> df = load_and_prepare("data.csv")
        >>> blocks = detect_blocks(df, gap_threshold_seconds=300)
        >>> print(f"Trovati {len(blocks)} blocchi")
        Trovati 117 blocchi
        >>> print(f"Primo blocco: righe {blocks[0][0]} - {blocks[0][1]}")
        Primo blocco: righe 0 - 23
    """
    # Validazione input
    if df.empty:
        error_msg = "Il DataFrame e' vuoto"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if time_column not in df.columns:
        error_msg = f"Colonna temporale '{time_column}' non trovata. Colonne disponibili: {list(df.columns)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug(f"Rilevamento blocchi temporali (soglia gap: {gap_threshold_seconds}s)")

    # Estrae i valori temporali
    time_values = df[time_column].values
    n_rows = len(time_values)

    # Caso speciale: DataFrame con una sola riga
    if n_rows == 1:
        logger.debug("DataFrame con una sola riga: singolo blocco")
        return [(0, 0)]

    # Calcola le differenze temporali tra righe consecutive
    time_diffs = np.diff(time_values)

    # Trova gli indici dove c'e' un gap maggiore della soglia
    # gap_indices contiene gli indici DOPO i quali c'e' un gap
    gap_indices = np.where(time_diffs > gap_threshold_seconds)[0]

    # Costruisce la lista dei blocchi
    blocks: list[tuple[int, int]] = []

    if len(gap_indices) == 0:
        # Nessun gap: tutti i dati sono un unico blocco continuo
        blocks.append((0, n_rows - 1))
    else:
        # Primo blocco: da 0 fino al primo gap
        blocks.append((0, int(gap_indices[0])))

        # Blocchi intermedi: tra gap consecutivi
        for i in range(len(gap_indices) - 1):
            start = int(gap_indices[i] + 1)
            end = int(gap_indices[i + 1])
            blocks.append((start, end))

        # Ultimo blocco: dal ultimo gap fino alla fine
        blocks.append((int(gap_indices[-1] + 1), n_rows - 1))

    logger.debug(f"Rilevati {len(blocks)} blocchi temporali")

    return blocks


def get_block_info(
    df: pd.DataFrame,
    blocks: list[tuple[int, int]],
    time_column: str = 'RelTime'
) -> dict[int, BlockInfo]:
    """
    Calcola informazioni dettagliate per ogni blocco temporale.

    Per ogni blocco calcola:
    - Durata in secondi
    - Numero di campioni
    - Timestamp di inizio e fine (RelTime e AbsTime se disponibile)

    Args:
        df: DataFrame con i dati PTR-MS
        blocks: Lista di tuple (start_idx, end_idx) come ritornata da detect_blocks()
        time_column: Nome della colonna temporale (default: 'RelTime')

    Returns:
        dict[int, BlockInfo]: Dizionario con chiave = block_id e valore = BlockInfo

    Example:
        >>> blocks = detect_blocks(df)
        >>> info = get_block_info(df, blocks)
        >>> for block_id, bi in info.items():
        ...     print(f"Blocco {block_id}: {bi.n_samples} campioni, {bi.duration_seconds:.1f}s")
        Blocco 0: 24 campioni, 230.0s
        Blocco 1: 24 campioni, 230.0s
    """
    logger.debug(f"Calcolo informazioni per {len(blocks)} blocchi...")

    has_abstime = 'AbsTime' in df.columns
    block_info: dict[int, BlockInfo] = {}

    for block_id, (start_idx, end_idx) in enumerate(blocks):
        # Estrae i dati del blocco
        block_data = df.iloc[start_idx:end_idx + 1]

        # Calcola le metriche
        n_samples = len(block_data)
        start_time = block_data[time_column].iloc[0]
        end_time = block_data[time_column].iloc[-1]
        duration = end_time - start_time

        # AbsTime se disponibile
        start_abstime = None
        end_abstime = None
        if has_abstime:
            start_abstime = block_data['AbsTime'].iloc[0]
            end_abstime = block_data['AbsTime'].iloc[-1]

        # Crea l'oggetto BlockInfo
        info = BlockInfo(
            block_id=block_id,
            start_idx=start_idx,
            end_idx=end_idx,
            n_samples=n_samples,
            duration_seconds=duration,
            start_time=start_time,
            end_time=end_time,
            start_abstime=start_abstime,
            end_abstime=end_abstime
        )

        block_info[block_id] = info

        logger.debug(f"Blocco {block_id}: idx {start_idx}-{end_idx}, "
                    f"{n_samples} campioni, {duration:.1f}s")

    # Log riepilogo
    total_samples = sum(bi.n_samples for bi in block_info.values())
    total_duration = sum(bi.duration_seconds for bi in block_info.values())

    logger.debug(f"Blocchi: {len(blocks)}, campioni totali: {total_samples}, durata: {total_duration:.1f}s")

    return block_info


def print_block_summary(block_info: dict[int, BlockInfo]) -> str:
    """
    Genera una stringa di riepilogo formattata dei blocchi.

    Utile per la visualizzazione nella finestra di log dell'applicazione web.

    Args:
        block_info: Dizionario con le informazioni dei blocchi

    Returns:
        str: Stringa formattata con il riepilogo dei blocchi

    Example:
        >>> info = get_block_info(df, blocks)
        >>> print(print_block_summary(info))
    """
    lines = []
    lines.append("=" * 70)
    lines.append("RIEPILOGO BLOCCHI TEMPORALI")
    lines.append("=" * 70)
    lines.append(f"{'Blocco':<8} {'Indici':<15} {'Campioni':<10} {'Durata (s)':<12} {'RelTime range'}")
    lines.append("-" * 70)

    for block_id in sorted(block_info.keys()):
        bi = block_info[block_id]
        idx_range = f"{bi.start_idx}-{bi.end_idx}"
        time_range = f"{bi.start_time:.1f} - {bi.end_time:.1f}"
        lines.append(f"{bi.block_id:<8} {idx_range:<15} {bi.n_samples:<10} "
                    f"{bi.duration_seconds:<12.1f} {time_range}")

    lines.append("-" * 70)

    # Statistiche riassuntive
    total_samples = sum(bi.n_samples for bi in block_info.values())
    total_duration = sum(bi.duration_seconds for bi in block_info.values())
    n_blocks = len(block_info)

    lines.append(f"TOTALE: {n_blocks} blocchi, {total_samples} campioni, "
                f"{total_duration:.1f}s ({total_duration/3600:.2f}h)")
    lines.append("=" * 70)

    return "\n".join(lines)


# =============================================================================
# Test del modulo
# =============================================================================
if __name__ == "__main__":
    """
    Test del modulo con il file di esempio.
    Eseguire con: python ptrms_blocks.py
    """
    from pathlib import Path

    # Import del parser
    from ptrms_parser import load_and_prepare

    # Configura logging per vedere i messaggi
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Percorso file di test
    test_file = Path(__file__).parent / "Run" / "stanza_01.h5_export.csv"

    print("\n" + "=" * 70)
    print("TEST MODULO ptrms_blocks.py")
    print("=" * 70)

    if not test_file.exists():
        print(f"File di test non trovato: {test_file}")
        exit(1)

    # Carica i dati
    print("\n--- Caricamento dati ---")
    df = load_and_prepare(test_file)
    print(f"Dati caricati: {len(df)} righe")

    # Test 1: detect_blocks con soglia default (300s)
    print("\n--- Test detect_blocks (soglia=300s) ---")
    blocks = detect_blocks(df, gap_threshold_seconds=300)
    print(f"Blocchi rilevati: {len(blocks)}")
    print(f"Primi 5 blocchi: {blocks[:5]}")

    # Test 2: get_block_info
    print("\n--- Test get_block_info ---")
    block_info = get_block_info(df, blocks)

    # Mostra primi 5 blocchi
    print("\nPrimi 5 blocchi:")
    for i in range(min(5, len(block_info))):
        bi = block_info[i]
        print(f"  Blocco {bi.block_id}: {bi.n_samples} campioni, "
              f"{bi.duration_seconds:.1f}s, RelTime {bi.start_time:.1f}-{bi.end_time:.1f}")

    # Test 3: print_block_summary
    print("\n--- Test print_block_summary ---")
    # Mostra solo primi 10 blocchi per brevita
    summary_info = {k: v for k, v in block_info.items() if k < 10}
    print(print_block_summary(summary_info))

    # Test 4: Verifica con soglia diversa
    print("\n--- Test con soglia diversa (1000s) ---")
    blocks_1000 = detect_blocks(df, gap_threshold_seconds=1000)
    print(f"Blocchi con soglia 1000s: {len(blocks_1000)}")

    # Test 5: Test caso singolo blocco (soglia molto alta)
    print("\n--- Test singolo blocco (soglia=1000000s) ---")
    blocks_single = detect_blocks(df, gap_threshold_seconds=1000000)
    print(f"Blocchi con soglia 1000000s: {len(blocks_single)}")
    assert len(blocks_single) == 1, "Dovrebbe esserci un solo blocco"
    print("  OK: singolo blocco come atteso")

    print("\n" + "=" * 70)
    print("TEST COMPLETATO CON SUCCESSO!")
    print("=" * 70)
