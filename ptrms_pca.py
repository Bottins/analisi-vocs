"""
PTR-MS PCA Analysis Module
==========================

Analisi delle Componenti Principali (PCA) sui dati VOC PTR-MS.

OBIETTIVI FISICI
----------------
1. PCA SINGOLA PER TESI
   Matrice: [blocchi x masse], ogni riga = misura a ~5 min, ogni colonna = VOC
   Risultato: loadings = quali VOC cambiano di piu' nel tempo per quella tesi
   Variabilita' spiegata da PC1 = struttura temporale principale (es. ciclo giorno/notte)

2. PCA CONGIUNTA A COPPIE (H0 vs H1 | H2 vs H3)
   Entrambe le tesi proiettate nello STESSO spazio PC (sistema di riferimento comune).
   COSI' i punti di H0 e H1 sono direttamente comparabili.
   Risultato: le traiettorie mostrano se le due tesi si separano nel tempo
   -> si allontanano di giorno (stress fotosintetico?) e si riavvicinano di notte?

3. PCA PER FASCIA ORARIA (slot orario)
   PCA separata per Mattina / Pomeriggio / Notte.
   Risponde: "Quali VOC caratterizzano ogni fascia oraria?"
   Heatmap dei loadings: massa con colore rosso forte la mattina = VOC biogenico

FASCE ORARIE (orario reale ricostruito da RelTime + datetime inizio run)
- Mattina:    06:00 - 14:00
- Pomeriggio: 14:00 - 22:00
- Notte:      22:00 - 06:00

STRUTTURA OUTPUT
- 06_pca_analysis/
    singola_per_tesi/{H0,H1,H2,H3}/  <- scree, scores, loadings, biplot, timeseries
    congiunta_H0_vs_H1/               <- traiettorie 2D/3D, ellissi, distanze
    congiunta_H2_vs_H3/               <- stesso per seconda fase
    per_slot_orario/                  <- heatmap loadings, score plot per slot

Autore: PHD-VOCS Project
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from ptrms_common import extract_mass_value

logger = logging.getLogger(__name__)

# =============================================================================
# COSTANTI
# =============================================================================

# Colori per tesi
TESI_COLORS = {
    "H0":  "#2196F3",    # Blu
    "H1":  "#F44336",    # Rosso
    "H2":  "#4CAF50",    # Verde
    "H3":  "#FF9800",    # Arancione
    "H0_pre": "#90CAF9",
    "H1_pre": "#EF9A9A",
}

# Colori per fascia oraria
SLOT_COLORS = {
    "Mattina":    "#FFC300",    # Giallo sole
    "Pomeriggio": "#E67E22",    # Arancione tramonto
    "Notte":      "#2C3E7A",    # Blu notte
}

SLOT_MARKERS = {
    "Mattina":    "o",   # cerchio
    "Pomeriggio": "s",   # quadrato
    "Notte":      "^",   # triangolo
}

SLOTS_ORDERED = ["Mattina", "Pomeriggio", "Notte"]


# =============================================================================
# FUNZIONI UTILITY
# =============================================================================

def _extract_median_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Estrae la matrice [blocchi x masse] usando solo le colonne _median.

    Returns:
        X: array shape (n_blocchi, n_masse)
        mass_names: lista nomi masse (es. ['m33.03', 'm45.03', ...])
    """
    median_cols = [c for c in df.columns if c.endswith('_median')]
    if not median_cols:
        # Formato diretto (colonne senza suffisso, es. post subtract_environment)
        median_cols = [c for c in df.columns if extract_mass_value(c) is not None]
        mass_names = list(median_cols)
    else:
        mass_names = [c.replace('_median', '') for c in median_cols]

    X = df[median_cols].values.astype(float)
    return X, mass_names


def _assign_time_slot(hour_of_day: float) -> str:
    """
    Assegna la fascia oraria dato un orario reale in ore (0-24).

    - Mattina:    06:00 - 14:00
    - Pomeriggio: 14:00 - 22:00
    - Notte:      22:00 - 06:00 (notte = prima delle 6 o dopo le 22)
    """
    h = hour_of_day % 24
    if 6.0 <= h < 14.0:
        return "Mattina"
    elif 14.0 <= h < 22.0:
        return "Pomeriggio"
    else:
        return "Notte"  # 22:00-24:00 e 00:00-06:00


def _build_time_info(
    df: pd.DataFrame,
    run_start_datetime: Optional[datetime],
    tesi_name: str,
    avg_block_duration_s: Optional[float] = None
) -> pd.DataFrame:
    """
    Costruisce il DataFrame con informazioni temporali per ogni blocco.

    IMPORTANTE sulla fascia oraria:
    - Se df ha colonna 'start_time': usa quei secondi direttamente (RelTime esatto)
    - Altrimenti: stima con avg_block_duration_s * block_id
      avg_block_duration_s DEVE essere calcolato dalla durata reale della fase
      (durata_ore * 3600 / n_blocchi), NON 300s (5 min) che e' la granularita' dei dati
      grezzi, non dei blocchi aggregati che durano ~1 ora ciascuno.

    - Se run_start_datetime e' fornito: aggiunge secondi all'orario di inizio per
      ottenere l'orario reale del giorno -> assegna fascia oraria corretta
    - Se non e' fornito: usa (rel_time % 86400) con assunzione di inizio a mezzanotte

    Args:
        df: DataFrame con block_id e opzionalmente start_time (RelTime in sec)
        run_start_datetime: datetime reale di inizio FASE (gia' corretto per acclim/split)
        tesi_name: nome della tesi
        avg_block_duration_s: durata media di un blocco aggregato in secondi.
            Se None, viene stimata come (range start_time / n_blocchi) se start_time
            e' presente, oppure come 3600s (1h) come fallback ragionevole.
            NON usare il default 300s per dati aggregati.

    Returns:
        DataFrame con: block_id, rel_time_s, day_num, hour_of_day, time_slot, tesi
    """
    n = len(df)
    block_ids = np.arange(n)

    # Tempo relativo in secondi per ogni blocco
    if 'start_time' in df.columns:
        rel_times_s = df['start_time'].values.astype(float)
    else:
        # Stima: ogni blocco dura avg_block_duration_s secondi
        # avg_block_duration_s deve essere fornito calcolato dalla durata reale della fase
        # (durata_ore * 3600 / n_blocchi). Se non fornito usiamo 3600s (1h) come
        # fallback ragionevole per dati aggregati PTR-MS (meglio di 300s).
        if avg_block_duration_s is None:
            avg_block_duration_s = 3600.0
            logger.warning(
                f"  {tesi_name}: avg_block_duration_s non fornito. "
                "Uso 3600s (1h) per blocco come stima. "
                "Fornisci avg_block_durations in run_all_pca per risultati precisi."
            )
        rel_times_s = block_ids * avg_block_duration_s

    # Giorno relativo: 0 = primo giorno della run
    day_nums = (rel_times_s / 86400.0).astype(int)

    # Orario reale del giorno
    if run_start_datetime is not None:
        # Calcolo esatto: aggiunge secondi all'orario di inizio run
        datetimes = [run_start_datetime + timedelta(seconds=float(t)) for t in rel_times_s]
        hours_of_day = np.array([dt.hour + dt.minute / 60.0 + dt.second / 3600.0
                                  for dt in datetimes])
    else:
        # Stima: modulo 24h. ATTENZIONE: non accurato se la run non inizia a mezzanotte.
        # Lasciamo un warning nel logger per segnalarlo.
        logger.warning(
            f"  {tesi_name}: run_start_datetime non fornito. "
            "La fascia oraria e' calcolata assumendo inizio run a mezzanotte. "
            "Fornisci RUN_START_DATETIMES per risultati accurati."
        )
        hours_of_day = (rel_times_s / 3600.0) % 24.0
        datetimes = None

    # Assegna fascia oraria a ogni blocco
    time_slots = [_assign_time_slot(h) for h in hours_of_day]

    result = pd.DataFrame({
        'block_id':    block_ids,
        'rel_time_s':  rel_times_s,
        'day_num':     day_nums,
        'hour_of_day': hours_of_day,
        'time_slot':   time_slots,
        'tesi':        tesi_name,
    })

    if datetimes is not None:
        result['datetime'] = datetimes

    # Log distribuzione fasce per verifica
    counts = result['time_slot'].value_counts()
    logger.info(f"  {tesi_name} fasce orarie: " +
                " | ".join(f"{s}: {counts.get(s, 0)}" for s in SLOTS_ORDERED))

    return result


def _impute_and_scale(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, object, object]:
    """
    Imputa NaN con la mediana di colonna e standardizza (z-score).

    PERCHE' z-score per la PCA di VOC:
    - Masse diverse hanno concentrazioni molto diverse in ppb (es. m33 acqua >> m137 isoprene)
    - Senza standardizzazione, la PC1 sarebbe dominata dai VOC piu' abbondanti
    - Con z-score, ogni VOC contribuisce in base alla sua VARIABILITA' RELATIVA
    - Questo e' il senso fisico corretto: cerchiamo quali VOC variano di piu',
      non quali sono piu' concentrati

    Rimuove prima le colonne completamente NaN (masse non rilevate).
    """
    valid_cols = ~np.all(np.isnan(X), axis=0)
    X_valid = X[:, valid_cols]

    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X_valid)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    return X_scaled, valid_cols, imputer, scaler


# =============================================================================
# PLOT HELPER RIUTILIZZABILI
# =============================================================================

def _plot_loadings(
    loadings: np.ndarray,
    mass_names: List[str],
    explained: np.ndarray,
    label: str,
    output_dir: Path,
    top_n: int = 20,
    mass_to_compound: Optional[Dict[str, str]] = None,
    filename_prefix: str = ""
):
    """
    Bar chart orizzontale dei loadings per PC1 e PC2.

    LETTURA: Loading alto (rosso) = questa massa contribuisce POSITIVAMENTE alla PC.
    Loading basso (blu) = contribuisce negativamente.
    I loadings di PC1 indicano la struttura di variabilita' principale
    (spesso il ciclo giorno/notte nei dati VOC di piante).
    I loadings di PC2 indicano la seconda struttura indipendente
    (spesso la risposta al trattamento o un trend giornaliero).
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, max(7, top_n * 0.45 + 2)))

    for pc_idx, ax in enumerate(axes):
        load_vec = loadings[pc_idx]
        # Seleziona top N per valore assoluto, poi ordina per valore decrescente
        top_idx = np.argsort(np.abs(load_vec))[::-1][:top_n]
        top_idx_sorted = top_idx[np.argsort(load_vec[top_idx])[::-1]]

        masses_top = [mass_names[i] for i in top_idx_sorted]
        loads_top  = load_vec[top_idx_sorted]

        labels = []
        for m in masses_top:
            if mass_to_compound and m in mass_to_compound:
                compound = mass_to_compound[m]
                short = compound.split('|')[0].strip()
                short = short[:28] + '...' if len(short) > 28 else short
                labels.append(f"{m}  {short}")
            else:
                labels.append(m)

        colors = ['#C62828' if v > 0 else '#1565C0' for v in loads_top]
        ax.barh(range(len(loads_top)), loads_top, color=colors, alpha=0.85,
                edgecolor='white', linewidth=0.4)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color='black', linewidth=1.0)
        ax.set_xlabel('Loading (contributo alla componente)', fontsize=9)
        pct = explained[pc_idx] * 100
        ax.set_title(
            f'PC{pc_idx+1} — {pct:.1f}% varianza spiegata\n'
            f'Top {top_n} masse per |loading| — {label}',
            fontsize=10
        )
        ax.grid(True, axis='x', alpha=0.3)

        # Annotazione interpretativa
        interp = ("Struttura principale di variabilita' temporale\n"
                  "(spesso ciclo giorno/notte)" if pc_idx == 0 else
                  "Seconda struttura di variabilita'\n(spesso risposta al trattamento)")
        ax.text(0.98, 0.02, interp, transform=ax.transAxes, fontsize=7,
                ha='right', va='bottom', style='italic',
                color='gray', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6))

    fig.suptitle(
        f'Loadings PCA — {label}\n'
        'Rosso = carica positiva sulla PC | Blu = carica negativa\n'
        'Le masse piu\' lontane da zero sono i VOC piu\' influenti',
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    fname = f'{filename_prefix}{label}_loadings_PC1_PC2.png'
    fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _draw_confidence_ellipse(x: np.ndarray, y: np.ndarray, ax,
                              color: str, alpha: float = 0.18, n_std: float = 1.0):
    """Ellisse di confidenza a 1-sigma per una nuvola di punti 2D."""
    from matplotlib.patches import Ellipse

    if len(x) < 3:
        return
    cov = np.cov(x, y)
    if np.any(np.isnan(cov)):
        return

    eigenvals, eigenvecs = np.linalg.eigh(cov)
    eigenvals = np.maximum(eigenvals, 0)
    order = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[order]
    eigenvecs = eigenvecs[:, order]

    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    width  = 2 * n_std * np.sqrt(eigenvals[0])
    height = 2 * n_std * np.sqrt(eigenvals[1])

    ellipse = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width, height=height, angle=angle,
        facecolor=color, alpha=alpha,
        edgecolor=color, linewidth=1.8, linestyle='--'
    )
    ax.add_patch(ellipse)


# =============================================================================
# PCA SINGOLA PER TESI
# =============================================================================

def pca_single_tesi(
    df: pd.DataFrame,
    tesi_name: str,
    run_start_datetime: Optional[datetime],
    output_dir: Path,
    n_components: int = 10,
    top_loadings: int = 20,
    mass_to_compound: Optional[Dict[str, str]] = None,
    avg_block_duration_s: Optional[float] = None
) -> Dict:
    """
    PCA su singola tesi: matrice [blocchi x masse].

    SENSO FISICO:
    - Ogni riga = una misura a ~5 minuti
    - Ogni colonna = un VOC (massa m/z, blank-subtracted)
    - PC1 spiega la struttura di variabilita' piu' forte nel tempo
      -> spesso il ciclo giorno/notte per i VOC biogenici
    - Loadings di PC1: VOC con loading alto cambiano molto durante il giorno
      -> probabilmente VOC fotosintesi-dipendenti (isoprene, monoterpeni)
    - Traiettoria nello spazio PC: se il cerchio si chiude in ~24h = ciclo circadiano
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"  PCA singola: {tesi_name}")

    X_raw, mass_names = _extract_median_matrix(df)
    n_blocks, n_masses_raw = X_raw.shape
    logger.info(f"    Matrice: {n_blocks} blocchi x {n_masses_raw} masse")

    if n_blocks < 5:
        logger.warning(f"    {tesi_name}: troppo pochi blocchi ({n_blocks}), PCA saltata")
        return {}

    X_scaled, valid_cols, imputer, scaler = _impute_and_scale(X_raw)
    mass_names_valid = [m for m, v in zip(mass_names, valid_cols) if v]
    n_masses = len(mass_names_valid)

    n_comp = min(n_components, n_blocks - 1, n_masses)
    pca = PCA(n_components=n_comp)
    scores  = pca.fit_transform(X_scaled)
    loadings = pca.components_
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    logger.info(f"    PC1: {explained[0]*100:.1f}%  PC2: {explained[1]*100:.1f}%  "
                f"PC1+PC2: {(explained[0]+explained[1])*100:.1f}%")

    time_info = _build_time_info(df, run_start_datetime, tesi_name,
                                  avg_block_duration_s=avg_block_duration_s)
    for i in range(n_comp):
        time_info[f'PC{i+1}'] = scores[:, i]

    rel_hours = time_info['rel_time_s'].values / 3600.0

    # ---- PLOT 1: Scree plot ----
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(range(1, n_comp + 1), explained * 100,
                  color='steelblue', alpha=0.8, label='Varianza per PC')
    ax.plot(range(1, n_comp + 1), cumulative * 100,
            'ro-', linewidth=2, markersize=6, label='Cumulativa (%)')
    ax.axhline(80, color='gray', linestyle='--', alpha=0.6, linewidth=1, label='80% soglia')
    ax.set_xlabel('Componente Principale (PC)')
    ax.set_ylabel('Varianza Spiegata (%)')
    ax.set_title(
        f'Scree Plot — {tesi_name}\n'
        f'Le prime {np.searchsorted(cumulative, 0.80)+1} PC spiegano >80% della variabilita\' totale',
        fontsize=11
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / f'{tesi_name}_scree.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- PLOT 2: Score plot PC1-PC2 colorato per fascia oraria ----
    _plot_scores_by_slot(scores, time_info, explained, tesi_name, output_dir,
                         title_extra="Ogni punto = un blocco da ~5 min")

    # ---- PLOT 3: Score plot colorato per giorno ----
    _plot_scores_by_day(scores, time_info, explained, tesi_name, output_dir)

    # ---- PLOT 4: Loadings bar chart ----
    _plot_loadings(loadings, mass_names_valid, explained,
                   tesi_name, output_dir, top_n=top_loadings,
                   mass_to_compound=mass_to_compound)

    # ---- PLOT 5: Biplot ----
    _plot_biplot(scores, loadings, mass_names_valid, explained,
                 time_info, tesi_name, output_dir, top_n=12,
                 mass_to_compound=mass_to_compound)

    # ---- PLOT 6: Serie temporale PC1 e PC2 ----
    _plot_pc_timeseries(scores, explained, time_info, tesi_name, output_dir)

    return {
        'tesi': tesi_name,
        'n_blocks': n_blocks,
        'n_masses': n_masses,
        'pca': pca,
        'scores': scores,
        'loadings': loadings,
        'explained': explained,
        'mass_names': mass_names_valid,
        'time_info': time_info,
    }


def _plot_scores_by_slot(
    scores: np.ndarray,
    time_info: pd.DataFrame,
    explained: np.ndarray,
    label: str,
    output_dir: Path,
    title_extra: str = "",
    filename_prefix: str = ""
):
    """Score plot PC1-PC2 con punti colorati per fascia oraria + traiettoria temporale."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Traiettoria temporale (linea sottile grigia)
    ax.plot(scores[:, 0], scores[:, 1], '-', color='lightgray',
            linewidth=0.9, alpha=0.6, zorder=0, label='_nolegend_')

    # Punti per fascia oraria
    slot_counts = {s: 0 for s in SLOTS_ORDERED}
    for slot in SLOTS_ORDERED:
        mask = time_info['time_slot'].values == slot
        n_slot = int(np.sum(mask))
        slot_counts[slot] = n_slot
        if n_slot > 0:
            ax.scatter(scores[mask, 0], scores[mask, 1],
                       c=SLOT_COLORS[slot], marker=SLOT_MARKERS[slot],
                       s=55, alpha=0.78, zorder=2,
                       edgecolors='white', linewidths=0.4,
                       label=f'{slot} ({n_slot} misure)')

    # Marcatori inizio / fine
    ax.plot(scores[0, 0], scores[0, 1], 'D', color='black',
            markersize=11, zorder=6, label='Inizio registrazione')
    ax.plot(scores[-1, 0], scores[-1, 1], '*', color='black',
            markersize=14, zorder=6, label='Fine registrazione')

    ax.set_xlabel(
        f'PC1 — {explained[0]*100:.1f}% varianza spiegata\n'
        '(asse di massima variabilita\' dei VOC)',
        fontsize=10
    )
    ax.set_ylabel(
        f'PC2 — {explained[1]*100:.1f}% varianza spiegata\n'
        '(seconda struttura indipendente)',
        fontsize=10
    )
    ax.set_title(
        f'Score Plot PCA — {label}\n'
        + (f'{title_extra}\n' if title_extra else '') +
        'Punti vicini = profilo VOC simile | Punti lontani = profilo diverso\n'
        'Colori = fascia oraria del blocco di misura',
        fontsize=10
    )
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f'{filename_prefix}{label}_scores_slot.png'
    fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_scores_by_day(
    scores: np.ndarray,
    time_info: pd.DataFrame,
    explained: np.ndarray,
    label: str,
    output_dir: Path,
    filename_prefix: str = ""
):
    """Score plot PC1-PC2 colorato per giorno (0=primo giorno, 1=secondo, ...)."""
    day_nums = time_info['day_num'].values
    unique_days = sorted(np.unique(day_nums))
    n_days = len(unique_days)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(scores[:, 0], scores[:, 1], '-', color='lightgray',
            linewidth=0.9, alpha=0.5, zorder=0)

    cmap = plt.cm.get_cmap('plasma', max(n_days, 2))
    for d in unique_days:
        mask = day_nums == d
        c = cmap(d / max(n_days - 1, 1))
        ax.scatter(scores[mask, 0], scores[mask, 1],
                   color=c, s=50, alpha=0.8, zorder=2,
                   edgecolors='white', linewidths=0.3,
                   label=f'Giorno {d}')

    ax.plot(scores[0, 0], scores[0, 1], 'D', color='black', markersize=10, zorder=6)
    ax.plot(scores[-1, 0], scores[-1, 1], '*', color='black', markersize=14, zorder=6)

    ax.set_xlabel(f'PC1 — {explained[0]*100:.1f}% varianza spiegata', fontsize=10)
    ax.set_ylabel(f'PC2 — {explained[1]*100:.1f}% varianza spiegata', fontsize=10)
    ax.set_title(
        f'Score Plot per Giorno — {label}\n'
        'Se la traiettoria si allontana progressivamente = deriva nel tempo\n'
        'Se i cerchi si sovrappongono = comportamento VOC stabile tra i giorni',
        fontsize=10
    )
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f'{filename_prefix}{label}_scores_giorno.png'
    fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_biplot(
    scores: np.ndarray,
    loadings: np.ndarray,
    mass_names: List[str],
    explained: np.ndarray,
    time_info: pd.DataFrame,
    label: str,
    output_dir: Path,
    top_n: int = 12,
    mass_to_compound: Optional[Dict[str, str]] = None,
    filename_prefix: str = ""
):
    """
    Biplot: scores (punti) + frecce loadings (VOC).

    LETTURA DEL BIPLOT:
    - Punti = misure temporali (ogni ~5 min), colorati per fascia oraria
    - Frecce = VOC; la direzione indica in quale "verso" quel VOC varia di piu'
    - Frecce lunghe = VOC molto influente (loading alto)
    - Punti nella direzione di una freccia = in quel momento quel VOC era alto
    - Frecce parallele = VOC correlati tra loro (variano insieme)
    - Frecce perpendicolari = VOC indipendenti
    """
    fig, ax = plt.subplots(figsize=(11, 9))

    # Normalizza gli scores per visualizzazione
    s_max = np.max(np.abs(scores[:, :2])) + 1e-8

    for slot in SLOTS_ORDERED:
        mask = time_info['time_slot'].values == slot
        if np.any(mask):
            ax.scatter(scores[mask, 0] / s_max, scores[mask, 1] / s_max,
                       c=SLOT_COLORS[slot], marker=SLOT_MARKERS[slot],
                       s=35, alpha=0.55, zorder=2,
                       edgecolors='white', linewidths=0.3, label=slot)

    # Frecce per le top N masse (per importanza combinata PC1+PC2)
    combined_importance = np.sqrt(loadings[0]**2 + loadings[1]**2)
    top_idx = np.argsort(combined_importance)[::-1][:top_n]
    load_scale = 0.85 / (np.max(np.abs(loadings[:2, top_idx])) + 1e-8)

    for idx in top_idx:
        lx = loadings[0, idx] * load_scale
        ly = loadings[1, idx] * load_scale
        ax.annotate('', xy=(lx, ly), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='#B71C1C',
                                   lw=1.8, alpha=0.85))
        m = mass_names[idx]
        label_text = m
        if mass_to_compound and m in mass_to_compound:
            short = mass_to_compound[m].split('|')[0].strip()
            label_text = f"{m}\n{short[:16]}" if len(short) <= 16 else m
        ax.text(lx * 1.12, ly * 1.12, label_text, fontsize=7.5,
                color='#7B0000', ha='center', va='center', fontweight='bold')

    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax.set_xlabel(f'PC1 ({explained[0]*100:.1f}%)', fontsize=10)
    ax.set_ylabel(f'PC2 ({explained[1]*100:.1f}%)', fontsize=10)
    ax.set_title(
        f'Biplot PCA — {label}\n'
        'Punti = misure (colore = fascia oraria) | Frecce = VOC (rosso)\n'
        'Freccia lunga = VOC piu\' influente | Punti nella direzione freccia = quel VOC era alto',
        fontsize=10
    )
    ax.legend(title='Fascia Oraria', fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fname = f'{filename_prefix}{label}_biplot.png'
    fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_pc_timeseries(
    scores: np.ndarray,
    explained: np.ndarray,
    time_info: pd.DataFrame,
    label: str,
    output_dir: Path,
    filename_prefix: str = ""
):
    """
    Serie temporale di PC1 e PC2, con sfondo colorato per fascia oraria.

    LETTURA:
    - Ogni "onda" su PC1 = ciclo di variabilita' VOC (tipicamente 24h)
    - Se l'ampiezza dell'onda cresce nel tempo = il ciclo si amplifica (risposta al trattamento?)
    - Cambi bruschi = eventi puntuali (apertura/chiusura camera, stress acuto)
    - Lo sfondo giallo/arancione/blu indica la fascia oraria di ogni punto
    """
    rel_hours = time_info['rel_time_s'].values / 3600.0
    n = len(scores)

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    for pc_idx, ax in enumerate(axes):
        # Sfondo per fascia oraria (disegna bande continue)
        if n > 1:
            slot_arr = time_info['time_slot'].values
            step = (rel_hours[-1] - rel_hours[0]) / (n - 1) if n > 1 else 1.0
            for i in range(n):
                ax.axvspan(rel_hours[i] - step / 2,
                           rel_hours[i] + step / 2,
                           alpha=0.12, color=SLOT_COLORS[slot_arr[i]], zorder=0)

        ax.plot(rel_hours, scores[:, pc_idx], 'k-', linewidth=1.3, alpha=0.9, zorder=3)
        ax.axhline(0, color='gray', linewidth=0.6, alpha=0.5, zorder=2)
        ax.set_ylabel(
            f'PC{pc_idx+1} Score\n({explained[pc_idx]*100:.1f}% varianza)',
            fontsize=9
        )
        ax.grid(True, alpha=0.25, zorder=1)

        # Segna i giorni sull'asse x
        for d in range(1, int(rel_hours[-1] / 24) + 2):
            ax.axvline(d * 24, color='black', linewidth=0.7, alpha=0.3, linestyle=':')

    axes[-1].set_xlabel('Ore dall\'inizio della registrazione', fontsize=10)
    axes[0].set_title(
        f'Evoluzione Temporale delle Componenti Principali — {label}\n'
        'Ogni "onda" di 24h = ciclo giorno/notte del profilo VOC | '
        'Sfondo: giallo=Mattina, arancione=Pomeriggio, blu=Notte',
        fontsize=10
    )

    # Legenda fasce orarie
    legend_patches = [mpatches.Patch(color=SLOT_COLORS[s], alpha=0.5, label=s)
                      for s in SLOTS_ORDERED]
    axes[0].legend(handles=legend_patches, loc='upper right', fontsize=9, title='Fascia oraria')

    plt.tight_layout()
    fname = f'{filename_prefix}{label}_pc_timeseries.png'
    fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# PCA CONGIUNTA A COPPIE (H0 vs H1 | H2 vs H3)
# =============================================================================

def pca_joint_pair(
    tesi_A_name: str,
    tesi_A_df: pd.DataFrame,
    tesi_B_name: str,
    tesi_B_df: pd.DataFrame,
    run_start_datetimes: Dict[str, Optional[datetime]],
    output_dir: Path,
    n_components: int = 3,
    top_loadings: int = 20,
    mass_to_compound: Optional[Dict[str, str]] = None,
    avg_block_durations: Optional[Dict[str, float]] = None
) -> Dict:
    """
    PCA congiunta su una COPPIA di tesi (es. H0 vs H1 oppure H2 vs H3).

    PERCHE' congiunta (e non due PCA separate)?
    Se faccio PCA separata su H0 e poi su H1, i sistemi di riferimento PC sono diversi:
    la PC1 di H0 non ha niente a che fare con la PC1 di H1.
    Con la PCA congiunta, ENTRAMBE le tesi usano lo stesso asse,
    e cosi' posso CONFRONTARE direttamente dove si trovano.

    COSTRUZIONE DELLA MATRICE:
    - Prendo i blocchi di tesi_A e tesi_B insieme (una dopo l'altra)
    - Ogni riga = un blocco, etichettato con il nome tesi
    - La PCA viene fatta sull'unione, poi i punti vengono separati per tesi
    - Le traiettorie sono DIRETTAMENTE confrontabili

    COSA CERCARE NEI PLOT:
    1. Le traiettorie delle due tesi si sovrappongono? -> VOC simili
    2. Si separano di giorno e si riavvicinano di notte?
       -> Il trattamento influenza i VOC durante il fotoperiodo
    3. La separazione cresce nei giorni successivi?
       -> Effetto cumulativo del trattamento

    Args:
        tesi_A_name, tesi_A_df: prima tesi (es. H0)
        tesi_B_name, tesi_B_df: seconda tesi (es. H1)
        run_start_datetimes: dict {nome_tesi: datetime_inizio}
        output_dir: directory dove salvare i plot
        n_components: numero di PC
        top_loadings: masse da mostrare nei loadings
        mass_to_compound: mapping massa -> nome composto
    """
    pair_name = f"{tesi_A_name}_vs_{tesi_B_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"  PCA congiunta: {pair_name}")

    # --- Trova masse comuni ---
    _, masses_A = _extract_median_matrix(tesi_A_df)
    _, masses_B = _extract_median_matrix(tesi_B_df)
    common_masses = sorted(set(masses_A) & set(masses_B))
    logger.info(f"    Masse comuni: {len(common_masses)}")

    if len(common_masses) < 3:
        logger.error(f"    {pair_name}: troppo poche masse comuni, skip")
        return {}

    # --- Costruisce matrice congiunta ---
    def _project_to_common(df, all_masses_list, tesi_nm):
        X_raw, mass_names = _extract_median_matrix(df)
        idx_map = {m: i for i, m in enumerate(mass_names)}
        X_out = np.full((len(df), len(all_masses_list)), np.nan)
        for j, m in enumerate(all_masses_list):
            if m in idx_map:
                X_out[:, j] = X_raw[:, idx_map[m]]
        return X_out

    X_A = _project_to_common(tesi_A_df, common_masses, tesi_A_name)
    X_B = _project_to_common(tesi_B_df, common_masses, tesi_B_name)

    X_joint = np.vstack([X_A, X_B])

    _abd = avg_block_durations or {}
    time_A = _build_time_info(tesi_A_df, run_start_datetimes.get(tesi_A_name), tesi_A_name,
                               avg_block_duration_s=_abd.get(tesi_A_name))
    time_B = _build_time_info(tesi_B_df, run_start_datetimes.get(tesi_B_name), tesi_B_name,
                               avg_block_duration_s=_abd.get(tesi_B_name))
    time_joint = pd.concat([time_A, time_B], ignore_index=True)

    # --- Imputa e scala ---
    X_scaled, valid_cols, _, _ = _impute_and_scale(X_joint)
    common_masses_valid = [m for m, v in zip(common_masses, valid_cols) if v]

    # --- PCA ---
    n_total = len(X_joint)
    n_comp = min(n_components, n_total - 1, len(common_masses_valid))
    pca = PCA(n_components=n_comp)
    scores  = pca.fit_transform(X_scaled)
    loadings = pca.components_
    explained = pca.explained_variance_ratio_

    logger.info(f"    PC1: {explained[0]*100:.1f}%  PC2: {explained[1]*100:.1f}%")

    # Aggiunge scores al time_info
    for i in range(n_comp):
        time_joint[f'PC{i+1}'] = scores[:, i]

    # n_A = numero blocchi di tesi_A
    n_A = len(tesi_A_df)

    # ---- PLOT 1: Traiettorie 2D colorate per fascia oraria ----
    _plot_pair_trajectories(
        time_joint, explained, tesi_A_name, tesi_B_name, n_A,
        output_dir, pair_name
    )

    # ---- PLOT 2: Score plot per tesi (con convex hull) ----
    _plot_pair_by_tesi(time_joint, explained, output_dir, pair_name)

    # ---- PLOT 3: Traiettorie 3D ----
    if n_comp >= 3:
        _plot_pair_3d(time_joint, explained, output_dir, pair_name)

    # ---- PLOT 4: Loadings ----
    _plot_loadings(loadings, common_masses_valid, explained,
                   pair_name, output_dir, top_n=top_loadings,
                   mass_to_compound=mass_to_compound)

    # ---- PLOT 5: Separazione per slot orario ----
    _plot_pair_slot_separation(time_joint, explained, output_dir, pair_name)

    # ---- PLOT 6: Distanza centroidi per giorno e slot ----
    _plot_centroid_distances(time_joint, explained, output_dir, pair_name,
                             tesi_A_name, tesi_B_name)

    return {
        'pair': pair_name,
        'pca': pca,
        'scores': scores,
        'loadings': loadings,
        'explained': explained,
        'mass_names': common_masses_valid,
        'time_info': time_joint,
        'n_A': n_A,
    }


def _plot_pair_trajectories(
    time_joint: pd.DataFrame,
    explained: np.ndarray,
    tesi_A: str,
    tesi_B: str,
    n_A: int,
    output_dir: Path,
    pair_name: str
):
    """
    Traiettorie delle due tesi nello spazio PC1-PC2.
    Linea colorata per tesi, punti colorati per fascia oraria.
    Frecce indicano la direzione temporale.

    LETTURA:
    - Le due linee (A=colorA, B=colorB) sono le traiettorie temporali
    - Se le linee si separano di mattina = differenza VOC durante il giorno
    - Se si avvicinano di notte = convergenza notturna (buio = meno attivita' fotosintetica?)
    - I pattern 'a spirale' indicano cicli circadiani regolari
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    for tesi_name in [tesi_A, tesi_B]:
        mask = time_joint['tesi'] == tesi_name
        t_sub = time_joint[mask]
        pc1 = t_sub['PC1'].values
        pc2 = t_sub['PC2'].values
        color = TESI_COLORS.get(tesi_name, 'gray')

        # Linea traiettoria
        ax.plot(pc1, pc2, '-', color=color, alpha=0.38, linewidth=1.4, zorder=1)

        # Punti per fascia oraria
        for slot in SLOTS_ORDERED:
            mask_slot = t_sub['time_slot'].values == slot
            n_s = int(np.sum(mask_slot))
            if n_s > 0:
                ax.scatter(pc1[mask_slot], pc2[mask_slot],
                           c=SLOT_COLORS[slot], marker=SLOT_MARKERS[slot],
                           s=50, alpha=0.78, zorder=3,
                           edgecolors=color, linewidths=1.2)

        # Inizio/fine
        if len(pc1) > 0:
            ax.plot(pc1[0],  pc2[0],  'D', color=color, markersize=11, zorder=6,
                    markeredgecolor='black', markeredgewidth=0.8)
            ax.plot(pc1[-1], pc2[-1], '*', color=color, markersize=14, zorder=6,
                    markeredgecolor='black', markeredgewidth=0.8)

    # Legenda tesi
    tesi_legend = [
        Line2D([0], [0], color=TESI_COLORS.get(t, 'gray'), linewidth=2.5,
               label=f'{t}  (♦=inizio, ★=fine)')
        for t in [tesi_A, tesi_B]
    ]
    # Legenda fasce
    slot_legend = [
        Line2D([0], [0], marker=SLOT_MARKERS[s], color='w',
               markerfacecolor=SLOT_COLORS[s], markersize=11,
               label=f'{s}', markeredgewidth=0)
        for s in SLOTS_ORDERED
    ]

    leg1 = ax.legend(handles=tesi_legend, loc='upper left', title='Tesi', fontsize=10)
    ax.add_artist(leg1)
    ax.legend(handles=slot_legend, loc='lower right', title='Fascia oraria', fontsize=10)

    ax.set_xlabel(
        f'PC1 — {explained[0]*100:.1f}% varianza spiegata\n'
        '← profilo VOC A  |  profilo VOC B →  (o viceversa)',
        fontsize=10
    )
    ax.set_ylabel(
        f'PC2 — {explained[1]*100:.1f}% varianza spiegata\n'
        '(seconda struttura indipendente)',
        fontsize=10
    )
    ax.set_title(
        f'Traiettorie Temporali — {pair_name}\n'
        'Le due linee mostrano come evolve il profilo VOC nel tempo per ogni tesi\n'
        'Se si separano = profilo VOC diverso | Se si sovrappongono = profilo simile\n'
        'Bordo dei punti = colore tesi | Riempimento = fascia oraria',
        fontsize=10
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / f'{pair_name}_traiettorie.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_pair_by_tesi(
    time_joint: pd.DataFrame,
    explained: np.ndarray,
    output_dir: Path,
    pair_name: str
):
    """Score plot con convex hull: mostra la regione dello spazio VOC occupata da ogni tesi."""
    try:
        from scipy.spatial import ConvexHull
        has_hull = True
    except ImportError:
        has_hull = False

    fig, ax = plt.subplots(figsize=(11, 8))
    tesi_list = list(time_joint['tesi'].unique())

    for tesi_name in tesi_list:
        mask = time_joint['tesi'] == tesi_name
        pc1 = time_joint.loc[mask, 'PC1'].values
        pc2 = time_joint.loc[mask, 'PC2'].values
        color = TESI_COLORS.get(tesi_name, 'gray')

        ax.scatter(pc1, pc2, c=color, s=40, alpha=0.65, label=tesi_name,
                   edgecolors='white', linewidths=0.3, zorder=2)
        ax.plot(pc1, pc2, '-', color=color, alpha=0.2, linewidth=0.8, zorder=1)

        if has_hull and len(pc1) >= 3:
            try:
                pts = np.column_stack([pc1, pc2])
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                hull_pts = np.vstack([hull_pts, hull_pts[0]])
                ax.fill(hull_pts[:, 0], hull_pts[:, 1], alpha=0.07, color=color, zorder=0)
                ax.plot(hull_pts[:, 0], hull_pts[:, 1], '--', color=color, alpha=0.45, linewidth=1.2)
            except Exception:
                pass

        # Centroide
        ax.plot(np.mean(pc1), np.mean(pc2), 'D', color=color,
                markersize=12, zorder=5, markeredgecolor='black', markeredgewidth=1,
                label=f'{tesi_name} (centroide)')

    ax.set_xlabel(f'PC1 — {explained[0]*100:.1f}% varianza', fontsize=10)
    ax.set_ylabel(f'PC2 — {explained[1]*100:.1f}% varianza', fontsize=10)
    ax.set_title(
        f'Spazio VOC Occupato da Ogni Tesi — {pair_name}\n'
        'Area tratteggiata = convex hull (regione occupata) | ♦ = centroide medio\n'
        'Aree sovrapposte = VOC simili | Aree separate = VOC distinti',
        fontsize=10
    )
    ax.legend(title='Tesi', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / f'{pair_name}_spazio_VOC.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_pair_3d(
    time_joint: pd.DataFrame,
    explained: np.ndarray,
    output_dir: Path,
    pair_name: str
):
    """Traiettorie 3D (PC1, PC2, PC3) per le due tesi."""
    tesi_list = list(time_joint['tesi'].unique())

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    for tesi_name in tesi_list:
        mask = time_joint['tesi'] == tesi_name
        pc1 = time_joint.loc[mask, 'PC1'].values
        pc2 = time_joint.loc[mask, 'PC2'].values
        pc3 = time_joint.loc[mask, 'PC3'].values
        color = TESI_COLORS.get(tesi_name, 'gray')

        ax.plot(pc1, pc2, pc3, '-', color=color, alpha=0.38, linewidth=1.2, label=tesi_name)

        for slot in SLOTS_ORDERED:
            mask_slot = time_joint.loc[mask, 'time_slot'].values == slot
            if np.any(mask_slot):
                ax.scatter(pc1[mask_slot], pc2[mask_slot], pc3[mask_slot],
                           c=SLOT_COLORS[slot], marker=SLOT_MARKERS[slot],
                           s=30, alpha=0.65, zorder=2)

        ax.scatter([pc1[0]],  [pc2[0]],  [pc3[0]],  c=[color], marker='D', s=80, zorder=5)
        ax.scatter([pc1[-1]], [pc2[-1]], [pc3[-1]], c=[color], marker='*', s=100, zorder=5)

    slot_legend = [Line2D([0], [0], marker=SLOT_MARKERS[s], color='w',
                          markerfacecolor=SLOT_COLORS[s], markersize=9, label=s)
                   for s in SLOTS_ORDERED]

    ax.set_xlabel(f'PC1 ({explained[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({explained[1]*100:.1f}%)')
    ax.set_zlabel(f'PC3 ({explained[2]*100:.1f}%)')
    ax.set_title(f'Traiettorie 3D — {pair_name}\n(♦=inizio ★=fine)', fontsize=10)

    leg1 = ax.legend(loc='upper left', fontsize=9, title='Tesi')
    ax.add_artist(leg1)
    ax.legend(handles=slot_legend, loc='lower left', fontsize=8, title='Fascia')
    plt.tight_layout()
    fig.savefig(output_dir / f'{pair_name}_3d.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_pair_slot_separation(
    time_joint: pd.DataFrame,
    explained: np.ndarray,
    output_dir: Path,
    pair_name: str
):
    """
    3 pannelli affiancati (uno per slot orario): posizione delle due tesi + ellissi 1-sigma.

    LETTURA:
    - In ogni pannello: se le due nuvole (ellissi) si sovrappongono = tesi simili in quel momento
    - Se non si sovrappongono = tesi diverse in quel momento
    - Confronta i tre pannelli: in quale fascia sono piu' separate?
    - Il centroide (rombo) mostra la posizione media della tesi nella fascia
    """
    slots_present = [s for s in SLOTS_ORDERED if (time_joint['time_slot'] == s).any()]
    n_slots = len(slots_present)
    if n_slots == 0:
        return

    tesi_list = list(time_joint['tesi'].unique())
    fig, axes = plt.subplots(1, n_slots, figsize=(7 * n_slots, 7), sharey=True, sharex=True)
    if n_slots == 1:
        axes = [axes]

    for ax, slot in zip(axes, slots_present):
        t_slot = time_joint[time_joint['time_slot'] == slot]

        for tesi_name in tesi_list:
            t_sub = t_slot[t_slot['tesi'] == tesi_name]
            if len(t_sub) < 2:
                continue

            pc1 = t_sub['PC1'].values
            pc2 = t_sub['PC2'].values
            color = TESI_COLORS.get(tesi_name, 'gray')
            n_pts = len(pc1)

            ax.scatter(pc1, pc2, c=color, s=35, alpha=0.65,
                       edgecolors='white', linewidths=0.3, zorder=2,
                       label=f'{tesi_name} ({n_pts} misure)')
            cx, cy = np.mean(pc1), np.mean(pc2)
            ax.plot(cx, cy, 'D', color=color, markersize=12, zorder=5,
                    markeredgecolor='black', markeredgewidth=1.0)
            _draw_confidence_ellipse(pc1, pc2, ax, color=color, alpha=0.18)

        # Distanza tra i centroidi
        centroids = []
        for tesi_name in tesi_list:
            t_sub = t_slot[t_slot['tesi'] == tesi_name]
            if len(t_sub) >= 1:
                centroids.append(np.array([t_sub['PC1'].mean(), t_sub['PC2'].mean()]))
        if len(centroids) == 2:
            d = np.linalg.norm(centroids[0] - centroids[1])
            ax.text(0.97, 0.03, f'Distanza centroidi:\n{d:.2f} u.a.',
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        ax.set_xlabel(f'PC1 ({explained[0]*100:.1f}%)', fontsize=9)
        ax.set_title(
            f'{slot}\n'
            f'({slot}: 06-14 Mattina / 14-22 Pomeriggio / 22-06 Notte)',
            fontsize=9
        )
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(f'PC2 ({explained[1]*100:.1f}%)', fontsize=9)

    handles = [mpatches.Patch(color=TESI_COLORS.get(t, 'gray'), label=t) for t in tesi_list]
    axes[-1].legend(handles=handles, title='Tesi', fontsize=10)

    fig.suptitle(
        f'Separazione per Fascia Oraria — {pair_name}\n'
        'Ellissi = 1σ (dispersione) | ♦ = centroide | Distanza = quanto sono diverse le tesi\n'
        'Fascia con ellissi NON sovrapposte = momento in cui i trattamenti producono VOC diversi',
        fontsize=11, y=1.02
    )
    plt.tight_layout()
    fig.savefig(output_dir / f'{pair_name}_separazione_slot.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_centroid_distances(
    time_joint: pd.DataFrame,
    explained: np.ndarray,
    output_dir: Path,
    pair_name: str,
    tesi_A: str,
    tesi_B: str
):
    """
    Distanza euclidea tra i centroidi di tesi_A e tesi_B, per giorno e fascia oraria.

    LETTURA:
    - Asse X = giorno (0=primo giorno della run)
    - Asse Y = distanza (nello spazio PC): > = piu' diversi
    - Colori delle barre = fascia oraria (giallo=Mattina, arancione=Pom, blu=Notte)
    - Se le barre gialle/arancioni sono alte e le blu basse:
      le tesi sono diverse di giorno ma convergono di notte
      -> conferma che il trattamento agisce sul metabolismo fotosintetico
    - Se la distanza cresce di giorno in giorno: effetto cumulativo del trattamento
    """
    max_day = int(time_joint['day_num'].max()) + 1
    slots_present = [s for s in SLOTS_ORDERED if (time_joint['time_slot'] == s).any()]
    n_slots = len(slots_present)

    distances = np.full((max_day, n_slots), np.nan)

    for day_idx in range(max_day):
        for s_idx, slot in enumerate(slots_present):
            mask = ((time_joint['day_num'] == day_idx) &
                    (time_joint['time_slot'] == slot))
            t_sub = time_joint[mask]

            sub_A = t_sub[t_sub['tesi'] == tesi_A]
            sub_B = t_sub[t_sub['tesi'] == tesi_B]

            if len(sub_A) >= 1 and len(sub_B) >= 1:
                cA = np.array([sub_A['PC1'].mean(), sub_A['PC2'].mean()])
                cB = np.array([sub_B['PC1'].mean(), sub_B['PC2'].mean()])
                distances[day_idx, s_idx] = np.linalg.norm(cA - cB)

    fig, ax = plt.subplots(figsize=(max(10, max_day * 2.5), 5))
    x = np.arange(max_day)
    width = 0.25
    n = len(slots_present)
    offsets = np.linspace(-(n - 1) * width / 2, (n - 1) * width / 2, n)

    for s_idx, (slot, offset) in enumerate(zip(slots_present, offsets)):
        vals = distances[:, s_idx]
        valid = ~np.isnan(vals)
        if np.any(valid):
            ax.bar(x[valid] + offset, vals[valid], width=width,
                   color=SLOT_COLORS[slot], alpha=0.85, label=slot,
                   edgecolor='white', linewidth=0.6)

    ax.set_xlabel('Giorno (0 = primo giorno della registrazione)', fontsize=10)
    ax.set_ylabel(
        f'Distanza tra centroidi di {tesi_A} e {tesi_B}\n(nello spazio PC1-PC2)',
        fontsize=9
    )
    ax.set_title(
        f'Separazione {tesi_A} vs {tesi_B} nel Tempo — {pair_name}\n'
        f'Distanza alta = profilo VOC molto diverso tra {tesi_A} e {tesi_B} in quel momento\n'
        'Barre gialle alte + barre blu basse = separano di giorno, convergono di notte',
        fontsize=10
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f'Giorno {d}' for d in range(max_day)], fontsize=9)
    ax.legend(title='Fascia Oraria', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / f'{pair_name}_distanza_centroidi.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# PCA PER SLOT ORARIO
# =============================================================================

def pca_by_slot(
    tesi_data: Dict[str, pd.DataFrame],
    run_start_datetimes: Dict[str, Optional[datetime]],
    output_dir: Path,
    pair_name: str,
    top_loadings: int = 15,
    mass_to_compound: Optional[Dict[str, str]] = None,
    avg_block_durations: Optional[Dict[str, float]] = None
) -> Dict:
    """
    PCA separata per fascia oraria: identifica quali VOC caratterizzano ogni fascia.

    Per ogni slot (Mattina / Pomeriggio / Notte):
    - Prende solo i blocchi di quella fascia da tutte le tesi
    - Fa PCA su questa sottomatrice
    - I loadings di PC1 = VOC piu' importanti IN QUELLA FASCIA

    FISICAMENTE:
    - Mattina: VOC biogenici da attivita' fotosintetica in partenza (isoprene, MeSA...)
    - Pomeriggio: picco di emissioni termiche e fotosintetiche
    - Notte: VOC da stress, fermentazione, processi non-fotosintetici
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"  PCA per slot orario ({pair_name})")

    mass_sets = [set(_extract_median_matrix(df)[1]) for df in tesi_data.values()]
    common_masses = sorted(set.intersection(*mass_sets))

    results = {}
    loading_matrix = {}

    for slot in SLOTS_ORDERED:
        X_slot, slot_labels = [], []

        _abd = avg_block_durations or {}
        for tesi_name, df in tesi_data.items():
            X_raw, mass_names = _extract_median_matrix(df)
            idx_map = {m: i for i, m in enumerate(mass_names)}
            t_info = _build_time_info(df, run_start_datetimes.get(tesi_name), tesi_name,
                                       avg_block_duration_s=_abd.get(tesi_name))
            mask_slot = t_info['time_slot'] == slot

            if not np.any(mask_slot.values):
                continue

            X_common = np.full((len(df), len(common_masses)), np.nan)
            for j, m in enumerate(common_masses):
                if m in idx_map:
                    X_common[:, j] = X_raw[:, idx_map[m]]

            X_slot.append(X_common[mask_slot.values])
            slot_labels.extend([tesi_name] * int(np.sum(mask_slot.values)))

        if not X_slot:
            logger.warning(f"    Nessun dato per slot {slot} — skip")
            continue

        X_combined = np.vstack(X_slot)
        n_combined = len(X_combined)
        logger.info(f"    {slot}: {n_combined} blocchi da {len(tesi_data)} tesi")

        if n_combined < 3:
            logger.warning(f"    {slot}: troppo pochi blocchi, skip")
            continue

        X_scaled, valid_cols, _, _ = _impute_and_scale(X_combined)
        masses_valid = [m for m, v in zip(common_masses, valid_cols) if v]
        n_comp = min(5, n_combined - 1, len(masses_valid))

        pca = PCA(n_components=n_comp)
        scores  = pca.fit_transform(X_scaled)
        loadings = pca.components_
        explained = pca.explained_variance_ratio_

        loading_matrix[slot] = (loadings[0], masses_valid)

        results[slot] = {
            'pca': pca, 'scores': scores, 'loadings': loadings,
            'explained': explained, 'mass_names': masses_valid,
            'labels': slot_labels,
        }

    # PLOT A: Heatmap loadings PC1 per fascia
    _plot_slot_loadings_heatmap(loading_matrix, output_dir, pair_name,
                                top_n=top_loadings, mass_to_compound=mass_to_compound)

    # PLOT B: Score plot per slot
    _plot_slot_scores(results, output_dir, pair_name)

    return results


def _plot_slot_loadings_heatmap(
    loading_matrix: Dict[str, Tuple],
    output_dir: Path,
    pair_name: str,
    top_n: int = 15,
    mass_to_compound: Optional[Dict[str, str]] = None
):
    """
    Heatmap: righe = masse, colonne = Mattina/Pom/Notte, valori = loading PC1.

    LETTURA:
    - Rosso intenso = questa massa ha loading POSITIVO alto in quella fascia
      -> VOC abbondante / molto variabile in quella fascia
    - Blu intenso = loading negativo forte
    - Bianco = loading vicino a zero = non influente in quella fascia
    - Confronta riga per riga: se una massa e' rossa di mattina e bianca di notte
      -> e' un VOC fotosintetico tipico (emesso solo durante il giorno)
    """
    if not loading_matrix:
        return

    slots = [s for s in SLOTS_ORDERED if s in loading_matrix]
    if not slots:
        return

    combined_importance = {}
    for slot in slots:
        loads, masses = loading_matrix[slot]
        for m, l in zip(masses, loads):
            combined_importance[m] = combined_importance.get(m, 0) + abs(l)

    top_masses = sorted(combined_importance, key=lambda m: combined_importance[m],
                        reverse=True)[:top_n]

    heatmap_data = np.zeros((len(top_masses), len(slots)))
    for j, slot in enumerate(slots):
        loads, masses = loading_matrix[slot]
        mass_idx = {m: i for i, m in enumerate(masses)}
        for i, m in enumerate(top_masses):
            if m in mass_idx:
                heatmap_data[i, j] = loads[mass_idx[m]]

    ylabels = []
    for m in top_masses:
        if mass_to_compound and m in mass_to_compound:
            short = mass_to_compound[m].split('|')[0].strip()
            short = short[:22] + '...' if len(short) > 22 else short
            ylabels.append(f"{m} | {short}")
        else:
            ylabels.append(m)

    fig, ax = plt.subplots(figsize=(9, max(7, top_n * 0.45 + 3)))
    vmax = np.max(np.abs(heatmap_data)) or 1.0
    im = ax.imshow(heatmap_data, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Loading PC1\n(contributo alla struttura principale)', fontsize=9)

    ax.set_xticks(range(len(slots)))
    ax.set_xticklabels(slots, fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8)

    for i in range(len(top_masses)):
        for j in range(len(slots)):
            v = heatmap_data[i, j]
            c = 'white' if abs(v) > vmax * 0.55 else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7, color=c)

    ax.set_title(
        f'Loadings PC1 per Fascia Oraria — {pair_name}\n'
        f'Top {top_n} VOC per importanza cumulativa\n'
        'Rosso = VOC importante in quella fascia | Blu = inversamente correlato | Bianco = non influente\n'
        'VOC rosso solo di mattina = biogenico/fotosintetico | VOC rosso di notte = non-fotosintetico',
        fontsize=10
    )
    plt.tight_layout()
    fig.savefig(output_dir / f'{pair_name}_slot_loadings_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_slot_scores(results: Dict, output_dir: Path, pair_name: str):
    """Score plot per slot orario, colori per tesi."""
    slots = [s for s in SLOTS_ORDERED if s in results]
    if not slots:
        return

    fig, axes = plt.subplots(1, len(slots), figsize=(7 * len(slots), 7))
    if len(slots) == 1:
        axes = [axes]

    for ax, slot in zip(axes, slots):
        res      = results[slot]
        scores   = res['scores']
        labels   = res['labels']
        explained = res['explained']

        tesi_unique = list(dict.fromkeys(labels))
        offset = 0
        for tesi_name in tesi_unique:
            n_this = labels.count(tesi_name)
            sc = scores[offset:offset + n_this]
            ax.scatter(sc[:, 0], sc[:, 1],
                       c=TESI_COLORS.get(tesi_name, 'gray'),
                       s=45, alpha=0.75, label=f'{tesi_name} ({n_this})',
                       edgecolors='white', linewidths=0.3)
            offset += n_this

        ax.set_xlabel(f'PC1 ({explained[0]*100:.1f}%)', fontsize=9)
        ax.set_ylabel(f'PC2 ({explained[1]*100:.1f}%)', fontsize=9)
        ax.set_title(
            f'{slot}\n'
            'Ogni punto = un blocco da ~5 min\n'
            'Punti separati per colore = VOC diversi in quella fascia',
            fontsize=9
        )
        ax.legend(title='Tesi', fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'Score Plot per Fascia Oraria — {pair_name}\n'
        'Se i colori si separano = le due tesi hanno profilo VOC diverso in quella fascia',
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    fig.savefig(output_dir / f'{pair_name}_slot_scores.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# ENTRY POINT: LANCIA TUTTE LE ANALISI PCA
# =============================================================================

def run_all_pca(
    List_final: List[pd.DataFrame],
    tesi_names: List[str],
    output_base_dir: Path,
    run_start_datetimes: Optional[Dict[str, datetime]] = None,
    mass_to_compound: Optional[Dict[str, str]] = None,
    tesi_for_joint: Optional[List[str]] = None,
    avg_block_durations: Optional[Dict[str, float]] = None
):
    """
    Lancia tutte le analisi PCA e salva i risultati in output/06_pca_analysis/.

    Struttura output:
    - singola_per_tesi/{H0,H1,H2,H3}/  <- PCA per singola tesi
    - congiunta_H0_vs_H1/               <- PCA congiunta coppia 1
    - congiunta_H2_vs_H3/               <- PCA congiunta coppia 2
    - per_slot_orario_H0vsH1/           <- slot orario per coppia 1
    - per_slot_orario_H2vsH3/           <- slot orario per coppia 2

    Args:
        List_final: lista DataFrame final (blank-subtracted), indici come tesi_names
        tesi_names: lista nomi, es. ['H0_pre','H0','H1_pre','H1','H2','H3']
        output_base_dir: directory di base (es. output/)
        run_start_datetimes: dict {tesi_name: datetime_reale_inizio_FASE}
            IMPORTANTE: per H0/H1 deve essere il datetime post-acclimatizzazione
            (run_start + acclim_hours + exclusion_hours), non il datetime grezzo di run.
            Per H2/H3 deve essere run_start + split_hours + exclusion_hours.
        mass_to_compound: mapping massa -> nome composto
        tesi_for_joint: NON USATO (mantenuto per compatibilita' - le coppie sono H0vsH1 e H2vsH3)
        avg_block_durations: dict {tesi_name: secondi_per_blocco_aggregato}
            IMPORTANTE: i blocchi aggregati durano ~1 ora (3600s), NON 300s.
            Calcola come: durata_fase_ore * 3600 / n_blocchi
            Se None, usa 3600s come fallback (molto meglio di 300s).
    """
    if run_start_datetimes is None:
        run_start_datetimes = {}
    if avg_block_durations is None:
        avg_block_durations = {}

    logger.info("=" * 70)
    logger.info("AVVIO ANALISI PCA")
    logger.info(f"  Tesi disponibili: {tesi_names}")
    if avg_block_durations:
        for t, d in avg_block_durations.items():
            logger.info(f"  avg_block_duration[{t}] = {d:.0f}s (~{d/3600:.2f}h per blocco)")
    else:
        logger.info("  avg_block_durations non fornito -> uso 3600s (1h) per blocco come fallback")
    logger.info("=" * 70)

    tesi_map = {name: df for name, df in zip(tesi_names, List_final)
                if df is not None and not df.empty}

    pca_dir = output_base_dir / "06_pca_analysis"
    pca_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # A) PCA SINGOLA PER TESI (H0, H1, H2, H3)
    # ==========================================================================
    single_dir = pca_dir / "singola_per_tesi"
    single_dir.mkdir(parents=True, exist_ok=True)
    logger.info("\n--- A) PCA Singola per Tesi ---")

    for tesi_name in ['H0', 'H1', 'H2', 'H3']:
        if tesi_name not in tesi_map:
            logger.warning(f"  {tesi_name}: non disponibile, skip")
            continue
        pca_single_tesi(
            df=tesi_map[tesi_name],
            tesi_name=tesi_name,
            run_start_datetime=run_start_datetimes.get(tesi_name),
            output_dir=single_dir / tesi_name,
            n_components=10,
            top_loadings=20,
            mass_to_compound=mass_to_compound,
            avg_block_duration_s=avg_block_durations.get(tesi_name)
        )

    # ==========================================================================
    # B) PCA CONGIUNTA: H0 vs H1 (prima fase, stesso periodo)
    # ==========================================================================
    logger.info("\n--- B) PCA Congiunta H0 vs H1 ---")
    if 'H0' in tesi_map and 'H1' in tesi_map:
        pca_joint_pair(
            tesi_A_name='H0', tesi_A_df=tesi_map['H0'],
            tesi_B_name='H1', tesi_B_df=tesi_map['H1'],
            run_start_datetimes=run_start_datetimes,
            output_dir=pca_dir / "congiunta_H0_vs_H1",
            n_components=3,
            top_loadings=20,
            mass_to_compound=mass_to_compound,
            avg_block_durations=avg_block_durations
        )
    else:
        logger.warning("  H0 o H1 non disponibili, skip")

    # ==========================================================================
    # C) PCA CONGIUNTA: H2 vs H3 (seconda fase)
    # ==========================================================================
    logger.info("\n--- C) PCA Congiunta H2 vs H3 ---")
    if 'H2' in tesi_map and 'H3' in tesi_map:
        pca_joint_pair(
            tesi_A_name='H2', tesi_A_df=tesi_map['H2'],
            tesi_B_name='H3', tesi_B_df=tesi_map['H3'],
            run_start_datetimes=run_start_datetimes,
            output_dir=pca_dir / "congiunta_H2_vs_H3",
            n_components=3,
            top_loadings=20,
            mass_to_compound=mass_to_compound,
            avg_block_durations=avg_block_durations
        )
    else:
        logger.warning("  H2 o H3 non disponibili, skip")

    # ==========================================================================
    # D) PCA PER SLOT ORARIO: H0 vs H1
    # ==========================================================================
    logger.info("\n--- D) PCA per Slot Orario — H0 vs H1 ---")
    pair_H0H1 = {k: tesi_map[k] for k in ['H0', 'H1'] if k in tesi_map}
    if len(pair_H0H1) == 2:
        pca_by_slot(
            tesi_data=pair_H0H1,
            run_start_datetimes=run_start_datetimes,
            output_dir=pca_dir / "per_slot_orario_H0vsH1",
            pair_name="H0vsH1",
            top_loadings=15,
            mass_to_compound=mass_to_compound,
            avg_block_durations=avg_block_durations
        )

    # ==========================================================================
    # E) PCA PER SLOT ORARIO: H2 vs H3
    # ==========================================================================
    logger.info("\n--- E) PCA per Slot Orario — H2 vs H3 ---")
    pair_H2H3 = {k: tesi_map[k] for k in ['H2', 'H3'] if k in tesi_map}
    if len(pair_H2H3) == 2:
        pca_by_slot(
            tesi_data=pair_H2H3,
            run_start_datetimes=run_start_datetimes,
            output_dir=pca_dir / "per_slot_orario_H2vsH3",
            pair_name="H2vsH3",
            top_loadings=15,
            mass_to_compound=mass_to_compound,
            avg_block_durations=avg_block_durations
        )

    logger.info("\n" + "=" * 70)
    logger.info(f"PCA COMPLETATA — Output in: {pca_dir}")
    logger.info("  singola_per_tesi/{H0,H1,H2,H3}/")
    logger.info("  congiunta_H0_vs_H1/  congiunta_H2_vs_H3/")
    logger.info("  per_slot_orario_H0vsH1/  per_slot_orario_H2vsH3/")
    logger.info("=" * 70)
