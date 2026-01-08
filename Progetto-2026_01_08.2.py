# =============================================================================
# ASSIGNMENT 2 - GESTIONE ENERGETICA
# PIPELINE COMPLETA: Punti 1, 2 e 3
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import string
from matplotlib.colors import ListedColormap
import matplotlib.dates as mdates # Importante per la formattazione delle date

# Configurazione stile grafico
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)

# -----------------------------------------------------------------------------
# 0. SETUP E CARICAMENTO DATI
# -----------------------------------------------------------------------------
print(">>> 0. SETUP: Caricamento dati...")
try:
    df_tot_3 = pd.read_csv('data/df_tot_3.csv', parse_dates=['date_time'])
except FileNotFoundError:
    df_tot_3 = pd.read_csv('df_tot_3.csv', parse_dates=['date_time'])

df_tot_3.set_index('date_time', inplace=True)
df_tot_3.sort_index(inplace=True)

# --- FONDAMENTALE: Salviamo una copia grezza per il confronto finale ---
# Questa copia contiene ancora gli errori (outlier, zeri, missing)
df_original = df_tot_3.copy()

# Colonne ausiliarie (saranno usate in tutti i punti)
df_tot_3['date'] = df_tot_3.index.date
df_tot_3['time'] = df_tot_3.index.time
df_tot_3['month'] = df_tot_3.index.month
df_tot_3['is_weekend'] = df_tot_3.index.dayofweek >= 5

print("Dataset caricato e copia di backup salvata.")

# =============================================================================
# PUNTO 1: PREPARAZIONE E PULIZIA DEL DATASET
# =============================================================================
print("\n" + "=" * 80)
print("AVVIO PUNTO 1: Pipeline di Pulizia")
print("=" * 80)

# ... (Il resto del codice del Punto 1 rimane uguale) ...
# ... (Inserisco qui solo le parti modificate o rilevanti per il contesto) ...

# -----------------------------------------------------------------------------
# STEP A: VISUALIZZAZIONE GREZZA (Raw Data)
# -----------------------------------------------------------------------------
print("\n>>> CARPET PLOT 1: Dati Grezzi...")

pivot_raw = df_tot_3.pivot_table(index='date', columns='time', values='power_C')

plt.figure(figsize=(15, 10))
# Heatmap Grezza: Mostra Outlier (colori estremi), Zeri (Viola) e Missing (Bianchi)
sns.heatmap(pivot_raw, cmap='Spectral_r', cbar_kws={'label': 'Power [kW]'},
            xticklabels=8, yticklabels=30)
plt.title('1. Carpet Plot GREZZO: Outlier, Zeri e Missing Value visibili')
plt.xlabel('Ora del Giorno')
plt.ylabel('Data')
plt.show()

# -----------------------------------------------------------------------------
# STEP B: IDENTIFICAZIONE OUTLIER (BOX PLOT ZOOMATO + COMPLETO)
# -----------------------------------------------------------------------------
print("\n>>> Analisi Outlier (Metodo IQR)...")

# 1. Calcolo Statistico Soglie
Q1 = df_tot_3['power_C'].quantile(0.25)
Q3 = df_tot_3['power_C'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"   Soglie Calcolate -> Inf: {lower_bound:.2f}, Sup: {upper_bound:.2f}")

# --- BOX PLOT A: ZOOMATO (Focus su Quartili) ---
plt.figure(figsize=(12, 6))
sns.boxplot(x=df_tot_3['power_C'], color='lightblue', showfliers=False)
plt.axvline(x=lower_bound, color='red', linestyle='--', linewidth=2, label='Soglia Inf')
plt.axvline(x=upper_bound, color='red', linestyle='--', linewidth=2, label='Soglia Sup')
plt.title('Box Plot A: ZOOMATO (Focus su Quartili e Soglie, Outlier nascosti)')
plt.legend()
plt.show()

# --- BOX PLOT B: COMPLETO (Stripplot Colorato) ---
# Creiamo colonna booleana temporanea per colorare i punti
df_tot_3['is_outlier'] = (df_tot_3['power_C'] < lower_bound) | (df_tot_3['power_C'] > upper_bound)
n_outliers = df_tot_3['is_outlier'].sum()
print(f"   Visualizzazione di {n_outliers} outlier in rosso...")

plt.figure(figsize=(15, 8))
# Boxplot base (bianco, senza outlier standard)
sns.boxplot(x=df_tot_3['power_C'], color='white', showfliers=False, whis=1.5)
# Sovrapposizione punti (Blu=Normali, Rosso=Outlier)
sns.stripplot(
    x=df_tot_3['power_C'],
    hue=df_tot_3['is_outlier'],
    palette={False: 'tab:blue', True: 'red'},
    dodge=False,
    alpha=0.6,
    size=3
)
plt.title('Box Plot B: COMPLETO (Punti Rossi = Outlier, Punti Blu = Dati Normali)')
plt.legend(title='È Outlier?', loc='upper right')
plt.xlabel('Power [kW]')
plt.show()

# Pulizia colonna temporanea
df_tot_3.drop(columns=['is_outlier'], inplace=True)

# -----------------------------------------------------------------------------
# STEP C: RIMOZIONE OUTLIER (Outlier -> NaN)
# -----------------------------------------------------------------------------
print("\n>>> Rimozione Outlier (Conversione in NaN)...")

outlier_mask = (df_tot_3['power_C'] < lower_bound) | (df_tot_3['power_C'] > upper_bound)
df_tot_3.loc[outlier_mask, 'power_C'] = np.nan
print(f"   Rimossi {outlier_mask.sum()} outlier statistici.")

print("\n>>> CARPET PLOT 2: Senza Outlier (Solo Missing Values)...")
pivot_no_outlier = df_tot_3.pivot_table(index='date', columns='time', values='power_C')

plt.figure(figsize=(15, 10))
# I buchi bianchi ora rappresentano i dati mancanti (originali + outlier rimossi)
sns.heatmap(pivot_no_outlier, cmap='Spectral_r', cbar_kws={'label': 'Power [kW]'},
            xticklabels=8, yticklabels=30)
plt.title('2. Carpet Plot REALE: Outlier rimossi (ora sono buchi bianchi da riempire)')
plt.xlabel('Ora del Giorno')
plt.ylabel('Data')
plt.show()

# -----------------------------------------------------------------------------
# STEP D: INTERPOLAZIONE LINEARE (Riempimento Buchi)
# -----------------------------------------------------------------------------
print("\n>>> Interpolazione Lineare (Rimozione Missing Values)...")

# Riempie i NaN (buchi bianchi) interpolando temporalmente
df_tot_3['power_C'] = df_tot_3['power_C'].interpolate(method='linear')

print("\n>>> CARPET PLOT 3: Interpolato (Con Zeri)...")
pivot_interp = df_tot_3.pivot_table(index='date', columns='time', values='power_C')

plt.figure(figsize=(15, 10))
# Non ci sono più buchi bianchi. Rimangono gli Zeri (linee viola).
sns.heatmap(pivot_interp, cmap='Spectral_r', cbar_kws={'label': 'Power [kW]'},
            xticklabels=8, yticklabels=30)
plt.title('3. Carpet Plot INTERPOLATO: Missing eliminati, ma rimangono gli Zeri (Viola)')
plt.xlabel('Ora del Giorno')
plt.ylabel('Data')
plt.show()

# -----------------------------------------------------------------------------
# STEP E: GESTIONE INTELLIGENTE DEGLI ZERI (Smart Imputation)
# -----------------------------------------------------------------------------
print("\n>>> Correzione Intelligente degli Zeri...")

# 1. Identifica Zeri
zeros_mask = df_tot_3['power_C'] <= 0
print(f"   Trovati {zeros_mask.sum()} valori a zero da correggere.")

# 2. Imposta a NaN per poterli riempire
df_tot_3.loc[zeros_mask, 'power_C'] = np.nan

# 3. Metodo Intelligente: Media di periodi simili (Mese, Weekend, Ora)
reference_profile = df_tot_3.groupby(['month', 'is_weekend', 'time'])['power_C'].transform('mean')
df_tot_3.loc[zeros_mask, 'power_C'] = reference_profile.loc[zeros_mask]

# 4. Interpolazione finale di sicurezza (per eventuali buchi residui)
if df_tot_3['power_C'].isna().sum() > 0:
    df_tot_3['power_C'] = df_tot_3['power_C'].interpolate(method='linear')

# =============================================================================
# PUNTO 2: VISUALIZZAZIONE DESCRITTIVA FINALE (DATASET PULITO)
# =============================================================================
print("\n" + "=" * 80)
print("AVVIO PUNTO 2: Report e Statistiche Finali")
print("=" * 80)

# -----------------------------------------------------------------------------
# STEP A: CARPET PLOT 4: FINALE
# -----------------------------------------------------------------------------

print("\n>>> 1. Carpet Plot Finale Pulito...")
pivot_final = df_tot_3.pivot_table(index='date', columns='time', values='power_C')

plt.figure(figsize=(15, 10))
sns.heatmap(pivot_final, cmap='Spectral_r', cbar_kws={'label': 'Power [kW]'},
            xticklabels=8, yticklabels=30)
plt.title('4. Carpet Plot FINALE: Pulito (Senza Missing, Senza Outlier, Senza Zeri)')
plt.xlabel('Ora del Giorno')
plt.ylabel('Data')
plt.show()

# -----------------------------------------------------------------------------
# STEP B: ULTERIORI CONFRONTI
# -----------------------------------------------------------------------------

# 1. Confronto line plot (prima vs dopo) - VERSIONE CORRETTA (ZOOM SULL'ASSE Y)
print("\n>>> 2. Confronto Serie Temporale (Zoom su ~5 giorni)...")
# Zoom su 500 punti (circa 5 giorni a 15 min) per vedere i dettagli della ricostruzione
N_SAMPLES = 480
start_idx = 1000  # Punto arbitrario con dati interessanti

# Campioniamo i dati
sample_orig = df_original.iloc[start_idx: start_idx + N_SAMPLES]
sample_clean = df_tot_3.iloc[start_idx: start_idx + N_SAMPLES]

plt.figure(figsize=(15, 6))

# Plottiamo l'originale (in grigio)
plt.plot(sample_orig.index, sample_orig['power_C'], label='Originale (Grezzo)', color='lightgray', alpha=0.8)

# Plottiamo il pulito (in blu)
plt.plot(sample_clean.index, sample_clean['power_C'], label='Pulito (Ricostruito)', color='blue', linewidth=1.5,
         linestyle='--')

# --- MODIFICA FONDAMENTALE PER VISIBILITA' ---
# Impostiamo i limiti dell'asse Y basandoci SOLO sui dati PULITI.
# Così l'outlier negativo (-4000) del dataset originale viene "tagliato fuori" visivamente
max_val_clean = sample_clean['power_C'].max()
plt.ylim(bottom=0, top=max_val_clean * 1.2)  # Lasciamo un 20% di margine sopra
# --------------------------------------------

plt.title(f'Confronto Dettagliato: Originale vs Ricostruito (Zoom su {N_SAMPLES} campioni)')
plt.ylabel('Power [kW]')
plt.legend()
plt.grid(True)
plt.show()

# 2. Trend giornaliero
print("\n>>> 3. Trend della Potenza Media Giornaliera...")
daily_trend = df_tot_3.groupby('date')['power_C'].mean()

plt.figure(figsize=(15, 6))
daily_trend.plot(color='green', linewidth=1.5)
plt.title('Andamento della Potenza Media Giornaliera (Stagionalità)')
plt.ylabel('Potenza Media [kW]')
plt.grid(True)
plt.show()

# 3. Istogramma distribuzione
print("\n>>> 4. Istogramma della Distribuzione (Dati Puliti)...")
plt.figure(figsize=(10, 6))
sns.histplot(df_tot_3['power_C'], bins=50, kde=True, color='teal')
plt.title('Distribuzione di Frequenza della Potenza Elettrica (Post-Pulizia)')
plt.xlabel('Power [kW]')
plt.show()

print("\n=== ELABORAZIONE PUNTI 1 e 2 COMPLETATA CON SUCCESSO ===")


# =============================================================================
# PUNTO 3: RIDUZIONE E TRASFORMAZIONE SAX (Z-SCORE LOCALE)
# Obiettivo: Testare diverse finestre (3, 4, 6h) e diversi simboli (4, 5, 6).
# Metodo: Normalizzazione Locale (ogni giorno normalizzato su se stesso).
# CORREZIONE: Rimossi Carpet Plot, Aggiunti Istogrammi Logaritmici.
# =============================================================================
print("\n" + "=" * 80)
print("AVVIO PUNTO 3: Applicazione SAX (Shape-based / Z-Score Locale)")
print("=" * 80)

from scipy.stats import norm
import string
import warnings

# Disabilita i warning di pandas per l'estetica dell'output
warnings.simplefilter(action='ignore', category=FutureWarning)


# -----------------------------------------------------------------------------
# 3.1 FUNZIONI SAX CUSTOM (NORMALIZZAZIONE LOCALE)
# -----------------------------------------------------------------------------

def get_breakpoints(n_symbols):
    """Calcola i punti di taglio (quantile) per una distribuzione Normale Standard."""
    return norm.ppf(np.linspace(1 / n_symbols, 1 - 1 / n_symbols, n_symbols - 1))


def apply_sax_to_day(row_values, window_hours, n_symbols):
    """
    Input: Array di valori di un giorno (96 punti).
    Output: Parola SAX, Valori PAA, Array dei Simboli
    """
    # 1. Z-Normalization LOCALE
    if np.std(row_values) == 0:
        n_seg = int(24 // window_hours)
        return "X" * n_seg, [], []

    row_norm = (row_values - np.mean(row_values)) / np.std(row_values)

    # 2. PAA
    points_per_window = int(window_hours * 4)
    n_segments = len(row_values) // points_per_window

    paa_values = []
    for i in range(n_segments):
        start = i * points_per_window
        end = start + points_per_window
        paa_values.append(np.mean(row_norm[start:end]))

    # 3. Discretizzazione
    breakpoints = get_breakpoints(n_symbols)
    letters = string.ascii_uppercase[:n_symbols]

    sax_symbols = []
    for val in paa_values:
        idx = 0
        for bp in breakpoints:
            if val < bp:
                break
            idx += 1
        sax_symbols.append(letters[idx])

    return "".join(sax_symbols), paa_values, sax_symbols


# -----------------------------------------------------------------------------
# 3.2 PREPARAZIONE DATI E PARAMETRI
# -----------------------------------------------------------------------------
daily_matrix = df_tot_3.pivot_table(index='date', columns='time', values='power_C')
daily_matrix.dropna(inplace=True)
print(f">>> Matrice pronta. Giorni analizzati: {len(daily_matrix)}")

# --- PARAMETRI ASSIGNMENT ---
window_options = [3, 4, 6]
alphabet_options = [4, 5, 6]

# -----------------------------------------------------------------------------
# 3.3 ILLUSTRAZIONE DEL METODO (PAA EFFECT)
# -----------------------------------------------------------------------------
print("\n>>> Generazione grafici illustrativi PAA (Separati per Finestra)...")

# Selezione Giorno Tipico (Mezza stagione, Feriale)
idx_dt = pd.to_datetime(daily_matrix.index)
candidates = daily_matrix.index[
    (idx_dt.month.isin([4, 5, 10])) & (idx_dt.dayofweek.isin([1, 2, 3]))
    ]
if len(candidates) > 0:
    sample_date = candidates[min(2, len(candidates) - 1)]
else:
    sample_date = daily_matrix.index[len(daily_matrix) // 2]

# Dati del giorno selezionato
sample_data = daily_matrix.loc[sample_date].values
# Normalizzazione Locale per il grafico di sfondo
row_norm = (sample_data - np.mean(sample_data)) / np.std(sample_data)

# Asse X e colori
x_axis = np.arange(96)
colors = ['#e41a1c', '#377eb8', '#4daf4a']  # Rosso, Blu, Verde

# --- LOOP PER GENERARE I 3 GRAFICI SEPARATI ---
for i, w in enumerate(window_options):
    plt.figure(figsize=(15, 6))  # Altezza leggermente ridotta per grafici singoli

    # A. Profilo Reale di Sfondo (Grigio)
    plt.plot(x_axis, row_norm, label='Profilo Reale (Z-Norm Locale)', color='silver', linewidth=4, alpha=0.6)

    # B. Calcolo Approssimazione PAA per la finestra corrente 'w'
    # (Usiamo 5 simboli come esempio intermedio per il calcolo, ma qui plottiamo i valori PAA numerici)
    _, paa_vals, _ = apply_sax_to_day(sample_data, w, 5)

    # Creiamo i gradini per il plot
    points_per_window = int(w * 4)
    steps_y = np.repeat(paa_vals, points_per_window)

    # Fix lunghezza array per il plot
    steps_y = steps_y[:96]
    if len(steps_y) < 96:
        steps_y = np.pad(steps_y, (0, 96 - len(steps_y)), 'edge')

    n_seg = 24 // w
    # Plot della linea PAA a gradini
    plt.plot(x_axis, steps_y, label=f'Approssimazione PAA (Finestra {w}h - {n_seg} segmenti)',
             color=colors[i], linewidth=3)

    # Cosmetica del grafico
    plt.title(f'Effetto PAA: Finestra di {w} Ore sul Giorno {sample_date}')
    plt.xlabel('Tempo (quarti d\'ora)')
    plt.ylabel('Potenza Normalizzata (Z-Score)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

print(">>> Grafici illustrativi PAA generati.")

# -----------------------------------------------------------------------------
# 3.4 ANALISI MASSIVA (LOOP SU TUTTE LE COMBINAZIONI)
# -----------------------------------------------------------------------------
results_summary = []

for w in window_options:
    for a in alphabet_options:
        word_len = 24 // w
        config_name = f"Win_{w}h_Sym_{a}"

        print(f"\n--- ELABORAZIONE: Finestra {w}h (Parola len={word_len}) | Simboli {a} ---")

        # 1. Calcolo SAX
        sax_output = daily_matrix.apply(
            lambda x: apply_sax_to_day(x.values, w, a), axis=1
        )

        words_series = sax_output.apply(lambda x: x[0])

        # 2. Statistiche
        word_counts = words_series.value_counts()
        n_unique = len(word_counts)
        top_word = word_counts.index[0]
        top_share = (word_counts.iloc[0] / len(words_series)) * 100

        results_summary.append({
            'Configurazione': config_name,
            'Finestra (h)': w,
            'Simboli': a,
            'Lunghezza Parola': word_len,
            'Pattern Unici': n_unique,
            'Top Pattern': top_word,
            'Top %': f"{top_share:.1f}%"
        })

        print(f"   Pattern unici trovati: {n_unique}")

        # ---------------------------------------------------------
        # B. ISTOGRAMMI FREQUENZA AFFIANCATI (Lineare vs Logaritmico)
        # ---------------------------------------------------------
        # Mostriamo Top 20 parole
        top_n = 20
        data_to_plot = word_counts.head(top_n)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Grafico 1: Scala Lineare
        data_to_plot.plot(kind='bar', ax=axes[0], color='teal', edgecolor='black', alpha=0.8)
        axes[0].set_title(f'Frequenza Lineare (Top {top_n}) - Win={w}h, Sym={a}')
        axes[0].set_xlabel('Parola SAX')
        axes[0].set_ylabel('Numero di Giorni')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)

        # Grafico 2: Scala Logaritmica (Come richiesto)
        data_to_plot.plot(kind='bar', ax=axes[1], color='firebrick', edgecolor='black', alpha=0.8)
        axes[1].set_yscale('log')  # <-- IL TRUCCO PER LA SCALA LOG
        axes[1].set_title(f'Frequenza Logaritmica (Top {top_n}) - Win={w}h, Sym={a}')
        axes[1].set_xlabel('Parola SAX')
        axes[1].set_ylabel('Numero di Giorni (Scala Log)')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', which='both', alpha=0.3)

        plt.tight_layout()
        plt.show()  #
# -----------------------------------------------------------------------------
# 3.5 RIASSUNTO FINALE
# -----------------------------------------------------------------------------
print("\n>>> RIASSUNTO CONFIGURAZIONI")
df_summary = pd.DataFrame(results_summary)
print(df_summary)

plt.figure(figsize=(10, 6))
sns.barplot(data=df_summary, x='Finestra (h)', y='Pattern Unici', hue='Simboli', palette='viridis')
plt.title('Discussione Risultati: Effetto dei Parametri sulla Varietà dei Pattern')
plt.ylabel('Numero di Pattern Unici')
plt.xlabel('Larghezza Finestra (ore)')
plt.show()

print("\n>>> Punto 3 completato.")

# =============================================================================
# PUNTO 4: ANALISI FREQUENZE E IDENTIFICAZIONE MOTIF/DISCORD
# Configurazione: 6h, 4 Simboli. Output: Grafici Dettagliati per Ogni Motif.
# =============================================================================
print("\n" + "=" * 80)
print("AVVIO PUNTO 4: Analisi Frequenze e Visualizzazione Dettagliata")
print("=" * 80)

# 1. Configurazione "Robusta" scelta
BEST_WIN = 6
BEST_SYM = 4

print(f"Configurazione scelta: Finestra {BEST_WIN}h, Simboli {BEST_SYM}")

# 2. Calcolo SAX
sax_series = daily_matrix.apply(
    lambda x: apply_sax_to_day(x.values, BEST_WIN, BEST_SYM)[0], axis=1
)

df_sax_analysis = daily_matrix.copy()
df_sax_analysis['word'] = sax_series

# 3. Analisi delle Frequenze
word_counts = df_sax_analysis['word'].value_counts()
max_count = word_counts.iloc[0]

# 4. Definizione Soglia (15% del max)
# -------------------------------------------------------------------------
# MODIFICA RICHIESTA: Soglia cambiata da 0.2 (20%) a 0.15 (15%)
# -------------------------------------------------------------------------
threshold = int(0.15 * max_count)

print(f"\n--- STATISTICHE ---")
print(f"Parola più frequente: '{word_counts.index[0]}' ({max_count} occorrenze)")
print(f"Soglia calcolata (0.15 * Max): {threshold}")

# 5. Classificazione
df_word_freq = word_counts.reset_index()
df_word_freq.columns = ['word', 'count']
df_word_freq['pattern_type'] = df_word_freq['count'].apply(
    lambda c: 'Motif' if c >= threshold else 'Discord'
)

# Merge
df_classified = pd.merge(df_sax_analysis.reset_index(), df_word_freq, on='word', how='left')
df_classified.set_index('date', inplace=True)

# ---------------------------------------------------------
# GRAFICO A: ISTOGRAMMA FREQUENZE (Panoramica)
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))
top_15_words = df_word_freq.head(15)
colors = ['green' if x == 'Motif' else 'red' for x in top_15_words['pattern_type']]

sns.barplot(x='word', y='count', data=top_15_words, palette=colors, edgecolor='black')
plt.axhline(y=threshold, color='blue', linestyle='--', linewidth=2, label=f'Soglia ({threshold})')
plt.title(f"Distribuzione Frequenze (Top 15 Pattern) - Win={BEST_WIN}h, Sym={BEST_SYM}")
plt.ylabel('Giorni')
plt.legend()
plt.show()

# ---------------------------------------------------------
# GRAFICO B: DETTAGLIO MOTIF (Uno per ogni pattern frequente)
# ---------------------------------------------------------
print("\n>>> Generazione grafici individuali per i MOTIF...")

# Prepariamo asse X
n_points = 96
x_axis = np.arange(n_points)
tick_indices = np.arange(0, n_points, 16)
tick_labels = [f"{h:02d}:00" for h in range(0, 24, 4)]

# Troviamo tutti i pattern Motif
motif_words_list = df_word_freq[df_word_freq['pattern_type'] == 'Motif']['word'].tolist()

# Creiamo una griglia di subplot dinamica
n_motifs = len(motif_words_list)
cols = 2
rows = (n_motifs + 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
axes = axes.flatten()  # Appiattiamo per iterare facilmente

for i, word in enumerate(motif_words_list):
    ax = axes[i]

    # Dati del Motif
    days_idx = df_classified[df_classified['word'] == word].index
    profiles = daily_matrix.loc[days_idx]
    mean_profile = profiles.mean()

    # Plot Sfondo (tutti i giorni)
    ax.plot(x_axis, profiles.values.T, color='gray', alpha=0.05)
    # Plot Media (Centroide)
    ax.plot(x_axis, mean_profile.values, color='green', linewidth=3, label='Centroide')

    ax.set_title(f"MOTIF: '{word}' (n={len(profiles)} giorni)")
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel('Potenza [kW]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

# Nascondiamo eventuali subplot vuoti
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# GRAFICO C: DETTAGLIO DISCORD (Top 4 più rari)
# ---------------------------------------------------------
print("\n>>> Generazione grafici individuali per i DISCORD...")

discord_words_list = df_word_freq[df_word_freq['pattern_type'] == 'Discord']['word'].tail(4).tolist()

if len(discord_words_list) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, word in enumerate(discord_words_list):
        if i >= 4: break  # Max 4 grafici
        ax = axes[i]

        # Dati del Discord (prendiamo il primo giorno trovato)
        day_idx = df_classified[df_classified['word'] == word].index[0]
        profile = daily_matrix.loc[day_idx]

        ax.plot(x_axis, profile.values, color='red', linewidth=2, label=f'Discord: {word}')
        ax.set_title(f"DISCORD: '{word}' (Giorno: {day_idx})")
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels)
        ax.set_ylabel('Potenza [kW]')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()

print(">>> Punto 4 completato.")

# =============================================================================
# PUNTO 5: VISUALIZZAZIONE RAGGRUPPAMENTI PROFILI (FACET GRID)
# Obiettivo: Visualizzare i profili di carico raggruppati per parola SAX,
# distinguendo Motif e Discord e mostrando il profilo medio (centroide).
# =============================================================================
print("\n" + "="*80)
print("AVVIO PUNTO 5: Visualizzazione Raggruppamenti (Small Multiples)")
print("="*80)

import matplotlib.dates as mdates

# 1. Preparazione Dataset per il Plotting
# Dobbiamo "srotolare" la matrice daily_matrix per averla in formato "long"
# (adatto a seaborn.FacetGrid)
# Indice: data, Colonne: orari (00:00...23:45) -> Trasformiamo in righe

# Prendiamo solo le colonne dei dati di potenza
df_plot_source = daily_matrix.copy()

# Aggiungiamo le info di classificazione calcolate nel Punto 4
# df_classified ha 'word' e 'pattern_type' (Motif/Discord) indicizzati per data
df_plot_source = df_plot_source.join(df_classified[['word', 'pattern_type']])

# Filtriamo: Teniamo solo le TOP N parole più frequenti per non fare 100 grafici
TOP_N_PLOT = 12 # Numero di riquadri nella griglia (es. 3x4)
top_words = df_word_freq.head(TOP_N_PLOT)['word'].tolist()
df_plot_filtered = df_plot_source[df_plot_source['word'].isin(top_words)].copy()

print(f"Generazione grafici per le {TOP_N_PLOT} parole più frequenti...")

# "Melt" del dataframe: da formato largo (colonne orarie) a formato lungo (righe orarie)
# Id vars: date, word, pattern_type. Value vars: tutti gli orari.
df_long = df_plot_filtered.reset_index().melt(
    id_vars=['date', 'word', 'pattern_type'],
    var_name='time_obj',
    value_name='power'
)

# -----------------------------------------------------------------------------
# FIX ASSE X ("25"): Creiamo un campo datetime fittizio
# -----------------------------------------------------------------------------
# Usiamo una data arbitraria (es. 2000-01-01) + l'orario reale
# Questo permette a matplotlib di interpretare correttamente l'asse x come tempo
df_long['time_plot'] = df_long['time_obj'].apply(
    lambda t: pd.Timestamp(f"2000-01-01 {t.strftime('%H:%M:%S')}")
)

# Ordiniamo le parole per frequenza (così i grafici appaiono in ordine)
# Definiamo l'ordine categorico
df_long['word'] = pd.Categorical(
    df_long['word'],
    categories=top_words,
    ordered=True
)
df_long.sort_values(['word', 'time_plot'], inplace=True)

# 2. Definizione Colori (Blu=Motif, Rosso=Discord)
palette_map = {'Motif': '#1f77b4', 'Discord': '#d62728'}

# 3. Creazione FacetGrid (Griglia di Grafici)
g = sns.FacetGrid(
    df_long,
    col='word',
    col_wrap=4,       # 4 grafici per riga
    height=3.5,       # Altezza singolo grafico
    aspect=1.2,       # Larghezza = 1.2 * Altezza
    hue='pattern_type',
    palette=palette_map,
    sharey=True       # Stessa scala Y per confronto
)

# A. Disegno delle curve "spaghetti" (tutti i giorni sovrapposti)
g.map_dataframe(
    sns.lineplot,
    x='time_plot',
    y='power',
    units='date',     # Importante: dice a seaborn che ogni data è una linea separata
    estimator=None,   # Nessuna aggregazione automatica qui
    alpha=0.15,       # Trasparenza alta per vedere la densità
    linewidth=0.8
)

# B. Disegno del CENTROIDE (Media) sopra tutto
# Definiamo una funzione custom per mappare la media
def plot_mean_profile(x, y, **kwargs):
    # Calcola la media per ogni timestamp
    mean_data = pd.DataFrame({'x': x, 'y': y}).groupby('x').mean()
    # Usa un colore scuro/nero per il centroide per farlo risaltare
    plt.plot(mean_data.index, mean_data['y'], color='black', linewidth=2, linestyle='--')

g.map(plot_mean_profile, 'time_plot', 'power')

# 4. Rifinitura Estetica
# Formattazione asse X (HH:MM)
for ax in g.axes.flatten():
    # Imposta il formattatore per mostrare solo ore e minuti
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6)) # Tic ogni 6 ore
    ax.set_xlabel("Ora")
    ax.set_ylabel("Potenza [kW]")
    ax.grid(True, linestyle=':', alpha=0.6)

# Titolo Generale
g.fig.suptitle(
    f"Raggruppamento Profili per Parola SAX (Top {TOP_N_PLOT})\n"
    f"Configurazione: {BEST_WIN}h / {BEST_SYM} Simboli. Linea nera = Centroide (Media)",
    fontsize=16,
    y=1.02 # Sposta il titolo un po' su
)

# Legenda custom (perché FacetGrid a volte fa confusione con le legende miste)
from matplotlib.lines import Line2D
custom_lines = [
    Line2D([0], [0], color=palette_map['Motif'], lw=2),
    Line2D([0], [0], color=palette_map['Discord'], lw=2),
    Line2D([0], [0], color='black', lw=2, linestyle='--')
]
g.fig.legend(
    custom_lines,
    ['Motif (Pattern Frequente)', 'Discord (Pattern Raro)', 'Centroide (Media)'],
    loc='upper right',
    bbox_to_anchor=(0.98, 1.02),
    frameon=True
)

plt.tight_layout()
plt.show()

print(">>> Punto 5 completato.")

# =============================================================================
# PUNTO 6: CLUSTERING GERARCHICO AVANZATO (No One-Hot, k=2)
# =============================================================================
print("\n" + "=" * 80)
print("AVVIO PUNTO 6: Clustering Discord (Confronto -> Average -> k=2 -> Tree Numerico)")
print("=" * 80)

from sklearn.metrics import davies_bouldin_score

# 1. Preparazione Dati Discord
discord_dates = df_classified[df_classified['pattern_type'] == 'Discord'].index

if len(discord_dates) < 3:
    print("!!! ATTENZIONE: Troppi pochi giorni Discord (<3). Analisi saltata.")
else:
    discord_matrix = daily_matrix.loc[discord_dates].copy()


    # Normalizzazione Z-Score (Locale, per confrontare la forma pura)
    def z_norm(row):
        return (row - row.mean()) / row.std() if row.std() != 0 else np.zeros_like(row)


    X_discord = discord_matrix.apply(z_norm, axis=1).values
    dates_discord = discord_matrix.index

    # -------------------------------------------------------------------------
    # FASE A: CONFRONTO VISIVO DEI 4 METODI DI LINKAGE
    # -------------------------------------------------------------------------
    print("\n>>> Generazione 4 dendrogrammi separati (Single, Complete, Average, Ward)...")

    methods = ['single', 'complete', 'average', 'ward']
    linkage_matrices = {}

    for method in methods:
        plt.figure(figsize=(20, 10))

        Z = linkage(X_discord, method=method)
        linkage_matrices[method] = Z

        dendrogram(
            Z,
            labels=[d.strftime('%Y-%m-%d') for d in dates_discord],
            leaf_rotation=90.,
            leaf_font_size=10.,
            show_contracted=True
        )

        plt.title(f"Dendrogramma Metodo: {method.capitalize()}", fontsize=18)
        plt.subplots_adjust(bottom=0.25)  # Margine per date
        plt.show()

    # -------------------------------------------------------------------------
    # FASE B: ANALISI METRICHE SU 'AVERAGE'
    # -------------------------------------------------------------------------
    selected_method = 'average'
    print(f"\n>>> METODO SELEZIONATO: {selected_method.upper()}")

    k_range = range(2, min(10, len(discord_dates)))
    sil_scores, db_scores = [], []

    for k in k_range:
        cl = AgglomerativeClustering(n_clusters=k, linkage=selected_method)
        labels = cl.fit_predict(X_discord)
        sil_scores.append(silhouette_score(X_discord, labels))
        db_scores.append(davies_bouldin_score(X_discord, labels))

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].plot(list(k_range), sil_scores, 'bx-', lw=2, markersize=8)
    ax[0].set_title(f'Silhouette ({selected_method}) - Max è meglio')
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(list(k_range), db_scores, 'rx-', lw=2, markersize=8)
    ax[1].set_title(f'Davies-Bouldin ({selected_method}) - Min è meglio')
    ax[1].grid(True, alpha=0.3)
    plt.show()

    # -------------------------------------------------------------------------
    # FASE C: SCELTA MANUALE DI K E TAGLIO
    # -------------------------------------------------------------------------
    # --- MODIFICA RICHIESTA: k=2 ---
    best_k = 2
    print(f"\n>>> SCELTA FINALE: Taglio del dendrogramma '{selected_method}' a k={best_k}")

    Z_avg = linkage_matrices[selected_method]

    # Calcolo altezza taglio
    if best_k > 1:
        cut_height = Z_avg[-(best_k - 1), 2]
        color_thresh = cut_height * 0.99
    else:
        cut_height = 0
        color_thresh = 0

    plt.figure(figsize=(20, 10))
    dendrogram(
        Z_avg,
        labels=[d.strftime('%Y-%m-%d') for d in dates_discord],
        leaf_rotation=90,
        leaf_font_size=10,
        color_threshold=color_thresh
    )
    plt.axhline(y=cut_height, c='black', ls='--', lw=2, label=f'Taglio k={best_k}')
    plt.title(f'Dendrogramma DEFINITIVO ({selected_method.capitalize()}, k={best_k})', fontsize=18)
    plt.legend()
    plt.subplots_adjust(bottom=0.25)
    plt.show()

    # -------------------------------------------------------------------------
    # FASE D: ASSEGNAZIONE E ALBERO DECISIONALE (NUMERICO)
    # -------------------------------------------------------------------------
    # 1. Assegnazione
    model = AgglomerativeClustering(n_clusters=best_k, linkage=selected_method)
    cluster_labels = model.fit_predict(X_discord)

    discord_matrix['cluster'] = cluster_labels
    df_discord_clusters = discord_matrix.copy()

    print("\nDistribuzione Giorni per Cluster:")
    print(df_discord_clusters['cluster'].value_counts().sort_index())

    # 2. Visualizzazione Profili
    print("Visualizzazione Profili...")
    df_plot = df_discord_clusters.reset_index().melt(id_vars=['date', 'cluster'], var_name='time', value_name='power')
    df_plot['time_plot'] = df_plot['time'].apply(lambda t: pd.Timestamp(f"2000-01-01 {t.strftime('%H:%M:%S')}"))

    g = sns.FacetGrid(df_plot, col='cluster', col_wrap=2, height=5, aspect=1.8, sharey=True)
    g.map_dataframe(sns.lineplot, x='time_plot', y='power', units='date', estimator=None, alpha=0.2, color='gray')


    def plot_centroid(x, y, **kwargs):
        data = pd.DataFrame({'x': x, 'y': y})
        mean = data.groupby('x').mean()
        plt.plot(mean.index, mean['y'], 'r--', lw=3)


    g.map(plot_centroid, 'time_plot', 'power')

    for ax in g.axes.flatten():
        ax.set_xlim(df_plot['time_plot'].min(), df_plot['time_plot'].max())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.grid(True, ls=':', alpha=0.5)

    g.fig.suptitle(f'Profili Anomalie (Metodo {selected_method}, k={best_k})', y=1.02, fontsize=16)
    plt.show()

    # 3. Albero Decisionale (NUMERICO - NO ONE HOT)
    print("\nAddestramento Albero Decisionale (Feature Numeriche)...")

    df_ml = df_discord_clusters[['cluster']].copy()
    df_ml.index = pd.to_datetime(df_ml.index)  # Sicurezza indice

    # --- Feature Numeriche ---
    df_ml['day_of_week'] = df_ml.index.dayofweek  # 0=Lun, 6=Dom
    df_ml['month'] = df_ml.index.month  # 1=Gen, 12=Dic
    df_ml['is_weekend'] = (df_ml['day_of_week'] >= 5).astype(int)

    X = df_ml[['day_of_week', 'month', 'is_weekend']]
    y = df_ml['cluster']

    # Fit
    clf = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
    clf.fit(X, y)

    # Plot Tree
    plt.figure(figsize=(14, 8))
    plot_tree(
        clf,
        feature_names=['GiornoSett (0-6)', 'Mese (1-12)', 'Weekend (0/1)'],
        filled=True,
        rounded=True,
        class_names=[f"Cluster {i}" for i in range(best_k)],
        fontsize=12
    )
    plt.title(f'Decision Tree Numerico (Regole per k={best_k})', fontsize=18)
    plt.show()

    # Score
    y_pred = clf.predict(X)
    print(f"Accuratezza Albero: {accuracy_score(y, y_pred) * 100:.1f}%")

print(">>> Punto 6 completato.")


# =============================================================================
# PUNTO 7: ANALISI CAUSE FISICHE (Confronto kW Reali Cabina vs Cooling)
# Obiettivo: Confrontare i profili Discord con il Cooling usando i valori REALI (kW).
# Soluzione: Doppio Asse Y + Rimozione Outlier Artificiali.
# =============================================================================
print("\n" + "=" * 80)
print("AVVIO PUNTO 7: Analisi Cause (kW Reali con Pulizia Outlier)")
print("=" * 80)

# 1. Caricamento e PULIZIA Dataset Cooling
try:
    try:
        df_cooling = pd.read_csv('data/df_cooling_3.csv', parse_dates=['date_time'])
    except FileNotFoundError:
        df_cooling = pd.read_csv('df_cooling_3.csv', parse_dates=['date_time'])

    df_cooling.set_index('date_time', inplace=True)
    df_cooling.sort_index(inplace=True)

    # --- RIMOZIONE OUTLIER ARTIFICIALI (COOLING) ---
    # Calcoliamo i quartili per eliminare i picchi assurdi inseriti dal prof
    Q1 = df_cooling['power_mech_room'].quantile(0.25)
    Q3 = df_cooling['power_mech_room'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Sostituiamo gli outlier con NaN e interpoliamo
    mask_outlier = (df_cooling['power_mech_room'] < lower_bound) | (df_cooling['power_mech_room'] > upper_bound)
    if mask_outlier.sum() > 0:
        print(f"Rilevati e rimossi {mask_outlier.sum()} outlier artificiali nel Cooling.")
        df_cooling.loc[mask_outlier, 'power_mech_room'] = np.nan
        df_cooling['power_mech_room'] = df_cooling['power_mech_room'].interpolate(method='linear')

    # Creazione Matrice
    df_cooling['date'] = df_cooling.index.date
    df_cooling['time'] = df_cooling.index.time
    cooling_matrix = df_cooling.pivot_table(index='date', columns='time', values='power_mech_room')

    # 2. Selezione Giorni Discord
    # Se non hai eseguito il punto 6, ricalcola i discord grezzi
    if 'discord_dates' not in locals():
        discord_dates = df_classified[df_classified['pattern_type'] == 'Discord'].index

    common_dates = discord_dates.intersection(cooling_matrix.index)

    if len(common_dates) == 0:
        print("Nessuna data comune trovata.")
    else:
        # Recupero Cluster dal Punto 6 (se esiste), altrimenti gruppo unico
        if 'df_discord_clusters' in locals():
            clusters_info = df_discord_clusters['cluster']
        else:
            clusters_info = pd.Series(0, index=common_dates)

        unique_clusters = np.unique(clusters_info)

        # Setup Assi X
        x_axis = np.arange(96)
        tick_indices = np.arange(0, 96, 12)
        tick_labels = [f"{h:02d}:00" for h in range(0, 24, 3)]

        # 3. Visualizzazione Comparativa (Doppio Asse Y)
        # ---------------------------------------------------------------------
        print(f"Generazione grafici per {len(unique_clusters)} tipologie di anomalie...")

        for c in unique_clusters:
            # Filtro date del cluster corrente
            dates_in_cluster = clusters_info[clusters_info == c].index.intersection(common_dates)

            if len(dates_in_cluster) == 0: continue

            # Dati Reali (kW)
            prof_tot = daily_matrix.loc[dates_in_cluster]  # Cabina
            prof_cool = cooling_matrix.loc[dates_in_cluster]  # Cooling

            # Medie
            mean_tot = prof_tot.mean()
            mean_cool = prof_cool.mean()

            # --- PLOT ---
            fig, ax1 = plt.subplots(figsize=(14, 7))

            # ASSE SINISTRO (BLU): Totale Cabina
            color_1 = 'tab:blue'
            ax1.set_xlabel('Ora del Giorno')
            ax1.set_ylabel('Potenza CABINA [kW]', color=color_1, fontweight='bold', fontsize=12)

            # Spaghetti Plot (Sfondo Blu)
            for d in dates_in_cluster:
                ax1.plot(x_axis, prof_tot.loc[d].values, color=color_1, alpha=0.1, linewidth=0.8)

            # Media (Linea Spessa Blu)
            ax1.plot(x_axis, mean_tot.values, color=color_1, linewidth=3, label='Media Cabina (Totale)')
            ax1.tick_params(axis='y', labelcolor=color_1)
            ax1.grid(True, alpha=0.3)

            # ASSE DESTRO (ROSSO): Cooling
            ax2 = ax1.twinx()
            color_2 = 'tab:red'
            ax2.set_ylabel('Potenza COOLING [kW]', color=color_2, fontweight='bold', fontsize=12)

            # Spaghetti Plot (Sfondo Rosso)
            for d in dates_in_cluster:
                ax2.plot(x_axis, prof_cool.loc[d].values, color=color_2, alpha=0.1, linewidth=0.8)

            # Media (Linea Spessa Rossa Tratteggiata)
            ax2.plot(x_axis, mean_cool.values, color=color_2, linewidth=3, linestyle='--', label='Media Cooling')
            ax2.tick_params(axis='y', labelcolor=color_2)

            # Titolo e Legende
            plt.title(
                f"CLUSTER {c}: Confronto Profili Reali (n={len(dates_in_cluster)} giorni)\n(Notare le diverse scale sugli assi Y)",
                fontsize=14)
            ax1.set_xticks(tick_indices)
            ax1.set_xticklabels(tick_labels)

            # Legenda Unica
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', frameon=True)

            plt.tight_layout()
            plt.show()

            # Calcolo Correlazione Pearson
            corr = mean_tot.corr(mean_cool)
            print(f">>> Cluster {c}: Correlazione Pearson = {corr:.2f}")
            if corr > 0.7:
                print("    -> L'anomalia segue l'andamento del Cooling (Causa Probabile: Gruppo Frigo).")
            elif corr < 0.3:
                print("    -> L'anomalia è indipendente dal Cooling (Causa Probabile: Altro).")
            else:
                print("    -> Correlazione parziale.")

except Exception as e:
    print(f"Errore: {e}")

print("\n>>> ANALISI COMPLETATA.")

# =============================================================================
# APPENDIX: STAMPA MASSIVA DI TUTTI I CONTEGGI
# (Da incollare alla fine dello script)
# =============================================================================
print("\n" + "#" * 80)
print("APPENDIX: CONTEGGIO PAROLE COMPLETO (TUTTE LE CONFIGURAZIONI)")
print("#" * 80)

# Iteriamo di nuovo sulle opzioni definite nel codice precedente
for w in window_options:
    for a in alphabet_options:
        config_label = f"Finestra: {w}h | Simboli: {a}"

        # Ricalcoliamo velocemente la serie SAX per questa configurazione
        # (Il calcolo è leggero, richiede frazioni di secondo)
        current_sax_series = daily_matrix.apply(
            lambda x: apply_sax_to_day(x.values, w, a)[0], axis=1
        )

        # Calcolo frequenze
        counts = current_sax_series.value_counts()

        print(f"\n---> {config_label}")
        print(f"     Numero totale pattern distinti: {len(counts)}")
        print("-" * 40)

        # Stampa la lista completa (senza i puntini '...')
        print(counts.to_string())
        print("-" * 40)

print("\n>>> Esecuzione terminata.")