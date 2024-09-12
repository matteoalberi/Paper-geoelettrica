import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import matplotlib as mpl
mpl.use('TkAgg')

# Percorso del file di input
input_file = 'C:/Users/lopan/PycharmProjects/STELLA/Res/DEFINITIVI/1/XYZr_Joint_80perc.txt'

# Carica i dati dal file
data = np.loadtxt('C:/Users/lopan/PycharmProjects/STELLA/Res/DEFINITIVI/1/XYZr_Joint_80perc.txt', skiprows=1)

# Estrai le colonne
X = data[:, 0]
Y = data[:, 1]
Z = data[:, 2]
resistivity = data[:, 3]

# Filtra i dati per il range specificato
mask = (X >= -9) & (X <= 9) & (Y >= -8) & (Y <= 8) & (Z >= -1) & (Z <= 0)
filtered_X = X[mask]
filtered_Y = Y[mask]
filtered_resistivity = resistivity[mask]

# Definisci le posizioni X specifiche
X_positions = [7.8, 4.8, 1.9, -1.5, -4.1, -7.0]

# Funzione per calcolare la media dei valori di resistività compresi tra i NaN
def compute_avg_resistivity(Y, resistivity):
    avg_resistivity = []
    avg_Y = []

    start_idx = 0
    for i in range(1, len(resistivity)):
        if np.isnan(resistivity[i]):
            if start_idx != i:
                avg_resistivity.append(np.nanmean(resistivity[start_idx:i]))
                avg_Y.append(np.nanmean(Y[start_idx:i]))
            start_idx = i + 1

    # Gestisci l'ultimo intervallo
    if start_idx < len(resistivity):
        avg_resistivity.append(np.nanmean(resistivity[start_idx:]))
        avg_Y.append(np.nanmean(Y[start_idx:]))

    return np.array(avg_Y), np.array(avg_resistivity)


# Percorso per salvare i file di testo
output_file_txt = os.path.join(os.path.dirname(input_file), 'resistivity_variation.txt')

# inizializzazione dataframe
df_res = pd.DataFrame({'y_avg': np.empty(0)})

# Itera su ciascuna posizione X specifica
for X_fixed in X_positions:
    Y_fixed_mask = filtered_X == X_fixed
    Y_fixed = filtered_Y[Y_fixed_mask]
    resistivity_fixed = filtered_resistivity[Y_fixed_mask]

    # Ordina i dati per Y
    sorted_indices = np.argsort(Y_fixed)
    Y_fixed_sorted = Y_fixed[sorted_indices]
    resistivity_fixed_sorted = resistivity_fixed[sorted_indices]

    # Calcola la media dei valori di resistività compresi tra i NaN
    Y_avg, resistivity_avg = compute_avg_resistivity(Y_fixed_sorted, resistivity_fixed_sorted)
    # popolamento dataframe
    df_res = df_res.merge(pd.DataFrame({'y_avg': Y_avg, 'rho_'+str(X_fixed): resistivity_avg}), on='y_avg', how='outer')

# interpolazione dati mancanti df
df_res = df_res.interpolate()
# Salva i risultati nel file di testo
df_res.to_csv(output_file_txt, sep='\t', float_format='%.3f', index=False)
print(f'File di testo salvato in: {output_file_txt}')
labels = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6']

# Create the plot and add the curves
plt.figure(figsize=(15, 10))
for i, X_fixed in enumerate(X_positions):
    plt.plot(df_res.y_avg, df_res['rho_'+str(X_fixed)], marker='o', linestyle='-', label=labels[i])


# Ora ottieni i limiti verticali e crea le fasce
y_min, y_max = plt.ylim()  # Ottieni i limiti verticali del grafico
phases = [(-6.4, -5.2), (-3.6, -2.4), (-0.8, 0.4), (2, 3.2), (4.8, 6) , (7.6, 8)]
labels = ['F1 Non arato', 'F2 arato', 'F3 Non arato', 'F4 Arato', 'F5 Non arato', 'F6 Arato']

# Aggiungi le fasce verticali con altezza costante
for (start, end), label in zip(phases, labels):
    plt.fill_betweenx([y_min, y_max], start, end, color='red', alpha=0.2)
    plt.text((start + end) / 2, y_min - 0.1 * (y_max - y_min), label, color='red', ha='center', va='top', fontsize=9)

# Aggiungi le stelle gialle con contorno nero
star_positions = [-7.2, -4.4, -1.6, 1.2, 4, 6.8]
for star_pos in star_positions:
    plt.plot(star_pos, y_max + 0.03 * (y_max - y_min), marker='*', color='yellow', markersize=15,
             markeredgecolor='black', markeredgewidth=1.5, label='Pianta' if star_pos == star_positions[0] else "")

# Configura il grafico
plt.xlabel('Y')
plt.ylabel('Resistivity (Ohm*m)')
plt.title('Average resistivity variation along Y (0 to -1 m Depth)')
plt.grid(True)
plt.legend(loc='center left', bbox_to_anchor=(1, 0), title='Legend') # Aggiunge la legenda


# Imposta i tick dell'asse x ogni metro
plt.xticks(np.arange(-8, 9, 1))

# Percorso per salvare il grafico come immagine
output_file_image = r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\resistivity_variation.png'
plt.savefig(output_file_image)  # Salva il grafico come immagine
plt.close()  # Chiude il grafico per liberare memoria