import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Percorso del file di input
input_file = 'C:/Users/lopan/PycharmProjects/STELLA/Res/DEFINITIVI/1/resistivity_variation.txt'

# Legge i dati dal file resistivity_variation.txt
df_res = pd.read_csv(input_file, sep='\t')

# Estrai le colonne che rappresentano le diverse posizioni X (quelle che iniziano con 'rho_')
rho_columns = [col for col in df_res.columns if col.startswith('rho_')]

# Estrai le posizioni X dalle colonne 'rho_X' (eliminando il prefisso 'rho_')
X_positions = [col.replace('rho_', '') for col in rho_columns]

# Crea il grafico e aggiungi le curve
plt.figure(figsize=(15, 10))
for i, col in enumerate(rho_columns):
    plt.plot(df_res['y_avg'], df_res[col], marker='o', linestyle='-', label=f'x={X_positions[i]}')

# Ottieni i limiti verticali e crea le fasce
y_min, y_max = plt.ylim()  # Ottieni i limiti verticali del grafico
phases = [(-6.2, -5.4), (-3.4, -2.6), (-0.6, 0.2), (2.2, 3), (5, 5.8), (7.6, 8)]
phase_labels = ['F1 Non arato', 'F2 arato', 'F3 Non arato', 'F4 Arato', 'F5 Non arato', 'F6 Arato']

# Aggiungi le fasce verticali con altezza costante
for (start, end), label in zip(phases, phase_labels):
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
plt.legend(loc='center left', bbox_to_anchor=(1, 0), title='Legend')  # Aggiunge la legenda

# Imposta i tick dell'asse x ogni metro
plt.xticks(np.arange(-8, 9, 1))

# Percorso per salvare il grafico come immagine
output_file_image = r'C:\Users\lopan\PycharmProjects\STELLA\Res\Joint\resistivity_variation.png'
plt.savefig(output_file_image)  # Salva il grafico come immagine
plt.close()  # Chiude il grafico per liberare memoria
