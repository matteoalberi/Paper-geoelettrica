import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Caricamento del file Excel con la prima riga come header
file_path = 'Andamento_inversioni_1.xlsx'
df = pd.read_excel(file_path, sheet_name='Foglio1', header=1)

# Rimozione degli spazi bianchi dai nomi delle colonne
df.columns = df.columns.str.strip()

# Pulizia dei dati
df_clean = df[
    pd.to_numeric(df['Objective_function'], errors='coerce').notnull() |
    pd.to_numeric(df['Chi_squared.2'], errors='coerce').notnull()
]

# Assicurarsi che l'iterazione 14 di JOINT sia presente
iteration_14 = df[df['Iteration.2'] == 14]
df_clean = df_clean.append(iteration_14, ignore_index=True)

# Pulizia e conversione delle colonne
df_clean['Iteration'] = pd.to_numeric(df_clean['Iteration'], errors='coerce')
df_clean['Iteration.1'] = pd.to_numeric(df_clean['Iteration.1'], errors='coerce')
df_clean['Iteration.2'] = pd.to_numeric(df_clean['Iteration.2'], errors='coerce')

# Estrazione dei dati necessari
iterations_ccr = df_clean['Iteration'].dropna().astype(int)
iterations_joint = df_clean['Iteration.2'].dropna().astype(int)

# Lettura di Beta
beta_ccr = df_clean['Beta'].astype(float)
beta_gcr = df_clean['Beta.1'].astype(float)
beta_joint = df_clean['Beta.2'].astype(float)

# Lettura di Regularization
regularization_ccr = df_clean['Regularization'].astype(float)
regularization_gcr = df_clean['Regularization.1'].astype(float)
regularization_joint = df_clean['Regularization.2'].astype(float)

# Lettura di Objective Function
objective_function_ccr = df_clean['Objective_function'].astype(float)
objective_function_gcr = df_clean['Objective_function.1'].astype(float)
objective_function_joint = df_clean['Objective_function.2'].astype(float)

# Lettura di Chi Squared
chi_squared = df_clean['Chi_squared'].astype(float)
chi_squared_gcr = df_clean['Chi_squared.1'].astype(float)
chi_squared_joint = df_clean['Chi_squared.2'].astype(float)

# Creazione del grafico con quattro pannelli (subplots)
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Parametri per font più grandi
label_fontsize = 16
tick_fontsize = 14

# Definisco i tick da 1 a 14
ticks = list(range(1, 15))

# Plot per χ² (pannello in alto a sinistra)
axs[0, 0].plot(iterations_ccr, chi_squared[:len(iterations_ccr)], marker='o', color='#00c4fe', label='CCR')
axs[0, 0].plot(iterations_ccr, chi_squared_gcr[:len(iterations_ccr)], marker='o', color='#4ce600', label='GCR')
axs[0, 0].plot(iterations_joint, chi_squared_joint[:len(iterations_joint)], marker='o', color='#ff0000', label='JOINT')
axs[0, 0].set_xlabel('Iterations', fontsize=label_fontsize)
axs[0, 0].set_ylabel(r'$\chi^2$', fontsize=label_fontsize)
axs[0, 0].set_xticks(ticks)
axs[0, 0].legend(loc='upper right', frameon=False, fontsize=label_fontsize-2, borderpad=0.8)
axs[0, 0].grid(False)
axs[0, 0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
axs[0, 0].text(0.05, 0.95, '(a)', transform=axs[0, 0].transAxes, fontsize=label_fontsize, fontweight='bold',
               ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
axs[0, 0].set_ylim(chi_squared.min() * 0, chi_squared.max() * 2.5)

# Plot per β (pannello in alto a destra)
axs[0, 1].plot(iterations_ccr, beta_ccr[:len(iterations_ccr)], marker='o', color='#00c4fe', label='CCR')
axs[0, 1].plot(iterations_ccr, beta_gcr[:len(iterations_ccr)], marker='o', color='#4ce600', label='GCR')
axs[0, 1].plot(iterations_joint, beta_joint[:len(iterations_joint)], marker='o', color='#ff0000', label='JOINT')
axs[0, 1].set_xlabel('Iterations', fontsize=label_fontsize)
axs[0, 1].set_ylabel(r'$\beta$', fontsize=label_fontsize)
axs[0, 1].set_yscale('log')
axs[0, 1].set_xticks(ticks)
axs[0, 1].legend(loc='upper right', frameon=False, fontsize=label_fontsize-2, borderpad=0.8)
axs[0, 1].grid(False)
axs[0, 1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
axs[0, 1].text(0.05, 0.95, '(b)', transform=axs[0, 1].transAxes, fontsize=label_fontsize, fontweight='bold',
               ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
axs[0, 1].set_ylim(min(beta_ccr.min(), beta_gcr.min(), beta_joint.min()) * 0, 
                   max(beta_ccr.max(), beta_gcr.max(), beta_joint.max()) * 4)

# Plot per Regularization (pannello in basso a sinistra)
axs[1, 0].plot(iterations_ccr, regularization_ccr[:len(iterations_ccr)], marker='o', color='#00c4fe', label='CCR')
axs[1, 0].plot(iterations_ccr, regularization_gcr[:len(iterations_ccr)], marker='o', color='#4ce600', label='GCR')
axs[1, 0].plot(iterations_joint, regularization_joint[:len(iterations_joint)], marker='o', color='#ff0000', label='JOINT')
axs[1, 0].set_xlabel('Iterations', fontsize=label_fontsize)
axs[1, 0].set_ylabel('Regularization', fontsize=label_fontsize)
axs[1, 0].set_xticks(ticks)
axs[1, 0].legend(loc='upper right', frameon=False, fontsize=label_fontsize-2, borderpad=0.8)
axs[1, 0].grid(False)
axs[1, 0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
axs[1, 0].text(0.05, 0.95, '(c)', transform=axs[1, 0].transAxes, fontsize=label_fontsize, fontweight='bold',
               ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
axs[1, 0].set_ylim(min(regularization_ccr.min(), regularization_gcr.min(), regularization_joint.min()) * -2, 
                   max(regularization_ccr.max(), regularization_gcr.max(), regularization_joint.max()) * 1.5)

# Plot per Objective Function (pannello in basso a destra)
axs[1, 1].plot(iterations_ccr, objective_function_ccr[:len(iterations_ccr)], marker='o', color='#00c4fe', label='CCR')
axs[1, 1].plot(iterations_ccr, objective_function_gcr[:len(iterations_ccr)], marker='o', color='#4ce600', label='GCR')
axs[1, 1].plot(iterations_joint, objective_function_joint[:len(iterations_joint)], marker='o', color='#ff0000', label='JOINT')
axs[1, 1].set_xlabel('Iterations', fontsize=label_fontsize)
axs[1, 1].set_ylabel('Objective function', fontsize=label_fontsize)
axs[1, 1].set_yscale('log')
axs[1, 1].set_xticks(ticks)
axs[1, 1].legend(loc='upper right', frameon=False, fontsize=label_fontsize-2, borderpad=0.8)
axs[1, 1].grid(False)
axs[1, 1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
axs[1, 1].text(0.05, 0.95, '(d)', transform=axs[1, 1].transAxes, fontsize=label_fontsize, fontweight='bold',
               ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
axs[1, 1].set_ylim(min(objective_function_ccr.min(), objective_function_gcr.min(), objective_function_joint.min()) * 0,
                   max(objective_function_ccr.max(), objective_function_gcr.max(), objective_function_joint.max()) * 2.5)

# Imposto uno stile unificato
plt.tight_layout()

# Salvataggio della figura
plt.savefig('Figure_4.png', dpi=300)  # Salva la figura come file PNG con una risoluzione di 300 dpi

# Visualizzo il grafico
plt.show()
