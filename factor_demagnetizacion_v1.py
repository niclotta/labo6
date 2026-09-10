#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 11:43:02 2026

@author: nclotta
"""

# Time-stamp: </Users/nclotta/Documents/__UBA/__LABO_6_SEIS/codigo/factor_demagnetizacion_v1.py, 2026-09-10 Thursday 15:52:42 nclotta>

import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np
import pathlib
import natsort
import math

from scipy.optimize import curve_fit
from scipy.stats    import linregress

# Datos

masa_dsc = 0.0075 # gramos
masa_pmx = 0.0348 # gramos

densidad_acero = 7.8 #g/cm3

err_mqna = 0.001  # emu
err_masa = 0.0001 # gramos

# Setup variable

lead_fct = "Fa0"
sample_g = False

# Setup permanente

geometrias = ["Prisma X", "Disco", "Prisma X vertical"]
colnames = ["Iteration", "Segment", "Field", "Moment", "Time Stamp", "Field Status", "Moment Status", "Invalid"]

def cifras(val, err):
    if err == 0:
        decimals = 0
    else:
        exp = math.floor(math.log10(abs(err)))
        decimals = max(0, 1 - exp)
    return f"{val:.{decimals}f}"

def indices_rango_lineal_centered(H, window_size=100):
    idx_zero_crossing = np.argmin(np.abs(H))
    half_window = window_size // 2
    start = max(0, idx_zero_crossing - half_window)
    stop = min(len(H), idx_zero_crossing + half_window)
    return start, stop

def indices_rango_lineal_field(H, field_limit=300):
    valid_indices = np.where((H <= field_limit) & (H >= -field_limit))[0]
    if len(valid_indices) < 2:
        return indices_rango_lineal_centered(H)

    start = valid_indices[0]
    stop = valid_indices[-1]
    return start, stop

def ajuste_N_d_demagnetizacion(H, M, geometry, sigma, dataset):
    def lineal(a, b, x):
        return a * x + b

    start, stop = indices_rango_lineal_field(H)
    H_rec = H[start:stop]
    M_rec = M[start:stop]

    popt, pcov = curve_fit(lineal, H_rec, M_rec, sigma=sigma[start:stop])
    N_d = 1/popt[0]
    N_d_err = (N_d ** 2) * np.sqrt(np.diag(pcov))[0]

    if sample_g:
        plt.scatter(H, M)
        plt.scatter(H_rec, M_rec, zorder=5)
        plt.title(rf"$N_d$ del {geometry.lower().replace(' x', ' X')} a {dataset.replace('g', '°')}")
        plt.grid()
        plt.show()
        plt.close()
    return N_d, N_d_err

for geometry in geometrias:
    if geometry == "Disco":
        major_head = 49
        filename = ["minor_1er_*Hysteresis*.csv", "major_1er_*Hysteresis*.csv"]
        datasets = ["0g", "30g", "60g", "90g", "120g", "150g"]
        angulos  = [0, 30, 60, 90, 120, 150]
        conj_med = "27_08"
        masa = masa_dsc
    elif geometry == "Prisma X":
        major_head = 85
        filename = ["minor_*Hysteresis*", "major_*Segment*"]
        datasets = ["0g", "15g", "30g", "45g", "60g", "75g", "90g"]
        angulos  = [0, 15, 30, 45, 60, 75, 90]
        conj_med = "03_09"
        masa = masa_pmx
    elif geometry == "Prisma X vertical":
        major_head = 85
        filename = ["minor_*Hysteresis*", "major_*Segment*"]
        datasets = ["0g", "15g", "30g", "45g", "60g", "90g"] # "75g",
        angulos  = [0, 15, 30, 45, 60, 90] # 75,
        conj_med = "08_09"
        masa = masa_pmx
    N_d_a = []
    N_d_e = []

    for i in range(len(datasets)):                     # /{conj_med}
        d = pathlib.Path(f"../data/mediciones/{lead_fct}/{geometry}/{datasets[i]}/")
        print(f"========== {geometry}: {datasets[i]} ==========")
        for q in natsort.natsorted(d.glob(f"{filename[1]}")):
            df = pd.read_csv(q, header=major_head, names=colnames)
            H_raw = df["Field"].values
            M_raw = df["Moment"].values / masa * densidad_acero
            M_raw_err = np.sqrt((err_mqna**2)/masa**2 + (M_raw * err_masa**2)/masa) # ** 3)
            M_raw_err[np.isnan(M_raw_err)] = 0.125

            idx_min = np.argmin(H_raw)
            idx_max_after_min = idx_min + np.argmax(H_raw[idx_min:])

            H_dec = H_raw[idx_max_after_min:]
            M_dec = M_raw[idx_max_after_min:]

            if len(H_dec) < 5: 
                idx_max_before_min = np.argmax(H_raw[:idx_min])
                H_dec = H_raw[idx_max_before_min:idx_min]
                M_dec = M_raw[idx_max_before_min:idx_min]
                M_dec_err = M_raw_err[idx_max_before_min:idx_min]

            N_d, N_d_err = ajuste_N_d_demagnetizacion(H_dec, M_dec, geometry, M_dec_err, datasets[i])
            N_d_e.append(N_d_err)
            N_d_a.append(N_d)
            print(f"${datasets[i].replace('g', '^\\circ')}\\to N_d=({cifras(N_d, N_d_err)}\\pm{N_d_err:.2g})$")

    plt.errorbar(angulos, N_d_a, yerr=N_d_e, marker="v", mfc='black', mec='black',
                     color="steelblue", linestyle="--", ecolor="red", capsize=4, elinewidth=1.5)
    plt.title(geometry)
    plt.xlabel(r"Angulo [°]", fontsize=14)
    plt.ylabel(r'$N_d$', fontsize=14)
    plt.show()
    plt.close()
            
