#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 11:43:02 2026

@author: nclotta
"""

# Time-stamp: </Users/nclotta/Documents/__UBA/__LABO_6_SEIS/codigo/factor_demagnetizacion_v1.py, 2026-10-15 Thursday 14:45:22 nclotta>

import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np
import pathlib
import natsort
import math

from scipy.optimize import curve_fit

import W_ext_v1

# Datos

densidad_acero = 7.8 #g/cm3

err_mqna = 0.001  # emu
err_masa = 0.0001 # gramos

# Setup variable

sample_g = False
graficos = True

# Setup permanente

lead_factors = ["Fa0", "Fa93", "Fa186"]
geometrias = ["Prisma X", "Prisma X vertical", "Cubo", "Disco", "Disco horizontal", "Polvo"]
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
    def lineal(x, a, b):
        return a * x + b

    start, stop = indices_rango_lineal_field(H)
    H_rec = H[start:stop]
    M_rec = M[start:stop]

    popt, pcov = curve_fit(lineal, H_rec, M_rec, sigma=sigma[start:stop], absolute_sigma=True)
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

for lead_fct in lead_factors:
    if lead_fct == "Fa0":
        masa_dsc = 0.0075 # gramos
        masa_pmx = 0.0348 # gramos
        masa_plv = 0.0032 # gramos
        masa_cbo = 0.0057 # gramos
    elif lead_fct == "Fa93":
        masa_dsc = 0.0042 # gramos
        masa_pmx = 0.0322 # gramos
        masa_plv = 0.0032 # gramos COMPLETAR
        masa_cbo = 0.0048 # gramos
    elif lead_fct == "Fa186":
        masa_dsc = 0.0046 # gramos
        masa_pmx = 0.0297 # gramos
        masa_plv = 0.0032 # gramos COMPLETAR
        masa_cbo = 0.0076 # gramos

    for geometry in geometrias:
        if geometry == "Disco":
            major_head = 49 if lead_fct == "Fa0" else 85
            if lead_fct == "Fa0":
                filename = ["minor_1er_*Hysteresis*.csv", "major_1er_*Hysteresis*.csv"]
            else:
                filename = ["minor_*Hysteresis*", "major_*Segment*"]
            datasets = ["0g", "30g", "60g", "90g", "120g", "150g"]
            angulos  = [0, 30, 60, 90, 120, 150]
            masa = masa_dsc
        if geometry == "Disco horizontal":
            major_head = 85
            filename = ["minor_*Hysteresis*", "major_*Segment*"]
            datasets = ["0g", "30g", "60g", "90g", "120g"]
            angulos  = [0, 30, 60, 90, 120]
            masa = masa_dsc
        elif geometry == "Prisma X":
            major_head = 85
            filename = ["minor_*Hysteresis*", "major_*Segment*"]
            datasets = ["0g", "15g", "30g", "45g", "60g", "75g", "90g"]
            angulos  = [0, 15, 30, 45, 60, 75, 90]
            masa = masa_pmx
        elif geometry == "Prisma X vertical":
            major_head = 85
            filename = ["minor_*Hysteresis*", "major_*Segment*"]
            datasets = ["0g", "15g", "30g", "45g", "60g", "75g", "90g"]
            angulos  = [0, 15, 30, 45, 60, 75, 90]
            masa = masa_pmx
        elif geometry == "Polvo":
            major_head = 85
            filename = ["minor_*Hysteresis*", "major_*Segment*"]
            datasets = ["30g", "90g"]
            angulos  = [30, 90]
            masa = masa_plv
        elif geometry == "Cubo":
            major_head = 85
            filename = ["minor_*Hysteresis*", "major_*Segment*"]
            datasets = ["0g", "15g", "30g", "45g", "60g"]
            angulos  = [0, 15, 30, 45, 60]
            masa = masa_cbo
        N_d_a = []
        N_d_e = []

        for i in range(len(datasets)):
            d = pathlib.Path(f"../data/mediciones/{lead_fct}/{geometry}/{datasets[i]}/")
            if not d.exists():
                continue
            print(f"========== {geometry} {lead_fct}: {datasets[i]} ==========")
            for q in natsort.natsorted(d.glob(f"{filename[1]}")):
                df = pd.read_csv(q, header=major_head, names=colnames)
                H_raw = df["Field"].values
                M_raw = df["Moment"].values / masa * densidad_acero
                M_raw_err = np.sqrt(((err_mqna * densidad_acero) / masa)**2 + ((M_raw * err_masa) / masa)**2)
                M_raw_err = np.sqrt((err_mqna**2)/masa**2 + (M_raw * err_masa**2)/masa)
                M_raw_err[np.isnan(M_raw_err)] = 0.125

                idx_min = np.argmin(H_raw)
                idx_max_after_min = idx_min + np.argmax(H_raw[idx_min:])

                H_dec = H_raw[idx_max_after_min:]
                M_dec = M_raw[idx_max_after_min:]
                M_dec_err = M_raw_err[idx_max_after_min:]

                if len(H_dec) < 5: 
                    idx_max_before_min = np.argmax(H_raw[:idx_min])
                    H_dec = H_raw[idx_max_before_min:idx_min]
                    M_dec = M_raw[idx_max_before_min:idx_min]
                    M_dec_err = M_raw_err[idx_max_before_min:idx_min]

                sat_indices = np.where(H_raw > 0.95 * np.max(H_raw))
                _, cruce = np.polyfit(H_raw[sat_indices], M_raw[sat_indices], 1)
                M_s = abs(cruce)
                if ((datasets[i] != "0g" and "Disco" in geometry)
                        or "Cubo" in geometry
                        or "Prisma X" in geometry
                        or "horizontal" in geometry):
                    M_s = M_raw[idx_max_after_min - 1]

                N_d, N_d_err = ajuste_N_d_demagnetizacion(H_dec, M_dec, geometry, M_dec_err, datasets[i])
                print(f"${datasets[i].replace('g', '^\\circ')}\\to N_d=({cifras(N_d, N_d_err)}\\pm{N_d_err:.2g})$")
                N_d_e.append(N_d_err)
                N_d_a.append(N_d)

                M_pos, H_pos = M_dec[H_dec > 0], H_dec[H_dec > 0]
                target_lambdas = [0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                cut_idx = [np.abs(M_pos/M_s - target).argmin() for target in target_lambdas]
                H_cut, M_cut = H_pos[cut_idx], M_pos[cut_idx]

                W_e_a = []
                W_i_a = []
                for j, M in enumerate(M_cut):
                    W_ext = W_ext_v1.calcular(H_cut[j], M, M_s, N_d)
                    W_in  = W_ext_v1.W_ext_integral(H_cut[j], M, M_s, N_d)
                    W_e_a.append(W_ext)
                    W_i_a.append(W_in)
                if graficos:
                    plt.errorbar(M_cut/M_s, W_e_a, marker="v", mfc='black', mec='black',
                                     color="steelblue", linestyle="--", ecolor="red", capsize=4, elinewidth=1.5)
                    #plt.title(fr"{geometry} ${datasets[i].replace('g', '^\\circ')}$")
                    plt.xlabel(r"$\lambda$", fontsize=14)
                    plt.ylabel(r'$W_{ext}$ [erg/cm$^3$]', fontsize=14)
                    plt.savefig(f"./img/{lead_fct}/{geometry}/W_ext_lambda_{datasets[i]}_{geometry}.png", dpi=300)
                    plt.close()

        if len(angulos) == len(N_d_a):
            plt.errorbar(angulos, N_d_a, yerr=N_d_e, marker="v", mfc='black', mec='black',
                         color="steelblue", linestyle="--", ecolor="red", capsize=4, elinewidth=1.5)
            plt.xlabel(r"Ángulo [°]", fontsize=14)
            plt.ylabel(r'$N_d$', fontsize=14)
            plt.savefig(f"./img/{lead_fct}/{geometry}/N_d_angulo_major_{geometry}.png", dpi=300)
            plt.close()

# eof
