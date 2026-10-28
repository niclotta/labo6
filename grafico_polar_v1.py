#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 00:51:26 2026

@author: nclotta
"""

# Time-stamp: </Users/nclotta/Documents/__UBA/__LABO_6_SEIS/codigo/grafico_polar_v1.py, 2026-10-27 Tuesday 19:18:33 nclotta>

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy  as np
import pathlib
import natsort

marker_dsc = {
    "0g":   "^",
    "30g":  "d",
    "60g":  "s",
    "90g":  "P",
    "120g": "p",
    "150g": "*",
    "180g": "h",
}
colores_verde = {
    "^": "#4dcb77",
    "d": "#37bd63",
    "s": "#30a657",
    "P": "#298e4a",
    "p": "#22763e",
    "*": "#1c6234",
    "h": "#164f29",
}
colores_azules = {
    "^": "#38bdf8",
    "d": "#0ea5e9",
    "s": "#0284c7",
    "P": "#0369a1",
    "p": "#075985",
    "*": "#0c4a6e",
    "h": "#082f49",
}
colores_violeta = {
    "^": "#c084fc",
    "d": "#a855f7",
    "s": "#9333ea",
    "P": "#7e22ce",
    "p": "#6b21a8",
    "*": "#581c87",
    "h": "#3b0764",
}

err_mqna = 0.001
err_masa = 0.0001
    
angulos = np.arange(0, 360, 15)
tita    = np.radians(angulos)

colnames = ["Iteration", "Segment", "Field", "Moment", "Time Stamp", "Field Status", "Moment Status", "Invalid"]

def grafico_polar_conjunto(resultados, archivo):
    estilo = {
        "Fa0":   {"color": colores_verde["s"],   "marker": "^"},
        "Fa93":  {"color": colores_azules["s"],  "marker": "o"},
        "Fa186": {"color": colores_violeta["s"], "marker": "s"},
    }
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    for lead_fct, (ang, M_s, M_s_err) in resultados.items():
        if len(ang) == 0:
            continue
        est = estilo[lead_fct]
        tita = np.radians(ang)
        ax.plot(np.append(tita, tita[0]), np.append(M_s, M_s[0]), color=est["color"],
                linewidth=1.5, linestyle='--', zorder=2)
        ax.errorbar(tita, M_s, yerr=M_s_err, fmt=est["marker"], color=est["color"], markersize=7,
                    markeredgecolor='black', markeredgewidth=0.5, linestyle='None', zorder=3,
                    label=lead_fct.replace("Fa", "LF "))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=14, loc="upper right", bbox_to_anchor=(1.2, 1.1))
    pathlib.Path(archivo).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(archivo, dpi=300, bbox_inches="tight")
    plt.close(fig)

resultados_polar = {}
for lead_fct in ["Fa93", "Fa186"]:
    if lead_fct == "Fa0":
        colores = colores_verde
        masa = 0.0075 # gramos
    elif lead_fct == "Fa93":
        colores = colores_azules
        masa = 0.0046 # gramos
    elif lead_fct == "Fa186":
        colores = colores_violeta
        masa = 0.0050 # gramos

    #hexes = [colores.get(marker) for marker in marker_dsc.values()]
    #cmap  = mcolors.LinearSegmentedColormap.from_list("custom_gradient", hexes)

    M_s_arr = []
    M_s_err = []
    for i in range(len(angulos)):
        d = pathlib.Path(f"../data/mediciones/{lead_fct}/Disco/{angulos[i]}g/")
        for q in natsort.natsorted(d.glob("major_*Segment*")):
            df = pd.read_csv(q, header=85, names=colnames)
            H_raw = df["Field"].values
            M_raw = df["Moment"].values / masa
            H_raw_err = err_mqna
            M_raw_err = np.sqrt((err_mqna**2)/masa**2 + (M_raw * err_masa**2)/masa) # ** 3)
            M_raw_err[np.isnan(M_raw_err)] = 0.125
            idx_min = np.argmin(H_raw)
            idx_max_after_min = idx_min + np.argmax(H_raw[idx_min:])

            H_inc = H_raw[idx_min:idx_max_after_min]
            M_inc = M_raw[idx_min:idx_max_after_min]

            H_dec = H_raw[idx_max_after_min:]
            M_dec = M_raw[idx_max_after_min:]
            M_dec_err = M_raw_err[idx_max_after_min:]

            if len(H_dec) < 5: 
                idx_max_before_min = np.argmax(H_raw[:idx_min])
                H_dec = H_raw[idx_max_before_min:idx_min]
                M_dec = M_raw[idx_max_before_min:idx_min]
                M_dec_err = M_raw_err[idx_max_before_min:idx_min]

            sat_indices = np.where(H_raw > 0.95 * np.max(H_raw))
            p, cov = np.polyfit(H_raw[sat_indices], M_raw[sat_indices], 1, cov=True)
            M_s = abs(p[1])
            M_s_e = np.sqrt(cov[1, 1])

            # revisar si esto hace falta
            if True:
                M_s = M_raw[idx_max_after_min - 1]
                M_s_e = M_raw_err[idx_max_after_min - 1]

            M_s_arr.append(M_s)
            M_s_err.append(M_s_e)

    M_s_arr = np.array(M_s_arr)
    M_s_err = np.array(M_s_err)
    resultados_polar[lead_fct] = (angulos, M_s_arr, M_s_err)
    tita_cerrado = np.append(tita, tita[0])
    M_s_cerrado  = np.append(M_s_arr, M_s_arr[0])
    M_e_cerrado  = np.append(M_s_err, M_s_err[0])

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    ax.plot(tita_cerrado, M_s_cerrado, color=colores['s'], linewidth=1.5, linestyle='--', zorder=2)
    ax.errorbar(tita, M_s_arr, yerr=M_s_err, fmt='o', color=colores['s'],markersize=7,
                    markeredgecolor='black', markeredgewidth=0.5, linestyle='None', zorder=3, label=r'$M_s$')
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.legend(fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"./img/{lead_fct}/Disco/Polar_M_s_Disco_{lead_fct}.png", dpi=300)
    
grafico_polar_conjunto(resultados_polar, "./img/Disco/Polar_M_s_Disco.png")

# eof
