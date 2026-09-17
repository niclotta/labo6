#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 22:43:47 2026

@author: nclotta
"""

# Time-stamp: </Users/nclotta/Documents/__UBA/__LABO_6_SEIS/codigo/magnetizacion_v1.py, 2026-09-17 Thursday 16:50:50 nclotta>

import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np
import pathlib
import natsort

# Datos

masa_dsc = 0.0075 # gramos
masa_pmx = 0.0348 # gramos

err_mqna = 0.001
err_masa = 0.0001

# Setup variable

lead_fct = "Fa0"
sample_g = True

# Setup permanente

geometrias = ["Prisma X", "Prisma X vertical", "Disco", "Disco horizontal"]
colnames = ["Iteration", "Segment", "Field", "Moment", "Time Stamp", "Field Status", "Moment Status", "Invalid"]
marker_dsc = {
    "0g":   "^",
    "30g":  "d",
    "60g":  "s",
    "90g":  "P",
    "120g": "p",
    "150g": "*"
}
marker_pmx = {
    "0g":   "^",
    "15g":  "d",
    "30g":  "s",
    "45g":  "P",
    "60g":  "p",
    "75g":  "*",
    "90g":  "h"
}
colores = {
    "^": "#4dcb77",
    "d": "#37bd63",
    "s": "#30a657",
    "P": "#298e4a",
    "p": "#22763e",
    "*": "#1c6234",
    "h": "#164f29",
}   

def magnetizacion_archivo(q, geometry, major_loop=False):
        df = pd.read_csv(q, header=major_head if major_loop else 49, names=colnames)
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

        if len(H_dec) < 5: 
            idx_max_before_min = np.argmax(H_raw[:idx_min])
            H_dec = H_raw[idx_max_before_min:idx_min]
            M_dec = M_raw[idx_max_before_min:idx_min]

        M_r_dec = np.interp(0.0, np.flip(H_dec), np.flip(M_dec))
        M_r_inc = np.interp(0.0, H_inc, M_inc)
        M_r = (abs(M_r_dec) + abs(M_r_inc)) / 2
        M_r_err = abs((M_r - abs(M_r_dec))/2)

        sort_dec = np.argsort(M_dec)
        H_c_dec = np.interp(0.0, M_dec[sort_dec], H_dec[sort_dec])        
        sort_inc = np.argsort(M_inc)
        H_c_inc = np.interp(0.0, M_inc[sort_inc], H_inc[sort_inc])
        H_c = (abs(H_c_dec) + abs(H_c_inc)) / 2
        H_c_err = abs((H_c - abs(H_c_dec))/2)

        W_F = 0.5 * np.abs(np.dot(H_raw, np.roll(M_raw, 1)) - np.dot(M_raw, np.roll(H_raw, 1)))
        W_F_err = 0.5 * np.sqrt(np.sum(
                ((np.roll(M_raw, -1) - np.roll(M_raw, 1)) * H_raw_err) ** 2 +
                ((np.roll(H_raw, 1) - np.roll(H_raw, -1)) * M_raw_err) ** 2))
        W_r = 0.5 * H_c * M_r
        W_r_err = np.sqrt(0.25 * (H_c**2 * M_r_err**2 + M_r**2 * H_c_err**2))

        if sample_g:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

        M_s = np.nan
        if major_loop:
            H_sat_indices = np.where(H_raw > 0.95 * np.max(H_raw))
            _, cruce = np.polyfit(H_raw[H_sat_indices], M_raw[H_sat_indices], 1)
            M_s = abs(cruce)
            if datasets[i] != "0g":
                M_s = M_raw[idx_max_after_min - 1]
            if sample_g:
                plt.scatter(np.max(H_raw), M_s, marker='d', color='g',
                            zorder=5, label=rf'$M_s$ ({M_s:.3f} [emu/g])')
        if sample_g:
            ax1.plot(H_raw, M_raw)
            ax1.scatter([0, 0], [M_r_dec, M_r_inc], marker='d', color='r',
                        zorder=5, label=rf'$M_r$ ({M_r:.3f} [emu/g])')
            ax1.scatter([H_c_dec, H_c_inc], [0, 0], marker='d', color='b',
                        zorder=5, label=rf'$H_c$ ({H_c:.1f} [Oe])')
            ax1.set_xlabel(r'$H_a$ [Oe]', fontsize=14)
            ax1.set_ylabel(r"$M_r$ [emu/g]", fontsize=14)
            ax1.set_xlim([-2 * H_c, H_c * 2])
            ax1.set_ylim([-2 * M_r, M_r * 2])
            ax1.legend(fontsize=14)
            ax1.grid()
            plt.plot(H_raw, M_raw)
            plt.scatter([0, 0], [M_r_dec, M_r_inc], marker='d', color='r',
                        zorder=5) #, label=rf'$M_r$ ({M_r:.3f} [emu/g])')
            plt.scatter([H_c_dec, H_c_inc], [0, 0], marker='d', color='b',
                        zorder=5) #, label=rf'$H_c$ ({H_c:.1f} [Oe])')
            plt.xlabel(r'$H_a$ [Oe]', fontsize=14)
            plt.ylabel(r"$M_r$ [emu/g]", fontsize=14)
            if major_loop:
                plt.legend(fontsize=14)
            plt.grid()
            plt.tight_layout()
            filename = "major" if major_loop else "minor"
            plt.savefig(f"./img/{lead_fct}/{geometry}/{filename}_{datasets[i]}_{int(df["Field"].max())}_Oe.png",
                            dpi=300)
            plt.close()

        datosout.append({
            "Dataset": datasets[i],
            "Archivo": q.name,
            "H_max": np.max(H_raw),
            "M_max": np.max(M_raw),
            "M_s": M_s,
            "M_r": M_r,
            "H_c": H_c,
            "W_F": W_F,
            "W_r": W_r,
            "M_r_err": np.abs(M_r_err),
            "H_c_err": np.abs(H_c_err),
            "W_F_err": np.abs(W_F_err),
            "W_r_err": np.abs(W_r_err)
        })

for geometry in geometrias:
    if geometry == "Disco":
        dict_marker = marker_dsc
        major_head = 49
        filename = ["minor_1er_*Hysteresis*.csv", "major_1er_*Hysteresis*.csv"]
        datasets = ["0g", "30g", "60g", "90g", "120g", "150g"]
        conj_med = "27_08"
        masa = masa_dsc
    if geometry == "Disco horizontal":
        major_head = 85
        filename = ["minor_*Hysteresis*", "major_*Segment*"]
        datasets = ["0g", "30g", "60g", "90g", "120g"]
        conj_med = "17_09"
        masa = masa_dsc
    elif geometry == "Prisma X":
        dict_marker = marker_pmx
        major_head = 85
        filename = ["minor_*Hysteresis*", "major_*Segment*"]
        datasets = ["0g", "15g", "30g", "45g", "60g", "75g", "90g"]
        conj_med = "03_09"
        masa = masa_pmx
    elif geometry == "Prisma X vertical":
        dict_marker = marker_pmx
        major_head = 85
        filename = ["minor_*Hysteresis*", "major_*Segment*"]
        datasets = ["0g", "15g", "30g", "45g", "60g", "90g"] # "75g",
        conj_med = "08_09"
        masa = masa_pmx
    datosout = []
    for i in range(len(datasets)):                     # /{conj_med}
        d = pathlib.Path(f"../data/mediciones/{lead_fct}/{geometry}/{datasets[i]}/")
        for q in natsort.natsorted(d.glob(f"{filename[0]}")):
            magnetizacion_archivo(q, geometry)
        for q in natsort.natsorted(d.glob(f"{filename[1]}")):
            magnetizacion_archivo(q, geometry, major_loop=True)

    dff = pd.DataFrame(datosout)
    dff.to_csv(f"./resultados/magnetizacion_{geometry}_{lead_fct}_{conj_med}.csv", index=False)

    minor_loops_df = dff[dff["Dataset"].isin(datasets)]

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    for dataset, group in minor_loops_df.groupby("Dataset"):
        ax1.plot(np.log(group["H_max"])/np.log(10), group["M_r"], linestyle='-',
                     marker=dict_marker.get(dataset), markersize=8, markeredgewidth=1.5,
                     color=colores.get(dict_marker.get(dataset)), label=f'{dataset.replace("g", "°")}')
    ax1.set_ylabel(r"$M_r$ [emu/g]", fontsize=14)
    ax1.set_xlabel(r'$\log H_{max}$ [Oe]', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax1.legend(fontsize=14)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"./img/{lead_fct}/{geometry}/M_r_segun_H_a_{geometry}_{lead_fct}_{conj_med}.png", dpi=300)
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for dataset, group in minor_loops_df.groupby("Dataset"):
        ax2.plot(np.log(group["H_max"])/np.log(10), group["H_c"], linestyle='-',
                     marker=dict_marker.get(dataset), markersize=8, markeredgewidth=1.5,
                     color=colores.get(dict_marker.get(dataset)), label=f'{dataset.replace("g", "°")}')

    ax2.set_ylabel(r"$H_c$ [Oe]", fontsize=14)
    ax2.set_xlabel(r'$\log H_{max}$ [Oe]', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax2.legend(fontsize=14)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"./img/{lead_fct}/{geometry}/H_c_segun_H_a_{geometry}_{lead_fct}_{conj_med}.png", dpi=300)
    plt.close()

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    for dataset, group in minor_loops_df.groupby("Dataset"):
        ax3.plot(np.log(group["H_max"])/np.log(10), group["M_max"], linestyle='-',
                     marker=dict_marker.get(dataset), markersize=8, markeredgewidth=1.5,
                     color=colores.get(dict_marker.get(dataset)), label=f'{dataset.replace("g", "°")}')

    ax3.set_ylabel(r"$M_{max}$ [emu/g]", fontsize=14)
    ax3.set_xlabel(r'$\log H_{max}$ [Oe]', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax3.legend(fontsize=14)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"./img/{lead_fct}/{geometry}/M_max_segun_H_a_{geometry}_{lead_fct}_{conj_med}.png", dpi=300)
    plt.close()

# eof
