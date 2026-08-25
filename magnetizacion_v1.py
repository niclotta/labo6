#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 22:43:47 2026

@author: nclotta
"""

# Time-stamp: </Users/nclotta/Documents/__UBA/__LABO_6_SEIS/codigo/magnetizacion_v1.py, 2026-08-24 Monday 23:36:00 nclotta>

import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np
import pathlib
import natsort

masa = 0.0075 # gramos
datasets = ["0g", "90g", "major"]
colnames = ["Iteration", "Segment", "Field", "Moment", "Time Stamp", "Field Status", "Moment Status", "Invalid"]
filename = "minor_loops"
datos_out = []

for i in range(len(datasets)):
    d = pathlib.Path(f"../data/mediciones/Fa0/Disco/{datasets[i]}/")
    if datasets[i] == "major":
        filename = "MajorLoops"
    for q in natsort.natsorted(d.glob(f"{filename}_*Hysteresis*.csv")):
        df = pd.read_csv(q, header=49, names=colnames)
        H_raw = df["Field"].values
        M_raw = df["Moment"].values / masa
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

        sort_dec = np.argsort(M_dec)
        H_c_dec = np.interp(0.0, M_dec[sort_dec], H_dec[sort_dec])        
        sort_inc = np.argsort(M_inc)
        H_c_inc = np.interp(0.0, M_inc[sort_inc], H_inc[sort_inc])
        H_c = (abs(H_c_dec) + abs(H_c_inc)) / 2

        M_s = np.nan
        if datasets[i] == "major":
            H_sat_indices = np.where(H_raw > 0.95 * np.max(H_raw))
            _, cruce = np.polyfit(H_raw[H_sat_indices], M_raw[H_sat_indices], 1)
            M_s = abs(cruce)

        fig, ax = plt.subplots(figsize=(12, 8))
        plt.plot(H_raw, M_raw)
        plt.scatter([0, 0], [M_r_dec, M_r_inc], color='r', zorder=5, label=rf'$M_r$ ({M_r:.3f} [emu/g])')
        plt.scatter([H_c_dec, H_c_inc], [0, 0], color='b', zorder=5, label=rf'$H_c$ ({H_c:.1f} [Oe])')
        plt.xlabel(r'$H_a$ [Oe]', fontsize=14)
        plt.ylabel(r"$M_r$ [emu/g]", fontsize=14)
        plt.legend(fontsize=14)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"./img/{filename}_{datasets[i]}_{int(df["Field"].max())}_Oe.png", dpi=300)
        plt.close()

        datos_out.append({
            "Dataset": datasets[i],
            "Filename": q.name,
            "H_max": np.max(H_raw),
            "M_r": M_r,
            "H_c": H_c,
            "M_s": M_s
        })

dff = pd.DataFrame(datos_out)
dff.to_csv("./resultados/magnetizacion_20_08.csv", index=False)

minor_loops_df = dff[dff["Dataset"].isin(["0g", "90g"])]

fig1, ax1 = plt.subplots(figsize=(8, 6))
for name, group in minor_loops_df.groupby("Dataset"):
    ax1.plot(group["H_max"], group["M_r"], 's-', label=f'{name.replace("g", "°")}')

ax1.set_ylabel(r"$M_r$ [emu/g]", fontsize=14)
ax1.set_xlabel(r'$H_a$ [Oe]', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax1.legend(fontsize=14)
plt.grid()
plt.tight_layout()
plt.savefig("./img/M_r_segun_H_a_20_08.png", dpi=300)
plt.close()

fig2, ax2 = plt.subplots(figsize=(8, 6))
for name, group in minor_loops_df.groupby("Dataset"):
    ax2.plot(group["H_max"], group["H_c"], 's-', label=f'{name.replace("g", "°")}')

ax2.set_ylabel(r"$H_c$ [Oe]", fontsize=14)
ax2.set_xlabel(r'$H_a$ [Oe]', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax2.legend(fontsize=14)
plt.grid()
plt.tight_layout()
plt.savefig("./img/H_c_segun_H_a_20_08.png", dpi=300)
plt.close()
