#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 22:43:47 2026

@author: nclotta
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np
import pathlib
import natsort

from matplotlib.lines import Line2D

# Datos

err_mqna = 0.001
err_masa = 0.0001

# Setup variable

sample_g = False
major_gr = True

# Setup permanente

lead_factors = ["Fa0", "Fa93", "Fa186"]
filename_default = ["minor_*Hysteresis*", "major_*Segment*"]
geometrias = ["Prisma X", "Prisma X vertical", "Cubo", "Disco", "Disco horizontal", "Polvo"]
colnames = ["Iteration", "Segment", "Field", "Moment", "Time Stamp", "Field Status", "Moment Status", "Invalid"]

marker_dsc = {
    "0g":   "^",
    "30g":  "d",
    "60g":  "s",
    "90g":  "P",
    "120g": "p",
    "150g": "*",
    "180g": "h",
}
marker_pmx = {
    "0g":   "^",
    "15g":  "d",
    "30g":  "s",
    "45g":  "P",
    "60g":  "p",
    "75g":  "*",
    "90g":  "h",
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
colores_lf = {
    "Fa0":   colores_verde,
    "Fa93":  colores_azules,
    "Fa186": colores_violeta,
}

# gramos
masas = {
    "Fa0":   {"dsc": 0.0075, "pmx": 0.0348, "plv": 0.0032, "cbo": 0.0057},
    "Fa93":  {"dsc": 0.0042, "pmx": 0.0322, "plv": 0.0032, "cbo": 0.0048},  # plv: COMPLETAR
    "Fa186": {"dsc": 0.0046, "pmx": 0.0297, "plv": 0.0032, "cbo": 0.0076},  # plv: COMPLETAR
}

masas = {
    "Fa0":   {"dsc": 0.0073, "pmx": 0.0348, "plv": 0.0032, "cbo": 0.0057},
    "Fa93":  {"dsc": 0.0046, "pmx": 0.0322, "plv": 0.0032, "cbo": 0.0048},  # plv: COMPLETAR
    "Fa186": {"dsc": 0.0048, "pmx": 0.0297, "plv": 0.0032, "cbo": 0.0076},  # plv: COMPLETAR
}

masas = {
    "Fa0":   {"dsc": 0.0073, "pmx": 0.0348, "plv": 0.0032, "cbo": 0.0057},
    "Fa93":  {"dsc": 0.0046, "pmx": 0.0322, "plv": 0.0032, "cbo": 0.0048},  # plv: COMPLETAR
    "Fa186": {"dsc": 0.0050, "pmx": 0.0406, "plv": 0.0032, "cbo": 0.0076},  # plv: COMPLETAR
}

def etiqueta_lf(lead_fct):
    return lead_fct.replace("Fa", "LF ")

def configurar(geometry, lead_fct):
    mas = masas[lead_fct]
    filename = filename_default
    if geometry == "Disco":
        dict_marker = marker_dsc
        major_head = 49 if lead_fct == "Fa0" else 85
        if lead_fct == "Fa0":
            filename = ["minor_1er_*Hysteresis*.csv", "major_1er_*Hysteresis*.csv"]
        datasets = ["0g", "30g", "60g", "90g", "120g", "150g"]
        masa = mas["dsc"]
    elif geometry == "Disco horizontal":
        dict_marker = marker_dsc
        major_head = 85
        datasets = ["0g", "30g", "60g", "90g", "120g"]
        masa = mas["dsc"]
    elif geometry == "Prisma X":
        dict_marker = marker_pmx
        major_head = 85
        datasets = ["0g", "15g", "30g", "45g", "60g", "75g", "90g"]
        masa = mas["pmx"]
    elif geometry == "Prisma X vertical":
        dict_marker = marker_pmx
        major_head = 85
        datasets = ["0g", "15g", "30g", "45g", "60g", "75g", "90g"]
        masa = mas["pmx"]
    elif geometry == "Polvo":
        dict_marker = marker_dsc
        major_head = 85
        datasets = ["30g", "90g"]
        masa = mas["plv"]
    elif geometry == "Cubo":
        dict_marker = marker_pmx
        major_head = 85
        datasets = ["0g", "15g", "30g", "45g", "60g"] #, "75g"]
        masa = mas["cbo"]
    return major_head, filename, datasets, dict_marker, masa

def magnetizacion_archivo(q, lead_fct, geometry, dataset, masa, major_head, datosout, major_loop=False):
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

        if sample_g or (major_gr and major_loop):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

        M_s = np.nan
        if major_loop:
            H_sat_indices = np.where(H_raw > 0.93 * np.max(H_raw))
            _, cruce = np.polyfit(H_raw[H_sat_indices], M_raw[H_sat_indices], 1)
            M_s = abs(cruce)
            if ((dataset != "0g" and "Disco" in geometry)
                    or "Cubo" in geometry
                    or "Prisma X" in geometry
                    or "horizontal" in geometry):
                M_s = M_raw[idx_max_after_min - 1]
            if sample_g or (major_gr and major_loop):
                plt.scatter(np.max(H_raw), M_s, marker='d', color='g',
                            zorder=5, label=rf'$M_s$ ({M_s:.3f} [emu/g])')
        if sample_g or (major_gr and major_loop):
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
            tipo = "major" if major_loop else "minor"
            plt.savefig(f"./img/{lead_fct}/{geometry}/{tipo}/{dataset}_{int(df["Field"].max())}_Oe.png",
                            dpi=300)
            plt.close()

        datosout.append({
            "Dataset": dataset,
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
        return H_raw, M_raw, H_raw_err, M_raw_err

def graficar_superpuesto(geometry, resultados, columna, ylabel, nombre):
    fig, ax = plt.subplots(figsize=(8, 6))
    angulos_presentes = {}

    for lead_fct, (minor_loops_df, dict_marker) in resultados.items():
        colores = colores_lf[lead_fct]
        for dataset, group in sorted(minor_loops_df.groupby("Dataset"),
                                     key=lambda kv: int(kv[0].rstrip("g"))):
            marker = dict_marker.get(dataset)
            ax.plot(np.log10(group["H_max"]), group[columna], linestyle='-',
                    marker=marker, markersize=8, markeredgewidth=1.5,
                    color=colores.get(marker))
            angulos_presentes[dataset] = marker

    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel(r'$\log H_{max}$ [Oe]', fontsize=14)
    ax.tick_params(labelsize=14)
    ax.grid()

    handles_lf = [Line2D([], [], color=colores_lf[lf]["P"], linewidth=3, label=etiqueta_lf(lf))
                  for lf in resultados]
    handles_ang = [Line2D([], [], color="0.35", marker=mk, markersize=8, markeredgewidth=1.5,
                          linestyle="-", label=ds.replace("g", "°"))
                   for ds, mk in sorted(angulos_presentes.items(), key=lambda kv: int(kv[0].rstrip("g")))]
    leg_lf = ax.legend(handles=handles_lf, title="Lead factor", fontsize=12, title_fontsize=12,
                       loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.add_artist(leg_lf)
    ax.legend(handles=handles_ang, title="Ángulo", fontsize=12, title_fontsize=12,
              loc="upper left", bbox_to_anchor=(1.02, 0.55))

    carpeta = pathlib.Path(f"./img/{geometry}")
    carpeta.mkdir(parents=True, exist_ok=True)
    fig.savefig(carpeta / f"{nombre}_segun_H_a_{geometry}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def graficar_M_raw_por_angulo(geometry, lead_factors, columnas=3):
    datasets_totales = set()
    for lead_fct in lead_factors:
        _, _, datasets, _, _ = configurar(geometry, lead_fct)
        datasets_totales.update(datasets)
    datasets_ordenados = sorted(datasets_totales, key=lambda x: int(x.rstrip("g")))
    curvas = {}
    for dataset in datasets_ordenados:
        for lead_fct in lead_factors:
            major_head, filename, datasets, dict_marker, masa = configurar(geometry, lead_fct)
            if dataset not in datasets:
                continue
            d = pathlib.Path(f"../data/mediciones/{lead_fct}/{geometry}/{dataset}/")
            if not d.exists():
                continue
            for q in natsort.natsorted(d.glob(f"{filename[1]}")):
                df = pd.read_csv(q, header=major_head, names=colnames)
                curvas.setdefault(dataset, []).append(
                    (lead_fct, df["Field"].values, df["Moment"].values / masa))

    if not curvas:
        return

    n = len(curvas)
    ncols = min(columnas, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    lf_presentes = []
    for idx, (ax, (dataset, lista)) in enumerate(zip(axes_flat, curvas.items())):
        for lead_fct, H_raw, M_raw in lista:
            ax.plot(H_raw, M_raw, linestyle='-', linewidth=1.5,
                    color=colores_lf[lead_fct]["P"], alpha=0.9)
            if lead_fct not in lf_presentes:
                lf_presentes.append(lead_fct)
        ax.set_title(rf"Ángulo: {dataset.replace('g', '°')}", fontsize=13)
        ax.tick_params(labelsize=11)
        ax.grid(True, linestyle='--', alpha=0.6)
        if idx % ncols == 0:
            ax.set_ylabel(r"$M_{raw}$ [emu/g]", fontsize=13)
        if idx + ncols >= n:
            ax.set_xlabel(r"$H_{raw}$ [Oe]", fontsize=13)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    handles_lf = [Line2D([], [], color=colores_lf[lf]["P"], linewidth=3, label=etiqueta_lf(lf))
                  for lf in lead_factors if lf in lf_presentes]
    fig.legend(handles=handles_lf, title="Lead factor", fontsize=12, title_fontsize=12,
               loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles_lf))
    fig.tight_layout()

    carpeta = pathlib.Path(f"./img/{geometry}")
    carpeta.mkdir(parents=True, exist_ok=True)
    fig.savefig(carpeta / f"M_raw_H_raw_{geometry}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def graficar_Ms_angulo(geometry, resultados):
    fig, ax = plt.subplots(figsize=(8, 6))
    graficados = []
    angulos_todos = set()

    for lead_fct, (df_lf, dict_marker) in resultados.items():
        colores = colores_lf[lead_fct]
        mayores = df_lf.dropna(subset=["M_s"]).copy()
        if mayores.empty:
            continue
        mayores["angulo"] = mayores["Dataset"].str.rstrip("g").astype(int)
        mayores = mayores.sort_values("angulo")

        ax.plot(mayores["angulo"], mayores["M_s"], linestyle="-", color=colores["P"], zorder=1)
        for ang, ms, ds in zip(mayores["angulo"], mayores["M_s"], mayores["Dataset"]):
            marker = dict_marker.get(ds)
            ax.plot(ang, ms, linestyle="none", marker=marker, markersize=8,
                    markeredgewidth=1.5, color=colores.get(marker), zorder=3)
        graficados.append(lead_fct)
        angulos_todos.update(mayores["angulo"])

    if not graficados:
        plt.close(fig)
        return

    ax.set_xticks(sorted(angulos_todos))
    ax.set_xlabel(r"Ángulo [°]", fontsize=14)
    ax.set_ylabel(r"$M_s$ [emu/g]", fontsize=14)
    ax.tick_params(labelsize=14)
    ax.grid()
    handles_lf = [Line2D([], [], color=colores_lf[lf]["P"], linewidth=3, label=etiqueta_lf(lf))
                  for lf in graficados]
    ax.legend(handles=handles_lf, title="Lead factor", fontsize=12, title_fontsize=12,
              loc="upper left", bbox_to_anchor=(1.02, 1.0))

    carpeta = pathlib.Path(f"./img/{geometry}")
    carpeta.mkdir(parents=True, exist_ok=True)
    fig.savefig(carpeta / f"M_s_angulo_{geometry}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

for geometry in geometrias:
    resultados = {}  # lead_fct -> (dataframe, dict_marker)

    for lead_fct in lead_factors:
        major_head, filename, datasets, dict_marker, masa = configurar(geometry, lead_fct)

        datosout = []
        for dataset in datasets:
            d = pathlib.Path(f"../data/mediciones/{lead_fct}/{geometry}/{dataset}/")
            for q in natsort.natsorted(d.glob(f"{filename[0]}")):
                x=magnetizacion_archivo(q, lead_fct, geometry, dataset, masa, major_head, datosout)
                H_raw, M_raw, H_raw_err, M_raw_err = x
            for q in natsort.natsorted(d.glob(f"{filename[1]}")):
                magnetizacion_archivo(q, lead_fct, geometry, dataset, masa, major_head, datosout, major_loop=True)

        if not datosout:
            continue

        dff = pd.DataFrame(datosout)
        dff.to_csv(f"./resultados/magnetizacion_{geometry}_{lead_fct}.csv", index=False)

        resultados[lead_fct] = (dff[dff["Dataset"].isin(datasets)], dict_marker)

    if not resultados:
        continue

    graficar_superpuesto(geometry, resultados, "M_r",   r"$M_r$ [emu/g]",     "M_r")
    graficar_superpuesto(geometry, resultados, "H_c",   r"$H_c$ [Oe]",        "H_c")
    graficar_superpuesto(geometry, resultados, "M_max", r"$M_{max}$ [emu/g]", "M_max")
    graficar_Ms_angulo(geometry, resultados)
    graficar_M_raw_por_angulo(geometry, lead_factors)

# eof
