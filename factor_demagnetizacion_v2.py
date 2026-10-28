#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 11:43:02 2026

@author: nclotta
"""

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

# gramos
masas = {
    "Fa0":   {"dsc": 0.0075, "pmx": 0.0348, "plv": 0.0032, "cbo": 0.0057},
    "Fa93":  {"dsc": 0.0042, "pmx": 0.0322, "plv": 0.0032, "cbo": 0.0048},  # plv: COMPLETAR
    "Fa186": {"dsc": 0.0046, "pmx": 0.0297, "plv": 0.0032, "cbo": 0.0076},  # plv: COMPLETAR
}

# gramos
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


# Paletas de magnetizacion: un color por lead factor (el tono "P", el mismo de su leyenda)
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

# Estilo de cada lead factor en los graficos superpuestos
estilo_lf = {
    "Fa0":   {"color": colores_verde["P"],   "marker": "v"},
    "Fa93":  {"color": colores_azules["P"],  "marker": "s"},
    "Fa186": {"color": colores_violeta["P"], "marker": "o"},
}

def etiqueta_lf(lead_fct):
    return lead_fct.replace("Fa", "LF ")

def cifras(val, err):
    if err == 0:
        decimals = 0
    else:
        exp = math.floor(math.log10(abs(err)))
        decimals = max(0, 1 - exp)
    return f"{val:.{decimals}f}"

def configurar(geometry, lead_fct):
    """Devuelve (major_head, filename, datasets, angulos, masa) de una geometria y lead factor."""
    mas = masas[lead_fct]
    if geometry == "Disco":
        major_head = 49 if lead_fct == "Fa0" else 85
        if lead_fct == "Fa0":
            filename = ["minor_1er_*Hysteresis*.csv", "major_1er_*Hysteresis*.csv"]
        else:
            filename = ["minor_*Hysteresis*", "major_*Segment*"]
        datasets = ["0g", "30g", "60g", "90g", "120g", "150g"]
        angulos  = [0, 30, 60, 90, 120, 150]
        masa = mas["dsc"]
    elif geometry == "Disco horizontal":
        major_head = 85
        filename = ["minor_*Hysteresis*", "major_*Segment*"]
        datasets = ["0g", "30g", "60g", "90g", "120g"]
        angulos  = [0, 30, 60, 90, 120]
        masa = mas["dsc"]
    elif geometry == "Prisma X":
        major_head = 85
        filename = ["minor_*Hysteresis*", "major_*Segment*"]
        datasets = ["0g", "15g", "30g", "45g", "60g", "75g", "90g"]
        angulos  = [0, 15, 30, 45, 60, 75, 90]
        masa = mas["pmx"]
    elif geometry == "Prisma X vertical":
        major_head = 85
        filename = ["minor_*Hysteresis*", "major_*Segment*"]
        datasets = ["0g", "15g", "30g", "45g", "60g", "75g", "90g"]
        angulos  = [0, 15, 30, 45, 60, 75, 90]
        masa = mas["pmx"]
    elif geometry == "Polvo":
        major_head = 85
        filename = ["minor_*Hysteresis*", "major_*Segment*"]
        datasets = ["30g", "90g"]
        angulos  = [30, 90]
        masa = mas["plv"]
    elif geometry == "Cubo":
        major_head = 85
        filename = ["minor_*Hysteresis*", "major_*Segment*"]
        datasets = ["0g", "15g", "30g", "45g", "60g"]
        angulos  = [0, 15, 30, 45, 60]
        masa = mas["cbo"]
    return major_head, filename, datasets, angulos, masa

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

for geometry in geometrias:
    # Resultados de esta geometria, uno por lead factor
    N_d_res = {}    # lead_fct -> (angulos medidos, N_d, N_d_err)
    W_ext_res = {}  # dataset -> {lead_fct: (lambda, W_ext)}

    for lead_fct in lead_factors:
        major_head, filename, datasets, angulos, masa = configurar(geometry, lead_fct)

        N_d_ang = []
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
                N_d_ang.append(angulos[i])
                N_d_e.append(N_d_err)
                N_d_a.append(N_d)

                M_pos, H_pos = M_dec[H_dec > 0], H_dec[H_dec > 0]
                target_lambdas = [0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                target_lambdas = [0.01, 0.05, 0.1, 0.18, 0.23, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.88, 0.96, 1.0]
                target_lambdas = [0.02, 0.09, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.88, 0.9, 0.91, 0.92, 0.96, 1.0]
                cut_idx = [np.abs(M_pos/M_s - target).argmin() for target in target_lambdas]
                H_cut, M_cut = H_pos[cut_idx], M_pos[cut_idx]

                W_e_a = []
                W_i_a = []
                for j, M in enumerate(M_cut):
                    W_ext = W_ext_v1.calcular(H_cut[j], M, M_s, N_d)
                    W_in  = W_ext_v1.W_ext_integral(H_cut[j], M, M_s, N_d)
                    W_e_a.append(W_ext)
                    W_i_a.append(W_in)

                W_ext_res.setdefault(datasets[i], {})[lead_fct] = (M_cut/M_s, np.array(W_e_a))

        N_d_res[lead_fct] = (N_d_ang, N_d_a, N_d_e)

    if not graficos:
        continue

    carpeta = pathlib.Path(f"./img/{geometry}")
    carpeta.mkdir(parents=True, exist_ok=True)

    # W_ext vs lambda: una figura por angulo, con los tres lead factors superpuestos
    for dataset, por_lf in W_ext_res.items():
        fig, ax = plt.subplots()
        for lead_fct, (lam, W_e) in por_lf.items():
            est = estilo_lf[lead_fct]
            ax.plot(lam, W_e, marker=est["marker"], mfc=est["color"], mec="black",
                    color=est["color"], linestyle="--", label=etiqueta_lf(lead_fct))
        ax.set_xlabel(r"$\lambda$", fontsize=14)
        ax.set_ylabel(r'$W_{ext}$ [erg/cm$^3$]', fontsize=14)
        ax.legend(fontsize=12)
        fig.savefig(carpeta / f"W_ext_lambda_{dataset}_{geometry}.png", dpi=300)
        plt.close(fig)

    # N_d vs angulo: una figura por geometria, con los tres lead factors superpuestos
    if any(len(ang) > 0 for ang, _, _ in N_d_res.values()):
        fig, ax = plt.subplots()
        for lead_fct, (ang, N_d_a, N_d_e) in N_d_res.items():
            if len(ang) == 0:
                continue
            est = estilo_lf[lead_fct]
            ax.errorbar(ang, N_d_a, yerr=N_d_e, marker=est["marker"], mfc=est["color"], mec="black",
                        color=est["color"], linestyle="--", ecolor=est["color"], capsize=4,
                        elinewidth=1.5, label=etiqueta_lf(lead_fct))
        ax.set_xlabel(r"Ángulo [°]", fontsize=14)
        ax.set_ylabel(r'$N_d$', fontsize=14)
        ax.legend(fontsize=12)
        fig.savefig(carpeta / f"N_d_angulo_major_{geometry}.png", dpi=300)
        plt.close(fig)

# eof
