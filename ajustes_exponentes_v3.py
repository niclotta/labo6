#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 14:02:44 2026

@author: nclotta
"""

# Time-stamp: </Users/nclotta/Documents/__UBA/__LABO_6_SEIS/codigo/ajustes_exponentes_v3.py, 2026-11-24 Tuesday 10:39:02 nclotta>

import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np

from scipy.optimize import curve_fit

from configuracion import (
    lead_factors, geometrias,
    estilo_lf, etiqueta_lf, cifras,
    configurar, recorte_W_F, angulos_reales,
    ruta_img, ruta_resultados,
)

# Setup variable

debug_graph = False
angle_graph = True
exponentes_graph = False  # tambien graficar n_f, n_r y n_c contra el angulo

def ajuste_ln_magnetizacion(W_ast, M_ast, M_rem, sigma, dataset, title, recorte=None):
    def lineal(a, b, x):
        return a * x + b
    ln_W = np.log(W_ast)
    ln_M = np.log(M_ast / M_rem)
    start, stop = (recorte or {}).get(dataset, (0, len(ln_W)))
    ln_W = ln_W[start:stop]
    ln_M = ln_M[start:stop]
    W_ast_sliced = W_ast.to_numpy(dtype=float)[start:stop]
    sigma_arr = sigma.to_numpy(dtype=float)[start:stop]
    sigma_ln_W = sigma_arr / W_ast_sliced
    popt, pcov = curve_fit(lineal, ln_M, ln_W, sigma=sigma_ln_W, absolute_sigma=True)
    ln_W_0, n = popt
    W_0 = np.exp(ln_W_0)
    ln_W_err, n_err = np.sqrt(np.diag(pcov))
    if debug_graph and title == "$W^0_F$":
        plt.scatter(ln_M, ln_W)
        plt.title(rf"{title}    {dataset.replace('g', '°')}")
        plt.show()
        plt.close()
    return W_0, W_0 * ln_W_err, n, n_err

def graficar_ajuste_angulo(geometry, por_lf, col, col_err, ylabel, nombre):
    """Un resultado del ajuste contra el angulo, con los lead factors superpuestos."""
    fig, ax = plt.subplots()
    hay_datos = False
    for lead_fct, df in por_lf.items():
        df = df.sort_values("angulo")
        est = estilo_lf[lead_fct]
        ax.errorbar(angulos_reales(geometry, lead_fct, df["angulo"]), df[col], yerr=df[col_err], marker=est["marker"],
                    mfc=est["color"], mec="black", color=est["color"], ecolor=est["color"],
                    linestyle="--", capsize=4, elinewidth=1.5, label=etiqueta_lf(lead_fct))
        hay_datos = True
    if not hay_datos:
        plt.close(fig)
        return
    ax.set_xlabel(r"Ángulo [°]", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(fontsize=12)
    carpeta = ruta_img / geometry
    carpeta.mkdir(parents=True, exist_ok=True)
    fig.savefig(carpeta / f"{nombre}_angulo_{geometry}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    resultados_ajuste = {}  # geometry -> {lead_fct: DataFrame}

    for lead_fct in lead_factors:
        for geometry in geometrias:
            datasets = configurar(geometry, lead_fct).datasets
            recorte = recorte_W_F(geometry, lead_fct)
            filas = []
            try:
                dff = pd.read_csv(ruta_resultados / f"magnetizacion_{geometry}_{lead_fct}.csv")
            except FileNotFoundError:
                continue
            minor_loops_df = dff[dff["Dataset"].isin(datasets)]
            for dataset, group in minor_loops_df.groupby("Dataset", sort=False):
                M_s    = group[group["Archivo"].str.contains("major")]['M_s'].values
                M_rem  = group[group["Archivo"].str.contains("major")]['M_r'].values
                M_r_er = group['M_r_err']
                H_c_er = group['H_c_err']
                W_F_er = group['W_F_err']
                W_r_er = group['W_r_err']
                M_max  = group['M_max']
                M_r_st = group['M_r']
                H_c_st = group['H_c']
                W_F_st = group['W_F']
                W_r_st = group['W_r']

                W_F_0, W_F_er, n_f, n_f_er = ajuste_ln_magnetizacion(W_F_st, M_max,  M_s,   W_F_er, dataset, "$W^0_F$", recorte)
                W_r_0, W_r_er, n_r, n_r_er = ajuste_ln_magnetizacion(W_r_st, M_r_st, M_rem, W_r_er, dataset, "$W^0_r$")
                H_c_0, H_0_er, n_c, n_c_er = ajuste_ln_magnetizacion(H_c_st, M_r_st, M_rem, H_c_er, dataset, "$H^0_c$")
                filas.append({
                    "angulo": int(dataset.rstrip("g")),
                    "W_F_0": W_F_0, "W_F_0_err": W_F_er, "n_f": n_f, "n_f_err": n_f_er,
                    "W_r_0": W_r_0, "W_r_0_err": W_r_er, "n_r": n_r, "n_r_err": n_r_er,
                    "H_c_0": H_c_0, "H_c_0_err": H_0_er, "n_c": n_c, "n_c_err": n_c_er,
                })
                print(f"========== {geometry} {lead_fct}: {dataset} ==========")
                print(f"$W^0_F = ({cifras(W_F_0, W_F_er)}\\pm{W_F_er:.3f})$ [ergs/g]")
                print(f"$n_f=({cifras(n_f, n_f_er)}\\pm{n_f_er:.2g})$")
                print(f"$W^0_r = ({cifras(W_r_0, W_r_er)}\\pm{W_r_er:.3f})$ [ergs/g]")
                print(f"$n_r=({cifras(n_r, n_r_er)}\\pm{n_r_er:.2g})$")
                print(f"$H^0_c = ({cifras(H_c_0, H_0_er)}\\pm{H_0_er:.3f})$ [Oe]")
                print(f"$n_c=({cifras(n_c, n_c_er)}\\pm{n_c_er:.2g})$")

            if filas:
                resultados_ajuste.setdefault(geometry, {})[lead_fct] = pd.DataFrame(filas)

    if debug_graph or angle_graph:
        for geometry, por_lf in resultados_ajuste.items():
            graficar_ajuste_angulo(geometry, por_lf, "W_F_0", "W_F_0_err", r"$W^0_F$ [erg/g]", "W0_F")
            graficar_ajuste_angulo(geometry, por_lf, "W_r_0", "W_r_0_err", r"$W^0_r$ [erg/g]", "W0_r")
            graficar_ajuste_angulo(geometry, por_lf, "H_c_0", "H_c_0_err", r"$H^0_c$ [Oe]",    "H0_c")
            if exponentes_graph:
                graficar_ajuste_angulo(geometry, por_lf, "n_f", "n_f_err", r"$n_F$", "n_F")
                graficar_ajuste_angulo(geometry, por_lf, "n_r", "n_r_err", r"$n_R$", "n_R")
                graficar_ajuste_angulo(geometry, por_lf, "n_c", "n_c_err", r"$n_c$", "n_c")

# eof
