#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 14:02:44 2026

@author: nclotta
"""

# Time-stamp: </Users/nclotta/Documents/__UBA/__LABO_6_SEIS/codigo/ajustes_exponentes_v1.py, 2026-09-10 Thursday 15:54:05 nclotta>

import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np
import math

from scipy.optimize import curve_fit

# Setup variable

lead_fct = "Fa0"
debug_graph = False
angle_graph = False

# Setup permanente

geometrias = ["Prisma X", "Disco", "Prisma X vertical"]
recorte_dsc = {
    "0g_$W^0_F$":   [8, 21],
    "30g_$W^0_F$":  [8, 25],
    "60g_$W^0_F$":  [10,18],
    "90g_$W^0_F$":  [22,30],
    "120g_$W^0_F$": [8, 33],
    "150g_$W^0_F$": [7, 24]
}
recorte_pmx = {
    "0g_$W^0_F$":   [8, 22],
    "15g_$W^0_F$":  [8, 22],
    "30g_$W^0_F$":  [8, 22],
    "45g_$W^0_F$":  [8, 24],
    "60g_$W^0_F$":  [8, 23],
    "75g_$W^0_F$":  [12,30],
    "90g_$W^0_F$":  [13,34]
}
recorte_pmV = {
#    "0g_$W^0_F$":   [8, 22],
#    "15g_$W^0_F$":  [8, 22],
#    "30g_$W^0_F$":  [8, 22],
#    "45g_$W^0_F$":  [8, 24],
#    "60g_$W^0_F$":  [8, 23],
#    "75g_$W^0_F$":  [12,30],
#    "90g_$W^0_F$":  [12,30]
}

def cifras(val, err):
    if err == 0:
        decimals = 0
    else:
        exp = math.floor(math.log10(abs(err)))
        decimals = max(0, 1 - exp)
    return f"{val:.{decimals}f}"

def ajuste_ln_magnetizacion(W_ast, M_ast, M_rem, sigma, dataset, title):
    def lineal(a, b, x):
        return a * x + b
    ln_W = np.log(W_ast)
    ln_M = np.log(M_ast / M_rem)
    start, stop = dict_recorte.get(f"{dataset}_{title}", [0,len(ln_W)]) #indices_rango_lineal(ln_M, ln_W)
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
        plt.scatter(ln_W, ln_M)
        plt.title(rf"{title}    {dataset.replace('g', '°')}")
        plt.show()
        plt.close()
    return W_0, W_0 * ln_W_err, n, n_err

if __name__ == "__main__":
    for geometry in geometrias:
        if geometry == "Disco":
            datasets = ["0g", "30g", "60g", "90g", "120g", "150g"]
            angulos  = [0, 30, 60, 90, 120, 150]
            dict_recorte = recorte_dsc
            conj_med = "27_08"
        elif geometry == "Prisma X":
            datasets = ["0g", "15g", "30g", "45g", "60g", "75g", "90g"]
            angulos  = [0, 15, 30, 45, 60, 75, 90]
            dict_recorte = recorte_pmx
            conj_med = "03_09"
        elif geometry == "Prisma X vertical":
            datasets = ["0g", "15g", "30g", "45g", "60g", "90g"] # "75g",
            angulos  = [0, 15, 30, 45, 60, 90] # 75,
            dict_recorte = recorte_pmV
            conj_med = "08_09"
        H_c_0_arr = []
        dff = pd.read_csv(f"./resultados/magnetizacion_{geometry}_{lead_fct}_{conj_med}.csv")
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

            W_F_0, W_F_er, n_f, n_f_er = ajuste_ln_magnetizacion(W_F_st, M_max,  M_s,   W_F_er, dataset, "$W^0_F$")
            W_r_0, W_r_er, n_r, n_r_er = ajuste_ln_magnetizacion(W_r_st, M_r_st, M_rem, W_r_er, dataset, "$W^0_r$")
            H_c_0, H_0_er, n_c, n_c_er = ajuste_ln_magnetizacion(H_c_st, M_r_st, M_rem, H_c_er, dataset, "$H^0_c$")
            H_c_0_arr.append(H_c_0)
            print(f"========== {geometry}: {dataset} ==========")
            print(f"$W^0_F = ({cifras(W_F_0, W_F_er)}\\pm{W_F_er:.3f})$ [ergs/g]")
            print(f"$n_f=({cifras(n_f, n_f_er)}\\pm{n_f_er:.2g})$")
            print(f"$W^0_r = ({cifras(W_r_0, W_r_er)}\\pm{W_r_er:.3f})$ [ergs/g]")
            print(f"$n_r=({cifras(n_r, n_r_er)}\\pm{n_r_er:.2g})$")
            print(f"$H^0_c = ({cifras(H_c_0, H_0_er)}\\pm{H_0_er:.3f})$ [Oe]")
            print(f"$n_c=({cifras(n_c, n_c_er)}\\pm{n_c_er:.2g})$")

        if debug_graph or angle_graph:
            plt.plot(angulos, H_c_0_arr, marker="s", linestyle='--')
            plt.title(geometry)
            plt.xlabel(r"Angulo [°]", fontsize=14)
            plt.ylabel(r'$H^0_c$ [Oe]', fontsize=14)
            plt.show()
            plt.close()
# eof
