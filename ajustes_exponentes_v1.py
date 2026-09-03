#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 14:02:44 2026

@author: nclotta
"""

# Time-stamp: </Users/nclotta/Documents/__UBA/__LABO_6_SEIS/codigo/ajustes_exponentes_v1.py, 2026-09-01 Tuesday 16:51:57 nclotta>

import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np
import pathlib
import natsort

from scipy.optimize import curve_fit
from scipy.stats    import linregress

conj_med = "27_08"
lead_fct = "Fa0"
geometry = "Disco"
datasets = ["0g", "30g", "60g", "90g", "120g", "150g"]
angulos  = [0, 30, 60, 90, 120, 150]

# Dataset, Archivo, H_max, M_r, H_c, H_c_err, M_s, W_F, W_r

debug_graph  = False
dict_recorte = {
    "0g_$W^0_F$":   [8, 21],
    "30g_$W^0_F$":  [8, 25],
    "60g_$W^0_F$":  [10,18],
    "90g_$W^0_F$":  [22,30],
    "120g_$W^0_F$": [8, 33],
    "150g_$W^0_F$": [7, 24]
}

def ajuste_ln_magnetizacion(W_ast, M_ast, M_rem, sigma, dataset, title):
    def lineal(a, b, x):
        return a * x + b
    ln_W = np.log(W_ast)
    ln_M = np.log(M_ast / M_rem)
    start, stop = dict_recorte.get(f"{dataset}_{title}", [0,len(ln_W)]) #indices_rango_lineal(ln_M, ln_W)
    ln_W = ln_W[start:stop]
    ln_M = ln_M[start:stop]
    popt, pcov = curve_fit(lineal, ln_M, ln_W)
    ln_W_0, n = popt
    W_0 = np.exp(ln_W_0)
    ln_W_err, n_err = np.sqrt(np.diag(pcov))
    if debug_graph:
        plt.scatter(ln_W, ln_M)
        plt.title(rf"{title}")
        plt.show()
        plt.close()
    return W_0, W_0 * ln_W_err, n, n_err

if __name__ == "__main__":
    H_c_0_arr = []
    dff = pd.read_csv(f"./resultados/magnetizacion_{geometry}_{lead_fct}_{conj_med}.csv")
    minor_loops_df = dff[dff["Dataset"].isin(datasets)] #.sort_values(by="Dataset", key=natsort.natsort_keygen())
    for dataset, group in minor_loops_df.groupby("Dataset", sort=False):
        M_s     = group[group["Archivo"].str.contains("major")]['M_s'].values
        M_rem   = group[group["Archivo"].str.contains("major")]['M_r'].values
        M_max   = group['M_max']
        H_c_er = group['H_c_err']
        M_r_st = group['M_r']
        H_c_st = group['H_c']
        W_F_st = group['W_F']
        W_r_st = group['W_r']

        W_F_0, W_F_err, n_f, n_f_err = ajuste_ln_magnetizacion(W_F_st, M_max, M_s,     dataset, "$W^0_F$")
        W_r_0, W_r_err, n_r, n_r_err = ajuste_ln_magnetizacion(W_r_st, M_r_st, M_rem, dataset, "$W^0_r$")
        H_c_0, H_0_err, n_c, n_c_err = ajuste_ln_magnetizacion(H_c_st, M_r_st, M_rem, H_c_er, dataset, "$H^0_c$")
        H_c_0_arr.append(H_c_0)
        print(f"========== {dataset} ==========")
        print(f"{dataset}: $W^0_F = ({W_F_0:.3f}\\pm{W_F_err:.3f})$ [ergs/g], $n_f=({n_f:.3f}\\pm{n_f_err:.3f})$")
        print(f"{dataset}: $W^0_r = ({W_r_0:.3f}\\pm{W_r_err:.3f})$ [ergs/g], $n_r=({n_r:.3f}\\pm{n_r_err:.3f})$")
        print(f"{dataset}: $H^0_c = ({H_c_0:.3f}\\pm{H_0_err:.3f})$ [Oe], $n_c = ({n_c:.3f}\\pm{n_c_err:.3f})$")

    if debug_graph:
        plt.plot(angulos, H_c_0_arr)
        plt.xlabel(r"Grados", fontsize=14)
        plt.ylabel(r'$H^0_c$ [Oe]', fontsize=14)
        plt.show()
        plt.close()
# eof
