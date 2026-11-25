#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estima el desfase angular de montaje (delta) de cada dataset a partir de N_d(theta).

Modelo: el campo gira en un plano que contiene dos ejes principales del cuerpo,
con factores N0 (eje a theta = 0) y N90 (eje a theta = 90). La pendiente de M(H)
medida a lo largo del campo es  cos^2/N0 + sin^2/N90, de modo que

    N_d(theta) = 1 / ( cos^2(theta - delta)/N0 + sin^2(theta - delta)/N90 )

Si la pieza estuvo montada con un error delta, el angulo real es
theta_real = theta_medido - delta.  Los tres parametros (N0, N90, delta) se
ajustan en escala logaritmica porque N_d recorre mas de una decada.

Solo se ajustan datasets con al menos 5 angulos (con 4 o menos no hay grados
de libertad para juzgar el ajuste).

Uso:  python3 rotacion_Nd.py [archivo]      (por defecto Nd.txt)
Requiere cocientes_Nd.py y configuracion.py en la misma carpeta.
"""

import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from cocientes_Nd import parsear, archivo_default
from configuracion import ruta_resultados

# Setup variable

min_angulos = 5
significancia = 3.0   # |delta| / sigma_delta para declarar una rotacion
mejora_minima = 5.0   # rms sin rotacion / rms con rotacion
anisotropia_minima = 2.0   # N90/N0: con cuerpos casi isotropos delta no esta definido

# Modelo y ajuste

def modelo(theta, N0, N90, delta):
    t = np.radians(theta - delta)
    return 1.0 / (np.cos(t) ** 2 / N0 + np.sin(t) ** 2 / N90)

def ajustar(theta, N, con_rotacion=True):
    """Devuelve (N0, N90, delta, sigma_delta, rms_log). sigma_delta es NaN sin grados de libertad."""
    theta = np.asarray(theta, float)
    logN = np.log(np.asarray(N, float))
    n_par = 3 if con_rotacion else 2

    def residuo(p):
        d = p[2] if con_rotacion else 0.0
        return np.log(modelo(theta, p[0], p[1], d)) - logN

    mejor = None
    for d0 in ((-20, -8, 0, 8, 20) if con_rotacion else (0,)):
        p0 = [np.min(N), np.max(N), d0] if con_rotacion else [np.min(N), np.max(N)]
        r = least_squares(residuo, p0, bounds=([1e-6, 1e-6, -90][:n_par], [np.inf, np.inf, 90][:n_par]))
        if mejor is None or r.cost < mejor.cost:
            mejor = r
    p = mejor.x
    gl = len(theta) - n_par
    rms = np.sqrt(2 * mejor.cost / len(theta))
    sigma = np.nan
    if con_rotacion and gl > 0:
        cov = np.linalg.inv(mejor.jac.T @ mejor.jac) * (2 * mejor.cost / gl)
        sigma = np.sqrt(cov[2, 2])
    delta = p[2] if con_rotacion else 0.0
    return p[0], p[1], delta, sigma, rms

def analizar(df):
    filas = []
    for (geom, lf), d in df.groupby(["geometria", "lead_factor"], sort=False):
        d = d.sort_values("angulo")
        if len(d) < min_angulos:
            filas.append({"geometria": geom, "lead_factor": lf, "n": len(d)})
            continue
        N0, N90, delta, sig, rms = ajustar(d.angulo, d.N_d)
        _, _, _, _, rms0 = ajustar(d.angulo, d.N_d, con_rotacion=False)
        filas.append({"geometria": geom, "lead_factor": lf, "n": len(d),
                      "N0": N0, "N90": N90, "delta": delta, "delta_err": sig,
                      "rms_log": rms, "rms_log_sin_rot": rms0,
                      "rotado": bool(abs(delta) > significancia * sig and rms0 > mejora_minima * rms
                                     and max(N90, N0) / min(N90, N0) > anisotropia_minima)})
    return pd.DataFrame(filas)

# Salida

def imprimir(res):
    for _, r in res.iterrows():
        et = f"{r.geometria} {r.lead_factor}"
        if "N0" not in r or pd.isna(r.get("N0")):
            print(f"{et:<24} {int(r.n)} angulos: insuficientes para ajustar")
            continue
        marca = "  <-- ROTADO" if r.rotado else ""
        print(f"{et:<24} N0={r.N0:7.3f}  N90={r.N90:7.2f}  "
              f"delta={r.delta:6.2f} +- {r.delta_err:4.2f} deg  "
              f"rms(log): {r.rms_log_sin_rot:.3f} -> {r.rms_log:.3f}{marca}")

if __name__ == "__main__":
    archivo = sys.argv[1] if len(sys.argv) > 1 else archivo_default
    df = parsear(archivo)
    res = analizar(df)
    imprimir(res)
    ruta_resultados.mkdir(parents=True, exist_ok=True)
    res.to_csv(ruta_resultados / "rotacion_Nd.csv", index=False)
    print(f"\nGuardado en {ruta_resultados / 'rotacion_Nd.csv'}")
