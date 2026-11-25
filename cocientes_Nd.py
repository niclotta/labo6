#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cocientes N_d(Fa93)/N_d(Fa0) y N_d(Fa186)/N_d(Fa0) por geometria y angulo,
a partir del texto que imprime factor_demagnetizacion. Tambien informa que
datos faltan respecto de lo definido en configuracion.py.

Formato esperado (uno por dataset):

    ========== Polvo Fa0: 90g ==========
    $90^\\circ\\to N_d=(0.86096\\pm0.00014)$

Uso:  python3 cocientes_Nd.py [archivo]      (por defecto Nd.txt)
"""

import re
import sys
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from configuracion import (
    lead_factors, geometrias, configurar,
    estilo_lf, etiqueta_lf, cifras,
    ruta_img, ruta_resultados,
)

# Setup variable

graficos = True
referencia = "Fa0"
archivo_default = "Nd.txt"

# Parseo

RE_CABECERA = re.compile(r"^=+\s*(.+?)\s+(Fa\d+):\s*(\d+)g\s*=+$")
RE_VALOR = re.compile(
    r"^\$\s*(\d+)\s*\^\s*\\circ\s*\\to\s*N_d\s*=\s*\(\s*([-+0-9.eE]+)\s*\\pm\s*([-+0-9.eE]+)\s*\)\s*\$$")

def parsear(ruta):
    filas = []
    pendiente = None   # (geometria, lead_factor, angulo, nro de linea de la cabecera)
    for n, linea in enumerate(pathlib.Path(ruta).read_text().splitlines(), 1):
        linea = linea.strip()
        if not linea:
            continue
        cab = RE_CABECERA.match(linea)
        if cab:
            if pendiente is not None:
                print(f"AVISO linea {pendiente[3]}: cabecera sin valor ({pendiente[0]} {pendiente[1]} {pendiente[2]}g)")
            pendiente = (cab.group(1), cab.group(2), int(cab.group(3)), n)
            continue
        val = RE_VALOR.match(linea)
        if val and pendiente is not None:
            geom, lf, ang, _ = pendiente
            if int(val.group(1)) != ang:
                print(f"AVISO linea {n}: la cabecera dice {ang} y el valor {val.group(1)} ({geom} {lf})")
            filas.append({"geometria": geom, "lead_factor": lf, "angulo": ang,
                          "N_d": float(val.group(2)), "N_d_err": float(val.group(3))})
            pendiente = None
            continue
        print(f"AVISO linea {n}: no reconocida: {linea[:70]}")
    if pendiente is not None:
        print(f"AVISO linea {pendiente[3]}: cabecera sin valor ({pendiente[0]} {pendiente[1]} {pendiente[2]}g)")

    df = pd.DataFrame(filas)
    dup = df.duplicated(["geometria", "lead_factor", "angulo"], keep="last")
    if dup.any():
        print("AVISO: datasets repetidos, se conserva el ultimo:")
        print(df[dup][["geometria", "lead_factor", "angulo"]].to_string(index=False))
        df = df[~dup]
    return df.reset_index(drop=True)

# Que falta

def faltantes(df):
    """Datasets definidos en configuracion.py que no estan en el texto."""
    filas = []
    for geom in geometrias:
        for lf in lead_factors:
            presentes = set(df[(df.geometria == geom) & (df.lead_factor == lf)].angulo)
            for ang in configurar(geom, lf).angulos:
                if ang not in presentes:
                    filas.append({"geometria": geom, "lead_factor": lf, "angulo": ang})
    return pd.DataFrame(filas, columns=["geometria", "lead_factor", "angulo"])

def fuera_de_config(df):
    """Datasets que estan en el texto pero no en configuracion.py."""
    filas = []
    for _, f in df.iterrows():
        if f.geometria not in geometrias or f.lead_factor not in lead_factors \
                or f.angulo not in configurar(f.geometria, f.lead_factor).angulos:
            filas.append(f)
    return pd.DataFrame(filas)

def resumen_faltantes(falt, df):
    print("\n==================== FALTANTES ====================")
    if falt.empty:
        print("No falta nada respecto de configuracion.py.")
        return
    for geom in geometrias:
        for lf in lead_factors:
            f = falt[(falt.geometria == geom) & (falt.lead_factor == lf)]
            if f.empty:
                continue
            esperados = configurar(geom, lf).angulos
            angs = ", ".join(f"{a}°" for a in f.angulo)
            if len(f) == len(esperados):
                print(f"{geom} {lf}: todo ({angs})")
            else:
                print(f"{geom} {lf}: {angs}")

# Cocientes

def cocientes(df):
    ref = (df[df.lead_factor == referencia]
           .rename(columns={"N_d": "N_d_ref", "N_d_err": "N_d_ref_err"})
           [["geometria", "angulo", "N_d_ref", "N_d_ref_err"]])
    otros = df[df.lead_factor != referencia]
    m = otros.merge(ref, on=["geometria", "angulo"], how="left")
    m["cociente"] = m.N_d / m.N_d_ref
    m["cociente_err"] = m.cociente * np.sqrt((m.N_d_err / m.N_d) ** 2 + (m.N_d_ref_err / m.N_d_ref) ** 2)
    m["lf_orden"] = m.lead_factor.map({lf: i for i, lf in enumerate(lead_factors)})
    m["geom_orden"] = m.geometria.map({g: i for i, g in enumerate(geometrias)})
    m = m.sort_values(["geom_orden", "lf_orden", "angulo"]).drop(columns=["lf_orden", "geom_orden"])
    return m.reset_index(drop=True)

def imprimir_cocientes(m):
    print(f"\n==================== COCIENTES N_d(LF) / N_d({referencia}) ====================")
    for geom in geometrias:
        g = m[m.geometria == geom]
        if g.empty:
            continue
        print(f"\n--- {geom} ---")
        for lf in lead_factors:
            if lf == referencia:
                continue
            h = g[g.lead_factor == lf]
            if h.empty:
                continue
            print(f"{etiqueta_lf(lf)} / {etiqueta_lf(referencia)}")
            for _, f in h.iterrows():
                if np.isnan(f.cociente):
                    print(f"  {f.angulo:>4}°   sin dato de {referencia} para dividir")
                else:
                    print(f"  {f.angulo:>4}°   {cifras(f.cociente, f.cociente_err)} ± {f.cociente_err:.2g}"
                          f"    (N_d = {f.N_d:.4g}, ref = {f.N_d_ref:.4g})")
            ok = h.dropna(subset=["cociente"])
            if len(ok) > 1:
                print(f"  media {ok.cociente.mean():.3f}   desvio {ok.cociente.std():.3f}"
                      f"   minimo {ok.cociente.min():.3f}   maximo {ok.cociente.max():.3f}")

def graficar_cocientes(m):
    for geom in geometrias:
        g = m[(m.geometria == geom)].dropna(subset=["cociente"])
        if g.empty:
            continue
        fig, ax = plt.subplots()
        ax.axhline(1.0, color="0.5", linestyle=":", linewidth=1.5)
        for lf in lead_factors:
            if lf == referencia:
                continue
            h = g[g.lead_factor == lf]
            if h.empty:
                continue
            est = estilo_lf[lf]
            ax.errorbar(h.angulo, h.cociente, yerr=h.cociente_err, marker=est["marker"],
                        mfc=est["color"], mec="black", color=est["color"], ecolor=est["color"],
                        linestyle="--", capsize=4, elinewidth=1.5,
                        label=f"{etiqueta_lf(lf)} / {etiqueta_lf(referencia)}")
        ax.set_xlabel(r"Ángulo [°]", fontsize=14)
        ref_tex = etiqueta_lf(referencia).replace(" ", r"\,")
        ax.set_ylabel(rf"$N_d\,/\,N_d^{{\mathrm{{{ref_tex}}}}}$", fontsize=14)
        ax.legend(fontsize=12)
        carpeta = ruta_img / geom
        carpeta.mkdir(parents=True, exist_ok=True)
        fig.savefig(carpeta / f"N_d_cociente_{geom}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

if __name__ == "__main__":
    archivo = sys.argv[1] if len(sys.argv) > 1 else archivo_default
    df = parsear(archivo)
    print(f"{len(df)} datasets leidos de {archivo}")

    extra = fuera_de_config(df)
    if not extra.empty:
        print("\nAVISO: datasets que no estan en configuracion.py (no se marcan como faltantes):")
        print(extra[["geometria", "lead_factor", "angulo"]].to_string(index=False))

    resumen_faltantes(faltantes(df), df)

    m = cocientes(df)
    imprimir_cocientes(m)

    ruta_resultados.mkdir(parents=True, exist_ok=True)
    m.to_csv(ruta_resultados / "cocientes_Nd.csv", index=False)

    if graficos:
        graficar_cocientes(m)
# eof
