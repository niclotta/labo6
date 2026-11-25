#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Time-stamp: </Users/nclotta/Documents/__UBA/__LABO_6_SEIS/codigo/configuracion.py, 2026-11-24 Tuesday 11:25:09 nclotta>

import math
import pathlib

import numpy as np

from typing import NamedTuple

# ----------------------------------------------------------------------
# Rutas
# ----------------------------------------------------------------------

ruta_datos = pathlib.Path("../data/mediciones")
ruta_img = pathlib.Path("./img")
ruta_resultados = pathlib.Path("./resultados")

# ----------------------------------------------------------------------
# Datos experimentales
# ----------------------------------------------------------------------

densidad_acero = 7.8   # g/cm3

err_mqna = 0.001       # emu
err_masa = 0.0001      # gramos

# gramos
masas = {
    "Fa0":   {"dsc": 0.0073, "pmx": 0.0348, "plv": 0.0032, "cbo": 0.0057},
    "Fa93":  {"dsc": 0.0046, "pmx": 0.0325, "plv": 0.0033, "cbo": 0.0050},
    "Fa186": {"dsc": 0.0050, "pmx": 0.0406, "plv": 0.0057, "cbo": 0.0078},
}

# ----------------------------------------------------------------------
# Lectura de los archivos del VSM
# ----------------------------------------------------------------------

colnames = ["Iteration", "Segment", "Field", "Moment", "Time Stamp", "Field Status", "Moment Status", "Invalid"]

minor_head = 49          # filas de encabezado de los loops minor
major_head_default = 85  # idem para los major (salvo override mas abajo)

filename_default = ["minor_*Hysteresis*", "major_*Segment*"]

# ----------------------------------------------------------------------
# Analisis de M_s
# ----------------------------------------------------------------------

# Fraccion del campo maximo a partir de la cual se ajusta la recta de saturacion
fraccion_saturacion = 0.93

def ms_campo_maximo(geometry, dataset):
    """
    True: M_s es el valor al campo maximo.
    False: M_s es la ordenada al origen del ajuste lineal en saturacion.
    """
    return ((dataset != "0g" and "Disco" in geometry)
            or "Cubo" in geometry
            or "Prisma X" in geometry
            or "horizontal" in geometry)

# ----------------------------------------------------------------------
# Colores y marcadores
# ----------------------------------------------------------------------

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

# Paleta completa de cada lead factor (marcador -> color)
colores_lf = {
    "Fa0":   colores_verde,
    "Fa93":  colores_azules,
    "Fa186": colores_violeta,
}

# Estilo de cada lead factor en los graficos contra el angulo: tono "P" de su paleta
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

# ----------------------------------------------------------------------
# Que se mide: lead factors, geometrias y angulos
# ----------------------------------------------------------------------

lead_factors = ["Fa0", "Fa93", "Fa186"]

# masa: clave dentro de masas[lead_fct]; marker: dict angulo -> marcador
_geometrias = {
    "Prisma X": {
        "masa": "pmx", "marker": marker_pmx,
        "datasets": ["0g", "15g", "30g", "45g", "60g", "75g", "90g"],
    },
    "Prisma X vertical": {
        "masa": "pmx", "marker": marker_pmx,
        "datasets": ["0g", "15g", "30g", "45g", "60g", "75g", "90g"],
    },
    "Cubo": {
        "masa": "cbo", "marker": marker_pmx,
        "datasets": ["0g", "15g", "30g", "45g", "60g"],
    },
    "Disco": {
        "masa": "dsc", "marker": marker_dsc,
        "datasets": ["0g", "30g", "60g", "90g", "120g", "150g"],
    },
    "Disco horizontal": {
        "masa": "dsc", "marker": marker_dsc,
        "datasets": ["0g", "30g", "60g", "90g", "120g"],
    },
    "Polvo": {
        "masa": "plv", "marker": marker_dsc,
        "datasets": ["30g", "90g"],
    },
}

geometrias = list(_geometrias)

# Casos que se apartan de lo anterior: (geometria, lead factor) -> campos a pisar
_excepciones = {
    ("Disco", "Fa0"): {
        "major_head": 49,
        "filename": ["minor_1er_*Hysteresis*.csv", "major_1er_*Hysteresis*.csv"],
    },
    ("Prisma X vertical", "Fa186"): {
        "datasets": ["0g", "30g", "45g", "60g", "75g", "90g", "120g", "150g"],
    },
}

class Config(NamedTuple):
    major_head: int
    filename: list
    datasets: list
    angulos: list
    dict_marker: dict
    masa: float

def configurar(geometry, lead_fct):
    g = _geometrias[geometry]
    exc = _excepciones.get((geometry, lead_fct), {})
    datasets = exc.get("datasets", g["datasets"])
    return Config(
        major_head=exc.get("major_head", major_head_default),
        filename=exc.get("filename", filename_default),
        datasets=datasets,
        angulos=[int(d.rstrip("g")) for d in datasets],
        dict_marker=g["marker"],
        masa=masas[lead_fct][g["masa"]],
    )

# ----------------------------------------------------------------------
# Recortes del ajuste de W_F: dataset -> (inicio, fin) en indices
# Un dataset que no figure se ajusta completo.
# ----------------------------------------------------------------------

_recortes_base = {
    "Disco": {
        "0g":   (8, 21),
        "30g":  (8, 25),
        "60g":  (10, 18),
        "90g":  (22, 30),
        "120g": (8, 33),
        "150g": (7, 24),
    },
    "Disco horizontal": {
        "0g":   (8, 21),
        "30g":  (8, 21),
        "60g":  (8, 21),
        "90g":  (8, 21),
        "120g": (8, 21),
        "150g": (8, 21),
    },
    "Prisma X": {
        "0g":   (8, 22),
        "15g":  (8, 22),
        "30g":  (8, 22),
        "45g":  (8, 24),
        "60g":  (8, 23),
        "75g":  (12, 30),
        "90g":  (13, 34),
    },
    "Prisma X vertical": {
        "0g":   (8, 20),
        "15g":  (8, 22),
        "30g":  (8, 22),
        "45g":  (8, 24),
        "60g":  (8, 23),
        "75g":  (12, 30),
        "90g":  (12, 30),
    },
    "Polvo": {
        "30g":  (8, 18),
        "90g":  (8, 18),
    },
    "Cubo": {
        "0g":   (8, 20),
        "15g":  (8, 22),
        "30g":  (8, 22),
        "45g":  (8, 24),
        "60g":  (8, 23),
    },
}

# (geometria, lead factor) -> recortes propios, si difieren de los base
_recortes_excepciones = {
    ("Disco", "Fa93"): {
        "0g":   (7, 17),
        "30g":  (7, 19),
        "60g":  (7, 19),
        "90g":  (12, 30),
        "120g": (7, 22),
        "150g": (7, 17),
    },
}

def recorte_W_F(geometry, lead_fct):
    return _recortes_excepciones.get((geometry, lead_fct), _recortes_base.get(geometry, {}))

# ----------------------------------------------------------------------
# Correccion del angulo de montaje
# ----------------------------------------------------------------------

# (geometria, lead factor) -> delta en grados; angulo_real = angulo_medido - delta
correccion_angulo = {("Disco", "Fa93"): 7.6}

def angulos_reales(geometry, lead_fct, angulos):
    return np.asarray(angulos, float) - correccion_angulo.get((geometry, lead_fct), 0.0)

def rotar_Ms(angulos, Ms, Ms_err, delta, periodo=360.0, remuestrear=False):
    """Aplica angulo_real = angulo_medido - delta a un M_s(theta).

    remuestrear=False: solo reetiqueta los angulos. Devuelve (angulo_real, Ms, Ms_err)
                       sin tocar los valores. Es exacto.
    remuestrear=True:  evalua la curva corregida en los angulos originales por
                       interpolacion lineal periodica, con el error propagado.
                       Devuelve (angulos, Ms_nuevo, Ms_err_nuevo).
    periodo: 360 si el barrido cubre toda la vuelta, 180 si M_s(theta) = M_s(theta+180).
    """
    ang = np.asarray(angulos, float)
    ms = np.asarray(Ms, float)
    err = np.asarray(Ms_err, float)
    nuevo = (ang - delta) % periodo
    if not remuestrear:
        return nuevo, ms, err

    o = np.argsort(nuevo)
    x, y, e = nuevo[o], ms[o], err[o]
    objetivo = ang % periodo
    i1 = np.searchsorted(x, objetivo, side="right")
    i0 = (i1 - 1) % len(x)
    i1 = i1 % len(x)
    paso = (x[i1] - x[i0]) % periodo
    w = ((objetivo - x[i0]) % periodo) / np.where(paso == 0, 1.0, paso)
    ms_new = (1 - w) * y[i0] + w * y[i1]
    err_new = np.hypot((1 - w) * e[i0], w * e[i1])
    return ang, ms_new, err_new

# eof
