#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 11:55:50 2026

@author: nclotta
"""

# Time-stamp: </Users/nclotta/Documents/__UBA/__LABO_6_SEIS/codigo/W_ext_v1.py, 2026-09-17 Thursday 16:08:45 nclotta>

import math
import pathlib
import matplotlib.pyplot as plt
import natsort
import numpy as np
import pandas as pd
from scipy.integrate import quad

def W_ext_integral(H_ext, M, M_sat, N_d):
    lambd = M/M_sat
    W_in, _ = quad(lambda x: (M_sat * (H_ext - N_d * M_sat * x)), 0, lambd, limit=20000)
    return W_in

def calcular(H_ext, M, M_sat, N_d):
    lambd = M/M_sat
    W_in  = W_ext_integral(H_ext, M, M_sat, N_d)
    W_ext = W_in + 0.5 * N_d * M_sat**2 * lambd**2
    return W_ext

def graficar(W_ext_arr, lambd_arr):
    return 0

# eof
