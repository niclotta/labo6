#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 13:38:46 2026

@author: nclotta
"""

# Time-stamp: </Users/nclotta/Documents/__UBA/__LABO_6_SEIS/codigo/graficos_prelim_1.py, 2026-08-20 Thursday 16:07:20 nclotta>

import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np
import pathlib
import natsort

mass = 0.0075 # gramos
# 0g, 90g, 
datasets = ["0g", "90g", "major"]
#minor_loops_v1_disco Step 2 Hysteresis Measurement
filename = "minor_loops"
#Step,Iteration,Segment,Field [Oe],Moment (m) [emu],Time Stamp [s],Field Status,Moment (m) Status,
colnames = ["Iteration", "Segment", "Field", "Moment", "Time Stamp", "Field Status", "Moment Status", "Invalid"]

for i in range(len(datasets)):
    d = pathlib.Path(f"../data/mediciones/Fa0/Disco/{datasets[i]}/")
    if datasets[i] == "major":
        filename = "MajorLoops"
    for q in natsort.natsorted(d.glob(f"{filename}_*Hysteresis*.csv")):
        df = pd.read_csv(q, header=49, names=colnames)
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.plot(df["Field"], df["Moment"]/mass)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax.set_ylabel(r"$M_r$ [emu/g]", fontsize=14)
        ax.set_xlabel('$H_a$ [Oe]', fontsize=14)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"./img/{filename}_{datasets[i]}_{int(df["Field"].max())}_Oe.png", dpi=300)
        plt.close()



# eof





"""
if not os.path.exists(f"./csv"):
    os.makedirs(f"./csv")
"""
