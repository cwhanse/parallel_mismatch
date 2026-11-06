# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 10:58:15 2026

@author: cliff
"""

# compute triangular distribution parameters for Io from distribution of target Voc degradation rates

import numpy as np
import pvlib
import matplotlib.pyplot as plt


asc = 0.0015
N = 1.3
mu_gamma = 0
il = 13.0
io = 3.9e-10

rsh = 1144
rsh0 = 3850
rs = .6
ns = 72
npts = 100

String_Params = {'alpha_sc': asc,
                 'gamma_ref': N,
                 'mu_gamma': mu_gamma,
                 'I_L_ref': il,
                 'I_o_ref': io,
                 'R_sh_ref': rsh,
                 'R_sh_0': rsh0,
                 'R_s': rs,
                 'cells_in_series': ns,
                 'R_sh_exp': 5.5,
                 'EgRef': 1.12,
                 'irrad_ref':1000,
                 'temp_ref':25,
                 }

base = pvlib.pvsystem.calcparams_pvsyst(1000., 25., **String_Params)

iomult = np.arange(15., 1., step=-0.001)

iov = base[1] * iomult

voc = pvlib.pvsystem.v_from_i(0., base[0], iov, base[2], base[3], base[4])

plt.plot(iomult, voc/voc[-1], '.')
plt.grid()


# typical
typ_v = (1. - np.array([0.0005, 0.001, 0.0015])) * voc[-1] 
typ_iomult = np.interp(typ_v, voc, iomult)

# 30 year
typ30_v = (1. - np.array([0.015, 0.03, 0.044])) * voc[-1]
typ30_iomult = np.interp(typ30_v, voc, iomult)

# severe
severe_v = (1. - np.array([0.0005, 0.001, 0.1])) * voc[-1] 
severe_iomult = np.interp(severe_v, voc, iomult)

