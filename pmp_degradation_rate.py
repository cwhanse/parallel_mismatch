# -*- coding: utf-8 -*-
"""
Calculates statistics for Pmp degradation rate

"""

# compute effective Pmp degradation rates from combination of IL and Io degradation

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

base_out = pvlib.pvsystem.singlediode(*base)

#%%

iomult = np.arange(15., 1., step=-0.001)

iov = base[1] * iomult

voc = pvlib.pvsystem.v_from_i(0., base[0], iov, base[2], base[3], base[4])

plt.plot(iomult, voc/voc[-1], '.')
plt.grid()


# typical
typ_v = (1. - np.array([0.0005, 0.001, 0.0015])) * voc[-1] 
typ_iomult = np.interp(typ_v, voc, iomult)
typ_ilmult = np.array([0.985, 0.99, 0.995])

# 30 year
typ30_v = (1. - np.array([0.015, 0.03, 0.044])) * voc[-1]
typ30_iomult = np.interp(typ30_v, voc, iomult)
typ30_ilmult = np.array([0.985**30, 0.99**30, 0.995**30])

# severe
severe_v = (1. - np.array([0.05, 0.1, 0.33])) * voc[-1] 
severe_iomult = np.interp(severe_v, voc, iomult)
#severe_ilmult = np.array([0.25, 0.99, 0.995])
severe_ilmult = np.array([0.25, 0.50, 0.75])

# for minimum and maximum
typ_iv_mm = pvlib.pvsystem.singlediode(base[0] * np.array([typ_ilmult.min(), typ_ilmult.max()]),
                                       base[1] * np.array([typ_iomult.max(), typ_iomult.min()]),
                                       base[2], base[3], base[4])
typ_deg_mm = (base_out['p_mp'] - typ_iv_mm['p_mp']) / base_out['p_mp'] * 100

typ30_iv_mm = pvlib.pvsystem.singlediode(base[0] * np.array([typ30_ilmult.min(), typ30_ilmult.max()]),
                                         base[1] * np.array([typ30_iomult.max(), typ30_iomult.min()]),
                                         base[2], base[3], base[4])
typ30_deg_mm = (base_out['p_mp'] - typ30_iv_mm['p_mp']) / base_out['p_mp'] * 100

severe_iv_mm = pvlib.pvsystem.singlediode(base[0] * np.array([severe_ilmult.min(), typ_ilmult.max()]),
                                          base[1] * np.array([severe_iomult.max(), typ_iomult.min()]),
                                          base[2], base[3], base[4])

severe_deg_mm = (base_out['p_mp'] - severe_iv_mm['p_mp']) / base_out['p_mp'] * 100

# for mean and median
rng = np.random.default_rng(seed=110457)

typ_il = rng.triangular(*typ_ilmult, [10000,]) * base[0]
typ_io = rng.triangular(*typ_iomult, [10000,]) * base[1]

typ30_il = rng.triangular(*typ30_ilmult, [10000,]) * base[0]
typ30_io = rng.triangular(*typ30_iomult, [10000,]) * base[1]

# severe degradatiom mixture of two triangular dists: 95% typical, 5% severe
severe_il = rng.triangular(*severe_ilmult, [10000,]) * base[0]
component = rng.binomial(1, 0.95, [10000,])
severe_il = np.where(component==1, typ_il, severe_il)

severe_io = rng.triangular(*severe_iomult, [10000,]) * base[1]
component = rng.binomial(1, 0.95, [10000,])
severe_io = np.where(component==1, typ_io, severe_io)

typ_iv = pvlib.pvsystem.singlediode(typ_il, typ_io, base[2], base[3], base[4])
typ30_iv = pvlib.pvsystem.singlediode(typ30_il, typ30_io, base[2], base[3], base[4])
severe_iv = pvlib.pvsystem.singlediode(severe_il, severe_io, base[2], base[3], base[4])

typ_deg_pmp = ((base_out['p_mp'] - typ_iv['p_mp']) / base_out['p_mp'] * 100)
typ30_deg_pmp = ((base_out['p_mp'] - typ30_iv['p_mp']) / base_out['p_mp'] * 100)
severe_deg_pmp = ((base_out['p_mp'] - severe_iv['p_mp']) / base_out['p_mp'] * 100)

print('Typical : {:.2f} {:.2f} {:.2f} {:.2f}%'.format(typ_deg_mm.min(), typ_deg_pmp.median(),
                                                      typ_deg_pmp.mean(), typ_deg_mm.max()))
print('Typical 30 : {:.2f} {:.2f} {:.2f} {:.2f}%'.format(typ30_deg_mm.min(), typ30_deg_pmp.median(),
                                                         typ30_deg_pmp.mean(), typ30_deg_mm.max()))
print('Severe : {:.2f} {:.2f} {:.2f} {:.2f}%'.format(severe_deg_mm.min(), severe_deg_pmp.median(),
                                                     severe_deg_pmp.mean(), severe_deg_mm.max()))

