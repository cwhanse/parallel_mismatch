# -*- coding: utf-8 -*-
"""
Runs all four scenarios: Typical Year, Typical 30 Year, Imbalanced String, Severe, to
compute the annual energy production for all array configurations.

Run time is substantial, exceeding 2 hours without parallelization (not done in this script)

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time


from scipy.optimize.elementwise import find_root, find_minimum
from pvlib.ivtools.utils import _schumaker_qspline
from scipy.interpolate import PPoly
from pvlib import singlediode as _singlediode
from pvlib.pvsystem import calcparams_pvsyst
import pvlib


def _iv_series_lambert_v_from_i(current, il, io, rs, rsh, a, neg_v_limit,
                                ndevices=None, idx=None):
    # wrapper for pvlib._singlediode._lambertw_v_from_i, handles
    # dimensions expected in series calculation
    # solve voltages at each current for each IV curve

    # slice each parameter on its ntimes dimension with idx
    if idx is not None:
        il, io, rs, rsh, a = (il[:, idx], io[:, idx], rs[:, idx], rsh[:, idx],
                              a[:, idx])

    voltages = _singlediode._lambertw_v_from_i(
        current.flatten(), il.flatten(), io.flatten(), rs.flatten(),
        rsh.flatten(), a.flatten())

    # apply negative voltage limit
    voltages[voltages < neg_v_limit] = neg_v_limit

    # reshape
    voltages = voltages.reshape(current.shape)
    return voltages


def _setup_currents(current_bkpts, string_isc, npts):
    r''' Form array of currents from string_isc down to 0.
    The array of currents will contain all values
    from current_bkpts which are less than string_isc. Remaining points are
    selected from a linear spacing [0, string_isc], avoiding points that are
    closest to values in current_bkpts.

    Parameters
    ----------
    device_isc : ndarray
        Shape (ndevices, ntimes)
    string_isc : ndarray
        Shape (ntimes,)
    npts : int
        number of current points in the returned array

    Returns
    -------
    ndarray
        shape (ntimes, npts)

    '''
    ntimes = len(string_isc)
    currents = np.zeros((ntimes, npts))

    u = current_bkpts < string_isc[np.newaxis, :]

    # have to loop on ntimes since count of device_isc < string_isc may
    # differ for each time
    for i in range(ntimes):
        # start with values in current_bkpts
        vals = np.unique(current_bkpts[u[:, i], i])
        # add string_isc and 0.
        vals = np.append(vals, [string_isc[i], 0.])
        k_i = len(vals)

        if k_i == 0:
            continue

        # Store these values
        currents[i, :k_i] = vals

        # how many more we need
        n_fill = npts - k_i
        if n_fill <= 0:
            continue

        # Build linear spacing to draw from
        grid = np.linspace(string_isc[i], 0., npts)

        # Compute distances to nearest points in current_bkpts etc.
        # shape: (grid_size, k_i)
        dists = np.abs(grid[:, None] - vals[None, :])

        # nearest distance per grid point
        nearest_dist = np.min(dists, axis=1)

        # inverse-distance weights (higher near must-have values)
        weights = 1.0 / (nearest_dist + 1e-12)  # 1e-12 to avoid div by 0

        # Avoid re-selecting original values exactly
        mask_existing = np.isclose(nearest_dist, 0.0, atol=1e-12)
        weights[mask_existing] = 0.0

        # Select top-weighted grid points
        idx = np.argpartition(weights, -n_fill)[-n_fill:]
        selected = grid[idx]

        # Combine and add to A
        currents[i, :] = np.concatenate([vals, selected])

    # return sorted in descending order for each time
    currents = -np.sort(-currents, axis=1)

    return currents


def _insert_mpp(vol, cur, vmp, imp):
    # insert MPP into current and voltage arrays
    cur_out = np.empty((vol.shape[0], vol.shape[1] + 1))
    vol_out = np.empty((vol.shape[0], vol.shape[1] + 1))
    # insert mpp value
    for i in range(vol.shape[0]):
        idx = vol[i, :].searchsorted(vmp[i], side='left')
        vol_out[i, :idx] = vol[i, :idx]
        vol_out[i, idx] = vmp[i]
        vol_out[i, idx+1:] = vol[i, idx:]
        cur_out[i, :idx] = cur[i, :idx]
        cur_out[i, idx] = imp[i]
        cur_out[i, idx+1:] = cur[i, idx:]

    return vol_out, cur_out


def _singlediode_mismatch(photocurrent, saturation_current, resistance_series,
                        resistance_shunt, nNsVth, neg_v_limit=0.,
                        npts=100):
    r'''Solve the IV curve for series-connected devices where each device
    is described by the single diode equation.

    Uses a simplified model for reverse bias behavior, where current is
    unbounded at a constant reverse bias voltage ``neg_v_limit``.

    Input parameter ``photocurrent`` must have shape (devices, times).
    Input parameters ``saturation_current``, ``resistance_series``,
    ``resistance_shunt``, ``nNsVth`` may be arrays. If arrays, must be
    broadcastable to the shape of ``photocurrent``.

    Parameters
    ----------
    photocurrent : numeric
        photocurrent (A). Must have shape (devices, times).
    saturation_current : numeric
        saturation current (A). Must be broadcastable with photocurrent.
    resistance_series : numeric
        series resistance (ohm). Must be broadcastable with photocurrent.
    resistance_shunt : numeric
        shunt resistance (ohm). Must be broadcastable with photocurrent.
    nNsVth : numeric
        product of diode factor n, number of series cells Ns, and
        thermal voltage (Vth), (V). Must be broadcastable with photocurrent.
    neg_v_limit : float, optional
        Limit on reverse bias voltage, from cell breakdown voltage or reverse
        bias diode activation voltage (V). Should be negative. For example,
        if neg_v_limit=-5, then at V=-5 current is unbounded in the positive
        direction.
    npts : int, optional
        Number of points used to discretize the returned IV curves.

    Returns
    -------
    string_isc : numeric
        Short-circuit current for the string. [A]
    string_voc : numeric
        Open-circuit voltage for the string. [V]
    string_imp : numeric
        Current at maximum power for the string. [A]
    string_vmp : numeric
        Voltage at maximum power for the string. [V]

    '''
    # target shape is ndevices x ntimes
    IL, I0, Rs, Rsh, a = \
        np.broadcast_arrays(photocurrent, saturation_current,
                            resistance_series, resistance_shunt, nNsVth)

    ndevices, ntimes = IL.shape

    # solve for current at negative voltage limit for each device.
    # these currents create breakpoints in the series IV curve
    current_bkpts = _singlediode._lambertw_i_from_v(
        neg_v_limit, IL, I0, Rs, Rsh, a)

    # find Isc for string IV curve
    # bounds, 1d array for each time
    max_isc = current_bkpts.max(axis=0) * 1.01
    min_isc = current_bkpts.min(axis=0) * 0.99

    # Use an index idx so that find_root can slice arguments
    # Internally find_root will slice current as each element converges
    # As an argument, idx lets find_root also slice the other parameters
    # Remove idx and use preserve_shape once available in find_root
    # https://github.com/scipy/scipy/issues/24869
    idx = np.arange(ntimes)
    def isc_optfn(current, idx):
        # current is ntimes only since it is common for all devices.
        # other parameters are ntimes x ndevices
        # broadcast current to ntimes x ndevices
        cur2 = np.broadcast_to(current[np.newaxis, :], (ndevices, len(current)))
        v = _iv_series_lambert_v_from_i(
            cur2, IL, I0, Rs,
            Rsh, a, neg_v_limit, ndevices, idx)
        # return string voltage
        return v.sum(axis=0)

    isc_result = find_root(
        isc_optfn,
        (min_isc, max_isc), args=(idx,))
    string_isc = isc_result.x  # 1d in ntimes

    # find Voc for string-level curves
    device_voc = _singlediode._lambertw_v_from_i(0., IL, I0, Rs, Rsh, a)
    string_voc = device_voc.sum(axis=0)

    # prepare for MPP calculation by computing voltages on a grid of currents
    # discretize current from string_isc down to 0 at each time step
    # Include each device's current at neg_v_limit since these will be the
    # curvature breakpoints in the series IV curve.
    # Leave a space for inserting MPP
    # currents is ntimes x npts-1, decreasing
    currents = _setup_currents(current_bkpts, string_isc, npts-1)

    # shape all arrays to be ndevices x ntimes x ncurrents
    cur3 = np.repeat(currents[np.newaxis, :, :], ndevices, axis=0)
    cur3, il, io, rs, rsh, a3 = np.broadcast_arrays(
        cur3, IL[:, :, np.newaxis], I0[:, :, np.newaxis], Rs[:, :, np.newaxis],
        Rsh[:, :, np.newaxis], a[:, :, np.newaxis])

    # solve voltages at each current for each IV curve
    device_voltages = _iv_series_lambert_v_from_i(
        cur3, il, io, rs, rsh, a3, neg_v_limit)

    # add voltage across devices to get string voltage
    # voltages is ntimes x ncurrents
    voltages = device_voltages.sum(axis=0)

    # objective function for MPP
    idx = np.arange(ntimes)
    def mpp_optfn(current, idx):
        # solve power at each current for each device
        # current is ntimes only since it is common for all devices.
        # other parameters are ntimes x ndevices
        # broadcast current to ntimes x ndevices
        cur2 = np.broadcast_to(current[np.newaxis, :],
                               (ndevices, len(current)))
        voltage = _iv_series_lambert_v_from_i(
            cur2, IL, I0, Rs, Rsh, a, neg_v_limit, ndevices, idx)

        # return negative of string power
        return current * -voltage.sum(axis=0)

    # mpp calculation
    idxs = np.argmax(currents * voltages, axis=1)
    # reversed since currents is decreasing
    idcs = idxs[:, np.newaxis] + [1, 0, -1]
    intervals = np.take_along_axis(currents, idcs, axis=1)
    init = (intervals[:, 0], intervals[:, 1], intervals[:, 2])

    idx = np.arange(ntimes)
    imp_result = find_minimum(mpp_optfn, init, args=(idx,))
    string_imp = imp_result.x

    # use string_imp to calculate string_vmp
    idx = np.arange(ntimes)
    cur2 = np.broadcast_to(string_imp[np.newaxis, :],
                           (ndevices, ntimes))
    device_vmp = _iv_series_lambert_v_from_i(
        cur2, IL, I0, Rs, Rsh, a, neg_v_limit, ndevices)
    string_vmp = device_vmp.sum(axis=0)

    voltages, currents = _insert_mpp(
        voltages, currents, string_vmp, string_imp)

    return string_isc, string_voc, string_imp, string_vmp, currents, voltages


def _v_from_i_mismatch(currents, photocurrent, saturation_current,
                       resistance_series,
                       resistance_shunt, nNsVth, neg_v_limit=0.):
    r'''Solve the IV curve for series-connected devices where each device
    is described by the single diode equation.

    Uses a simplified model for reverse bias behavior, where current is
    unbounded at a constant reverse bias voltage ``neg_v_limit``.

    Input parameter ``photocurrent`` must have shape (devices, times).
    Input parameters ``saturation_current``, ``resistance_series``,
    ``resistance_shunt``, ``nNsVth`` may be arrays. If arrays, must be
    broadcastable to the shape of ``photocurrent``.

    Parameters
    ----------
    currents : numeric
        String current (A) at which string voltage is to be computed.
        Must have shape (times, currents).
    photocurrent : numeric
        photocurrent (A). Must have shape (devices, times).
    saturation_current : numeric
        saturation current (A). Must be broadcastable with photocurrent.
    resistance_series : numeric
        series resistance (ohm). Must be broadcastable with photocurrent.
    resistance_shunt : numeric
        shunt resistance (ohm). Must be broadcastable with photocurrent.
    nNsVth : numeric
        product of diode factor n, number of series cells Ns, and
        thermal voltage (Vth), (V). Must be broadcastable with photocurrent.
    neg_v_limit : float, optional
        Limit on reverse bias voltage, from cell breakdown voltage or reverse
        bias diode activation voltage (V). Should be negative. For example,
        if neg_v_limit=-5, then at V=-5 current is unbounded in the positive
        direction.

    Returns
    -------
    voltages : numeric
        String voltage (V) at the current points, shape (times).

    '''
    # target shape is ndevices x ntimes
    IL, I0, Rs, Rsh, a = \
        np.broadcast_arrays(photocurrent, saturation_current,
                            resistance_series, resistance_shunt, nNsVth)

    ndevices, ntimes = IL.shape

    # shape all arrays to be ndevices x ntimes x ncurrents
    cur3 = np.repeat(currents[np.newaxis, :, :], ndevices, axis=0)
    cur3, il, io, rs, rsh, a3 = np.broadcast_arrays(
        cur3, IL[:, :, np.newaxis], I0[:, :, np.newaxis], Rs[:, :, np.newaxis],
        Rsh[:, :, np.newaxis], a[:, :, np.newaxis])

    # solve voltages at each current for each IV curve
    # applies neg_v_limit
    device_voltages = _iv_series_lambert_v_from_i(
        cur3, il, io, rs, rsh, a3, neg_v_limit)

    # sum voltage across devices to get string voltage
    # voltages is ntimes x ncurrents
    voltages = device_voltages.sum(axis=0)

    return voltages


def _iv_series_lambertw(photocurrent, saturation_current, resistance_series,
                        resistance_shunt, nNsVth, neg_v_limit=0.,
                        npts=100):
    r'''Solve the IV curve for series-connected devices where each device
    is described by the single diode equation.

    Uses a simplified model for reverse bias behavior, where current is
    unbounded at a constant reverse bias voltage ``neg_v_limit``.

    Input parameter ``photocurrent`` must have shape (devices, times).
    Input parameters ``saturation_current``, ``resistance_series``,
    ``resistance_shunt``, ``nNsVth`` may be arrays. If arrays, must be
    broadcastable to the shape of ``photocurrent``.

    Parameters
    ----------
    photocurrent : numeric
        photocurrent (A). Must have shape (devices, times).
    saturation_current : numeric
        saturation current (A). Must be broadcastable with photocurrent.
    resistance_series : numeric
        series resistance (ohm). Must be broadcastable with photocurrent.
    resistance_shunt : numeric
        shunt resistance (ohm). Must be broadcastable with photocurrent.
    nNsVth : numeric
        product of diode factor n, number of series cells Ns, and
        thermal voltage (Vth), (V). Must be broadcastable with photocurrent.
    neg_v_limit : float, optional
        Limit on reverse bias voltage, from cell breakdown voltage or reverse
        bias diode activation voltage (V). Should be negative. For example,
        if neg_v_limit=-5, then at V=-5 current is unbounded in the positive
        direction.
    npts : int, optional
        Number of points used to discretize the returned IV curves.

    Returns
    -------
    voltages : numeric
        Voltage points for the series IV curves (V), shape
        (times, npts).
    currents : numeric
        Current points for the series IV curves (A), shape
        (times, npts).
    string_vmp : numeric
        Maximum power voltage for the series IV curves (V), shape
        (ntimes)
    string_imp : numeric
        Maximum power current for the series IV curves (A), shape
        (ntimes)

    '''
    # target shape is ndevices x ntimes
    IL, I0, Rs, Rsh, a = \
        np.broadcast_arrays(photocurrent, saturation_current,
                            resistance_series, resistance_shunt, nNsVth)

    ndevices, ntimes = IL.shape

    # solve for current at negative voltage limit for each device.
    # these currents create breakpoints in the series IV curve
    current_bkpts = _singlediode._lambertw_i_from_v(
        neg_v_limit, IL, I0, Rs, Rsh, a)

    # find Isc for string IV curve
    # bounds, 1d array for each time
    max_isc = current_bkpts.max(axis=0) * 1.1
    min_isc = current_bkpts.min(axis=0) * 0.9

    # Use an index idx so that find_root can slice arguments
    # Internally find_root will slice current as each element converges
    # As an argument, idx lets find_root also slice the other parameters
    # Remove idx and use preserve_shape once available in find_root
    # https://github.com/scipy/scipy/issues/24869
    idx = np.arange(ntimes)
    def isc_optfn(current, idx):
        # current is ntimes only since it is common for all devices.
        # other parameters are ntimes x ndevices
        # broadcast current to ntimes x ndevices
        cur2 = np.broadcast_to(current[np.newaxis, :], (ndevices, len(current)))
        v = _iv_series_lambert_v_from_i(
            cur2, IL, I0, Rs,
            Rsh, a, neg_v_limit, ndevices, idx)
        # return string voltage
        return v.sum(axis=0)

    isc_result = find_root(
        isc_optfn,
        (min_isc, max_isc), args=(idx,))
    string_isc = isc_result.x  # 1d in ntimes

    # discretize current from string_isc down to 0 at each time step
    # Include each device's current at neg_v_limit since these will be the
    # curvature breakpoints in the series IV curve.
    # Leave a space for inserting MPP
    # currents is ntimes x npts-1, decreasing
    currents = _setup_currents(current_bkpts, string_isc, npts-1)

    # shape all arrays to be ndevices x ntimes x ncurrents
    cur3 = np.repeat(currents[np.newaxis, :, :], ndevices, axis=0)
    cur3, il, io, rs, rsh, a3 = np.broadcast_arrays(
        cur3, IL[:, :, np.newaxis], I0[:, :, np.newaxis], Rs[:, :, np.newaxis],
        Rsh[:, :, np.newaxis], a[:, :, np.newaxis])

    # solve voltages at each current for each IV curve
    device_voltages = _iv_series_lambert_v_from_i(
        cur3, il, io, rs, rsh, a3, neg_v_limit)

    # add voltage across devices to get string voltage
    # voltages is ntimes x ncurrents
    voltages = device_voltages.sum(axis=0)

    # objective function for MPP
    idx = np.arange(ntimes)
    def mpp_optfn(current, idx):
        # solve power at each current for each device
        # current is ntimes only since it is common for all devices.
        # other parameters are ntimes x ndevices
        # broadcast current to ntimes x ndevices
        cur2 = np.broadcast_to(current[np.newaxis, :],
                               (ndevices, len(current)))
        voltage = _iv_series_lambert_v_from_i(
            cur2, IL, I0, Rs, Rsh, a, neg_v_limit, ndevices, idx)

        # return negative of string power
        return current * -voltage.sum(axis=0)

    # mpp calculations
    idxs = np.argmax(currents * voltages, axis=1)
    # reversed since currents is decreasing
    idcs = idxs[:, np.newaxis] + [1, 0, -1]
    intervals = np.take_along_axis(currents, idcs, axis=1)
    init = (intervals[:, 0], intervals[:, 1], intervals[:, 2])

    idx = np.arange(ntimes)
    imp_result = find_minimum(mpp_optfn, init, args=(idx,))
    string_imp = imp_result.x

    # use string_imp to calculate string_vmp
    idx = np.arange(ntimes)
    cur2 = np.broadcast_to(string_imp[np.newaxis, :],
                           (ndevices, ntimes))
    device_vmp = _iv_series_lambert_v_from_i(
        cur2, IL, I0, Rs, Rsh, a, neg_v_limit, ndevices)
    string_vmp = device_vmp.sum(axis=0)

    voltages, currents = _insert_mpp(
        voltages, currents, string_vmp, string_imp)

    return voltages, currents, string_vmp, string_imp


def _interpolate_curve(vx, voltages, currents, npts):
    r'''Interpolate IV curve to vx values.

    Parameters
    ----------
    vx : ndarray
        Must be shape (ntimes,)
    voltages : ndarray
        Must be shape (ntimes, points).
        Expected to be sorted and range from 0. to Voc for each time.
    currents : ndarray
       Must be of the same shape as ``voltages``.
    npts : int, optional
        Number of points in the returned IV curve. The default is 100.

    Returns
    -------
    tuple of ndarray
        (voltages, currents), where each array is shape (ntimes, npts)

    '''
    # interpolate each IV curve to get current at each value of vs
    # use Schumaker spline, which preserves monotonicity.
    voc = voltages.max(axis=1)
    ntimes=len(voc)
    current_int = np.zeros((ntimes, npts), dtype=float)

    pps = [None] * ntimes

    for k in range(ntimes):
        t, c, _, _ = _schumaker_qspline(voltages[k, :], currents[k, :])
        c = c.T  # swap (nintervals, ndegree+1) to (ndegree+1, nintervals)
        try:
            pp = PPoly(c, t)
        except:
            print('error at ' + str(k))
        current_int[k, :] = pp(vx[k, :])
        pps[k] = pp

    # apply bypass diode assumption
    current_int[vx > voc[:, np.newaxis]] = 0.

    return current_int, pps


def _iv_parallel(ivs, npts=100, refine_mpp=True):
    r'''Calculate the IV curves for strings connected in parallel.

    Assumes bypass diodes anti-parallel with each device.

    Parameters
    ----------
    ivs : iterable
        Each element must be a tuple (voltages, currents). Both voltages
        and currents must be ndarrays with the same shape (ntimes, npoints).
        For each time, voltages is expected to be sorted from 0. to Voc
        along the npoints axis.
    npts : int, optional
        Number of points in the returned IV curves. The default is 100.

    Returns
    -------
    list
        Each element is a tuple of ndarray
        (voltages, currents), where each array is shape (ntimes, npts)

    '''
    nstrings = len(ivs)
    c0 = ivs[0]
    ntimes, npts_in = c0[0].shape

    # form series voltages into a common 3d array (nstrings, ntimes, npts_in)
    # for efficient extraction of voc for each time
    vs_big = np.zeros((nstrings, ntimes, npts_in))
    for k in range(nstrings):
        vs_big[k, :, :] = ivs[k][0]
    voc = vs_big.max(axis=2).max(axis=0)  # should be 1d (ntimes,)
    # set up array to contain interpolated current for each string and time
    currents = np.zeros_like(vs_big)

    # done with vs_big
    del vs_big

    # set up common voltages for parallel IV curves (ntimes, npts)
    vs = np.linspace(0., voc, num=npts, axis=1)

    # set up to contain interpolators for each string
    # each element of pps is a list of interpolators of length ntimes
    pps = [None] * nstrings

    for j, iv in enumerate(ivs):
        currents[j, :, :], pps[j] = _interpolate_curve(vs, iv[0], iv[1], npts)

    # parallel current summed over strings
    cs = currents.sum(axis=0)

    # find maximum power point on combined curve at each time
    vmpp = np.zeros_like(voc)
    impp = np.zeros_like(voc)

    # make evaluation functions
    # rearrange pps to a list of length ntimes, elements of length nstrings
    pps = [[pps[j][k] for j in range(nstrings)] for k in range(ntimes)]
    for k in range(ntimes):
        # rough estimate
        cx = sum([pps[k][j](vs[k, :]) for j in range(nstrings)])
        idx = np.argmax(cx * vs[k, :])
        vmpp[k] = vs[k, idx]
        impp[k] = cx[idx]
        # refined estimate
        if refine_mpp:
            vref = np.linspace(vs[k, idx-3], vs[k, idx+3], 1000)
            cref = sum([pps[k][j](vref) for j in range(nstrings)])
            idx = np.argmax(cref * vref)
            vmpp[k] = vref[idx]
            impp[k] = cref[idx]
    return vs, cs, vmpp, impp


class PVSystem:
    def __init__(self, nMPPT, nstrings  , samples =100):
        # ntimes         : number of Time samples
        # nMPPT          : number of MPPT trackers
        # ndevices       : number of devices in each string
        # Strings_Params : Base String Params
        self.nMPPT = nMPPT

        self.nstrings = nstrings
        self.samples = samples
        #self.Strings = Strings
        #self.nSpM = ndevices/nMPPT


def get_factors(n):
    factors = set()
    # Iterate only up to the square root of n
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            factors.add(i)
            factors.add(n // i)
    return sorted(list(factors))


def Sim(SysParams, Strings, Vis=False, k=0):

    print('   ' + str(SysParams.nMPPT))
    MPPT_IV =[]
    if Vis:
        c_j=1
        plt.figure()
    ccc=0
    for i in range(0, len(Strings), SysParams.nMPPT):
        block = Strings[i: i + SysParams.nMPPT]
        ivs = [(block[k][0], block[k][1]) for k in range(len(block))]
        if Vis:
            key_c = list(mcolors.TABLEAU_COLORS.keys())[ccc]
            ccc+=1
            for c, [vs, cs] in enumerate(ivs):
                 plt.plot(vs[k, :], cs[k, :], '--', c=mcolors.TABLEAU_COLORS[key_c],
                          label=f'String_{c+1}_Array{c_j}')

        v, i, vmp, imp = _iv_parallel(ivs, npts=SysParams.samples)
        MPPT_IV.append((v, i, vmp, imp))
        if Vis:

            plt.plot(MPPT_IV[-1][0][k, :], MPPT_IV[-1][1][k, :], '.-', c=mcolors.TABLEAU_COLORS[key_c],label = f'Array_{c_j }')
            plt.plot(MPPT_IV[-1][2][k], MPPT_IV[-1][3][k], marker = 's',c=mcolors.TABLEAU_COLORS['tab:red'])
            print(f'Array_{i+1} Power :{MPPT_IV[-1][2][k] * MPPT_IV[-1][3][k]}')
            c_j+=1
    if Vis:
        plt.plot(MPPT_IV[-1][2][k], MPPT_IV[-1][3][k], 'rs',label ='Mpp')
        plt.xlabel('Voltage')
        plt.ylabel('Current')
        plt.legend(bbox_to_anchor=(1, 1))
        plt.show()


    return(MPPT_IV)


def Create_Weather_Data(ndevices=10 , lat=35.054750, lon=-106.540055,
                        Weather_mean=1, Weather_Var=.1, Cutoff=15):


    # Create mount
    mount = pvlib.pvsystem.FixedMount(surface_tilt = 29.3,
                                      surface_azimuth = 180.0,
                                      racking_model = 'open_rack')

    # Location abq


    # Fetch weather data
    data, _ = pvlib.iotools.get_pvgis_tmy(lat, lon)

    # Filter for high sun
    solpos = pvlib.solarposition.get_solarposition(data.index, lat, lon)
    Mask = solpos.elevation>=Cutoff
    solpos = solpos.loc[Mask]
    data = data.loc[solpos.index]

    # Get POA
    G = pvlib.irradiance.get_total_irradiance(surface_tilt=mount.surface_tilt,
                                              surface_azimuth=mount.surface_azimuth,
                                              solar_zenith=solpos.zenith,
                                              solar_azimuth=solpos.azimuth,
                                              dni = data.dni,
                                              ghi = data.ghi,
                                              dhi = data.dhi
                                              ).poa_global
    # model temerature
    T = pvlib.temperature.pvsyst_cell(G, data.temp_air,data.wind_speed)

    return G, T, Mask


def Print_Gain_Results(out):
    for test in out:
        power = 0
        for r in test:
            power += sum(r[2]*r[3])
        nMPPT = len(test)
        print(f'{int(len(out[0]) / nMPPT)} Strings per MPPT produce {round(power/1000,6)} KWh ')
    print()
    test0 = out[0]
    test1 = out[-1]

    powermin = 0
    powermax = 0
    for r in test0:
        powermax += sum(r[2]*r[3])

    for r in test1:
        powermin += sum(r[2]*r[3])

    print(f'Ratio of {len(test0)} MPPT over {len(test1)} MPPT :   {(powermax/powermin)} ')
    return(None)

#%%
# Choose number of modules per string
ndevices = 25
# Choose total number of strings
total_Strings = 96*2

# generate random Il starting points
## 8760 is the number of hours in a year
rng = np.random.default_rng(seed=110457)
il_rand_master = rng.uniform(.99, 1.05, [ndevices, 8760, total_Strings])*13  # 13 A is I_L_ref

# Generate consistent IL starting points
il_master = np.ones([ndevices, 8760, total_Strings])*13

# Create POA(G) and Temp(T) data for Abq NM with a minimum sun angle > 15
## Mask to go from 8760 -> sun angle>15 is also returned

G, T, Mask = Create_Weather_Data(ndevices=ndevices, lat=35.054750, lon=-106.540055, Cutoff = 15)

# shape arrays to ndevices x ntimes
G = np.tile(G, (ndevices, 1))
T = np.tile(T, (ndevices, 1))

Deg_Results = {}
Deg_Results['nMPPT'] = get_factors(total_Strings)


# degradation distributions
# for photocurrent: triangular, annual rate
deg_mode_c = 0.99
deg_min_c = 0.985
deg_max_c = 0.995

# for saturation current: triangular, annual rate
deg_mode_v = 1.0255  # 0.999 change in Voc
deg_max_v  =  1.038  # 0.9985
deg_min_v  = 1.0132  # 0.9995


#%% Typical year

start = time.time()
# show Graphs?
Vis = False

Strings = []
for i in range(total_Strings):
    # String parameters with degradation built in
    asc = 0.0015
    N = 1.3
    mu_gamma = 0
    # apply deg rate in decreasing as time goes on
    deg_rate_per_hour_c = rng.triangular(deg_min_c, deg_mode_c, deg_max_c, [ndevices,]) ** (1./8760)
    hour_range_c =  np.arange(1, 8760+1, 1)
    deg_mult_per_hour_c = deg_rate_per_hour_c[:, np.newaxis]**hour_range_c
    il = il_rand_master[:,:,i].copy() * deg_mult_per_hour_c
    il = il[:,Mask]

    deg_rate_per_hour_v = rng.triangular(deg_min_v,deg_mode_v,deg_max_v, [ndevices,]) ** (1./8760)
    hour_range_v =  np.arange(1, 8760+1, 1)
    deg_mult_per_hour_v =  deg_rate_per_hour_v[:, np.newaxis]**hour_range_v
    io = 3.9e-10 * deg_mult_per_hour_v
    io = io[:,Mask]

    rsh = 1144
    rsh0 = 3850
    rs = .6
    ns = 72
    npts = 100 # points in IV curve

    # Form in dictionary to pass to calcparams
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

    # Generate 5 SDM params for modules in the array and add to list
    Strings.append( calcparams_pvsyst(G, T, **String_Params) )

# Generate IV curves for each string
print("starting series IV calculation for {} strings".format(len(Strings)))
start = time.time()
Strings_Series = [_iv_series_lambertw(il_, io_, rs_, rsh_, nNsVth_, neg_v_limit= -1., npts=npts) for (il_, io_, rs_, rsh_, nNsVth_) in Strings]
end = time.time()
print('done')
print(end - start)

#%%

# Configs to explore
MPPT_Config = get_factors(total_Strings)

# Create inputs for Simulation
inputs = []
for nMPPT in MPPT_Config:
    # NUmber of strings in //
    nstrings = int(total_Strings/nMPPT)

    SysParams =PVSystem(nMPPT = nMPPT,
                    nstrings = nstrings
                    )
    inputs.append(SysParams)
# Run simulation
# multiprocessed for speed

start = time.time()
out = []
for inp in inputs:
    print(inp.nMPPT)
    out.append(Sim(inp, Strings_Series, Vis=Vis))

end = time.time()

print(f'Time to execute {end - start}')
print(len(out))

Results_Array =  []
for test in out:
    power = 0
    for r in test:
        power += sum(r[2]*r[3])
    nMPPT = len(test)
    Results_Array.append(power)

key = 'Typical Year'
Deg_Results[key] = np.array(Results_Array)

#%% Imbalance strings

# remove a module from one string
idx = 0

Short_String = []
Short_String.append(tuple([Strings[idx][i][1:,:] for i in range(len(Strings[idx]))]))

# Generate IV curve for the short string
Short_Strings_Series = [_iv_series_lambertw(il_, io_, rs_, rsh_, nNsVth_, neg_v_limit= -1., npts=npts) for (il_, io_, rs_, rsh_, nNsVth_) in Short_String]

String_Series_short = Strings_Series.copy()
String_Series_short[idx] = Short_Strings_Series[0]

# Configs to explore
MPPT_Config = get_factors(total_Strings)

# Create inputs for Simulation
inputs = []
for nMPPT in MPPT_Config:
    # NUmber of strings in //
    nstrings = int(total_Strings/nMPPT)

    SysParams =PVSystem(nMPPT = nMPPT,
                    nstrings = nstrings
                    )
    inputs.append(SysParams)
# Run simulation
# multiprocessed for speed
out = [Sim(SysInput, String_Series_short ,Vis=Vis) for SysInput in inputs]

Results_Array =  []
for test in out:
    power = 0
    for r in test:
        power += sum(r[2]*r[3])
    nMPPT = len(test)
    Results_Array.append(power)

key ='Imbalanced String'
Deg_Results[key] = np.array(Results_Array)

#%% 30 year degradation

years = 30

deg_mode_v = 2.071  # 0.97 = 0.999**30
deg_min_v  = 1.44  # 0.985 = 0.9995**30
deg_max_v  = 2.907  # 0.956 = 0.9985**30

deg_mode_c = 0.99**years
deg_min_c = 0.985**years
deg_max_c = 0.995**years

start = time.time()
# show Graphs?
Vis = False

Strings = []
for i in range(total_Strings):
    # String parameters with degradation built in
    asc =0.0015
    N = 1.3
    mu_gamma = 0
    # apply deg rate in decreasing as time goes on
    deg_rate_per_hour_c = rng.triangular(deg_min_c,deg_mode_c,deg_max_c, [ndevices,]) ** (1./8760)
    hour_range_c =  np.arange(1, 8760+1, 1)
    deg_mult_per_hour_c = deg_rate_per_hour_c[:, np.newaxis]**hour_range_c
    il = il_rand_master[:,:,i].copy()*deg_mult_per_hour_c
    il = il[:,Mask]

    deg_rate_per_hour_v = rng.triangular(deg_min_v,deg_mode_v,deg_max_v, [ndevices,]) ** (1./8760)
    hour_range_v =  np.arange(1, 8760+1, 1)
    deg_mult_per_hour_v =  deg_rate_per_hour_v[:, np.newaxis]**hour_range_v
    io = 3.9e-10 * deg_mult_per_hour_v
    io = io[:,Mask]

    rsh = 1144
    rsh0 = 3850
    rs = .6
    ns = 72
    npts = 100 # points in IV curve

    # Form in dictionary to pass to calcparams
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

    # Generate 5 SDM params for modules in the array and add to list
    Strings.append( calcparams_pvsyst(G, T, **String_Params) )

# Generate IV curves for each string
# multiprocessed for speed
Strings_Series=[_iv_series_lambertw(il_, io_, rs_, rsh_, nNsVth_, neg_v_limit= -1., npts=npts) for (il_, io_, rs_, rsh_, nNsVth_) in Strings]

# Configs to explore
MPPT_Config = get_factors(total_Strings)

# Create inputs for Simulation
inputs = []
for nMPPT in MPPT_Config:
    # NUmber of strings in //
    nstrings = int(total_Strings/nMPPT)

    SysParams =PVSystem(nMPPT = nMPPT,
                    nstrings = nstrings
                    )
    inputs.append(SysParams)
# Run simulation
# multiprocessed for speed
out = [Sim(SysInput,Strings_Series,Vis=Vis) for SysInput in inputs]

end = time.time()

print(f'Time to execute {end - start}')

Results_Array =  []
for test in out:
    power = 0
    for r in test:
        power += sum(r[2]*r[3])
    nMPPT = len(test)
    Results_Array.append(power)

key = 'Typical 30 Year'
Deg_Results[key] = np.array(Results_Array)


#%% Severe degradation

severe_iomult = np.array([3.36168, 11.2896, 15])
severe_ilmult = np.array([0.25, 0.50, 0.75])

typ_iomult = np.array([1.0132, 1.0255, 1.038])
typ_ilmult = np.array([0.985, 0.99, 0.995])

start = time.time()
# show Graphs?
Vis = False

Strings = []
for i in range(total_Strings):
    # String parameters with degradation built in
    asc = 0.0015
    N = 1.3
    mu_gamma = 0

    # severe degradatiom mixture of two triangular dists: 95% typical, 5% severe
    typ_il = rng.triangular(*typ_ilmult, [ndevices,])
    typ_io = rng.triangular(*typ_iomult, [ndevices,])
    severe_il = rng.triangular(*severe_ilmult, [ndevices,])
    component = rng.binomial(1, 0.95, [ndevices,])
    severe_il = np.where(component==1, typ_il, severe_il)
    
    severe_io = rng.triangular(*severe_iomult, [ndevices,])
    component = rng.binomial(1, 0.95, [ndevices,])
    severe_io = np.where(component==1, typ_io, severe_io)

    # apply deg rate in decreasing as time goes on
    deg_rate_per_hour_c = severe_il ** (1./8760)
    hour_range_c =  np.arange(1, 8760+1, 1)
    deg_mult_per_hour_c = deg_rate_per_hour_c[:, np.newaxis]**hour_range_c
    il = il_rand_master[:,:,i].copy()*deg_mult_per_hour_c
    il = il[:,Mask]

    deg_rate_per_hour_v = severe_io ** (1./8760)
    hour_range_v =  np.arange(1, 8760+1, 1)
    deg_mult_per_hour_v =  deg_rate_per_hour_v[:, np.newaxis]**hour_range_v
    io = 3.9e-10 * deg_mult_per_hour_v
    io = io[:,Mask]

    rsh = 1144
    rsh0 = 3850
    rs = .6
    ns = 72
    npts = 100 # points in IV curve

    # Form in dictionary to pass to calcparams
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

    # Generate 5 SDM params for modules in the array and add to list
    Strings.append( calcparams_pvsyst(G, T, **String_Params) )

# Generate IV curves for each string
# multiprocessed for speed
Strings_Series=[_iv_series_lambertw(il_, io_, rs_, rsh_, nNsVth_, neg_v_limit= -1., npts=npts) for (il_, io_, rs_, rsh_, nNsVth_) in Strings]

# Configs to explore
MPPT_Config = get_factors(total_Strings)

# Create inputs for Simulation
inputs = []
for nMPPT in MPPT_Config:
    # NUmber of strings in //
    nstrings = int(total_Strings/nMPPT)

    SysParams =PVSystem(nMPPT = nMPPT,
                    nstrings = nstrings
                    )
    inputs.append(SysParams)
# Run simulation
# multiprocessed for speed
out = [Sim(SysInput,Strings_Series,Vis=Vis) for SysInput in inputs]

end = time.time()

print(f'Time to execute {end - start}')

Results_Array =  []
for test in out:
    power = 0
    for r in test:
        power += sum(r[2]*r[3])
    nMPPT = len(test)
    Results_Array.append(power)


key ='Severe Degradation'
Deg_Results[key] = np.array(Results_Array)

#%% summary

Results_DF = pd.DataFrame(Deg_Results)
Results_DF.to_csv('Results.csv', index=False)


#%%

with open('Results.csv', 'r') as infile:
    Results_DF = pd.read_csv(infile)

keys = ['Typical Year','Typical 30 Year','Severe Degradation','Imbalanced String']
for key in keys:
    Results_DF[key+ ' Loss'] = np.round((Results_DF[key].values/ Results_DF[key].values[0])*100,3)
labels=[]
for i in range(len(Results_DF['nMPPT'])):
    labels.append(f'{Results_DF['nMPPT'].values[i]}x{Results_DF['nMPPT'].values[-i-1]}')
# print( labels)
Results_DF['Strings in parallel x nMPPT'] = labels

plt.figure()
for key in ['Typical Year Loss' , 'Typical 30 Year Loss','Severe Degradation Loss','Imbalanced String Loss']:
    plt.plot(Results_DF[key].values,label = key[:-5])
plt.xticks(np.arange(len(Results_DF['nMPPT'])), Results_DF['nMPPT'] )
plt.xlabel('Strings in parallel per MPPT')
plt.ylabel('Ratio of Energy' )
plt.grid(True)
plt.legend()
plt.savefig('CombineDeg.png')

plt.figure()
for key in ['Typical Year Loss' , 'Typical 30 Year Loss','Severe Degradation Loss','Imbalanced String Loss']:
    plt.plot(Results_DF[key].values,label = key[:-5])
plt.xticks(np.arange(len(Results_DF['nMPPT'])), Results_DF['nMPPT'] )
plt.xlabel('Strings in parallel per MPPT')
plt.ylabel('Ratio of Energy' )
plt.ylim(99.5,100)
plt.grid(True)
plt.legend()
plt.savefig('CombineDeg_Zoom.png')

