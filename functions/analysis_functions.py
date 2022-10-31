import numpy as np
import pandas as pd
import segyio
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from sklearn.linear_model import LinearRegression


def traces_cut(traces_raw, horizon, left_boundary, right_boundary, dis_step):
    """

    Cut signals from seismogram or seismic section to subsequent analysis or graphic analysis.

    Parameters
    ----------
    traces_raw : array_like
        Seismogram or seismic section

    horizon : array_like
        1D array from Petrel or constant value array. Must be same length as signal_raw.

    left_boundary : int
        Left boundary for cutting im milliseconds.

    right_boundary : int
        Right boundary for cutting in milliseconds.

    dis_step : int
        Discretization step in signal.

    Returns
    -------
    signal_cut : array_like
        Cutted signals.
    """
    if traces_raw.shape[0] != horizon.shape[0]:
        raise ValueError('Traces shape and horizon shape doesnt match')

    signal_cut = np.array([])
    for ind, signal in enumerate(traces_raw):
        trace = signal[int(horizon[ind] // dis_step - left_boundary // dis_step):int(
            horizon[ind] // dis_step + right_boundary // dis_step)]
        signal_cut = np.vstack([signal_cut, trace]) if signal_cut.size else trace

    return signal_cut


def window_smooth(signal_raw, dis_step, dtw):
    """

    Convolute signal with Blackman window.

    Parameters
    ----------
    signal_raw : array_like
        Signal to smooth.

    dtw : int
        Size of window.

    dis_step : int
        Discretization step in signal.

    Returns
    -------
    smoothed_signal : array_like
        return smoothed signal

    """
    if dtw < dis_step:
        raise ValueError('Window len is 0. Choose dtw > dis_step')

    pointNumForWind = int(dtw // dis_step)
    window = np.blackman(2 * pointNumForWind)
    new_signal = signal_raw.copy()
    for ind, signal in enumerate(new_signal):
        signal[:pointNumForWind] = signal[:pointNumForWind] * window[:pointNumForWind]
        signal[np.size(signal) - pointNumForWind:] = signal[np.size(signal) - pointNumForWind:] * window[
                                                                                                  pointNumForWind:2 * pointNumForWind]

    print('Points in signal to smooth - ', 2 * pointNumForWind)
    return new_signal


def spectrum(signal, calc_long=True):
    """

    Calculate spectrum of signal with fourier transform.

    Parameters
    ----------
    signal : array_like
        Signal to calculate spectrum of.

    calc_long : bool
        if True add zeros to signal to smooth spectrum.

    Returns
    -------
    signal_spectrum : array_like
        return spectrum of signal.

    """
    if calc_long:
        signal_spectrum = np.zeros((signal.shape[0], 2500))
        for ind, sig in enumerate(signal):
            signal_spectrum[ind] = np.append(sig, np.zeros((2500 - len(sig))))
            signal_spectrum[ind] = np.abs(np.fft.fft(signal_spectrum[ind]) - np.mean(signal_spectrum[ind]))

    else:
        signal_spectrum = np.zeros((signal.shape[0], len(signal[0])))
        for ind, sig in enumerate(signal):
            signal_spectrum[ind] = np.abs(np.fft.fft(sig) - np.mean(sig))

    return signal_spectrum[:, :1250]


def Q_calculate(amplitudes_origin, amplitudes_attenuated, f1, f2, t, dis_step):
    """

    Calculate attenuation parameter Q with spectrum amplitudes.

    Parameters
    ----------
    amplitudes_origin : array_like
        Amplitde spectrum of initial signal.

    amplitudes_attenuated : array_like
        Amplitde spectrum of attenuated signal.

    f1 : int
        Left band for frequency

    f2 : int
        Right band for frequency

    t : int
        Time between first wave and second wave.

    dis_step : int
        Discretization step in signal.

    Returns
    -------
    Q : array_like
        return quality parameter for pairs of signals.

    """
    Q = np.zeros(len(amplitudes_origin))

    for ind, sig in enumerate(amplitudes_origin):
        frequencies = fftfreq(2500, 1 / (1000 / dis_step))[:1250]
        r = np.log(amplitudes_attenuated[ind] / amplitudes_origin[ind])
        # if ind == 300:
        # return amplitudes_attenuated[ind]
        # Linear approximation
        model = LinearRegression()
        X = frequencies[(frequencies > f1) & (frequencies < f2)]
        y = r[(frequencies > f1) & (frequencies < f2)]
        reg_coef = model.fit(X.reshape(-1, 1), y).coef_

        Q[ind] = -reg_coef / (t * np.pi)
    return Q
