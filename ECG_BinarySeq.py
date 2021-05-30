#for ecg QRS detection with NeuroKit Framework
#https://github.com/neuropsychology/NeuroKit
import neurokit2 as nk
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pprint
#%matplotlib inline   //use this if in jupyter notebook
#plt.rcParams['figure.figsize'] = [8,5]


'''
   Get_ECG()
   The function loads an ECG signal utilizing the NeuroKit framework.
   -> From https://neurokit2.readthedocs.io/en/latest/_modules/neurokit2/ecg/ecg_peaks.html#data

    Parameters
    ----------
    dataset : str
        The name of the dataset. The list and description is
        available `here <https://neurokit2.readthedocs.io/en/master/datasets.html#>`_.

    Returns
    -------
    DataFrame
        The data.

'''
def Get_ECG():
    # Retrieve ECG data from NeuroKit
    ecg_signal = nk.data(dataset="ecg_3000hz")['ECG']
    return ecg_signal

'''
   Get_R_Peaks()

   The function utilizes the NeuroKit framework to return R fiducual peak.
   The method will find R-peaks in an ECG Signal and return a DataFrame
      the length of the input singal with the R-peaks marked as "1"
   
   -> From https://neurokit2.readthedocs.io/en/latest/_modules/neurokit2/ecg/ecg_peaks.html#ecg_peaks

    Parameters
    ----------
    ecg_cleaned : Union[list, np.array, pd.Series]
        The cleaned ECG channel as returned by `ecg_clean()`.
    sampling_rate : int
        The sampling frequency of `ecg_signal` (in Hz, i.e., samples/second).
        Defaults to 1000.
    method : string
        The algorithm to be used for R-peak detection. Can be one of 'neurokit' (default), 'pamtompkins1985',
        'hamilton2002', 'christov2004', 'gamboa2008', 'elgendi2010', 'engzeemod2012' or 'kalidas2017'.
    correct_artifacts : bool
        Whether or not to identify artifacts as defined by Jukka A. Lipponen & Mika P. Tarvainen (2019):
        A robust algorithm for heart rate variability time series artefact correction using novel beat
        classification, Journal of Medical Engineering & Technology, DOI: 10.1080/03091902.2019.1640306.

    Returns
    -------
    signals : DataFrame
        A DataFrame of same length as the input signal in which occurences of R-peaks marked as "1"
        in a list of zeros with the same length as `ecg_cleaned`. Accessible with the keys "ECG_R_Peaks".
    info : dict
        A dictionary containing additional information, in this case the samples at which R-peaks occur,
        accessible with the key "ECG_R_Peaks", as well as the signals' sampling rate.


    References
    ----------
    - Gamboa, H. (2008). Multi-modal behavioral biometrics based on hci and electrophysiology.
      PhD ThesisUniversidade.

    - W. Zong, T. Heldt, G.B. Moody, and R.G. Mark. An open-source algorithm to detect onset of arterial
      blood pressure pulses. In Computers in Cardiology, 2003, pages 259–262, 2003.

    - Hamilton, Open Source ECG Analysis Software Documentation, E.P.Limited, 2002.

    - Jiapu Pan and Willis J. Tompkins. A Real-Time QRS Detection Algorithm. In: IEEE Transactions on
      Biomedical Engineering BME-32.3 (1985), pp. 230–236.

    - C. Zeelenberg, A single scan algorithm for QRS detection and feature extraction, IEEE Comp. in
      Cardiology, vol. 6, pp. 37-42, 1979

    - A. Lourenco, H. Silva, P. Leite, R. Lourenco and A. Fred, "Real Time Electrocardiogram Segmentation
      for Finger Based ECG Biometrics", BIOSIGNALS 2012, pp. 49-54, 2012.
'''
def Get_R_Peaks(ecg_signal):
    # Extract R-peak locations from ecg signal file
    _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=3000)
    pprint.pprint(rpeaks, width=1)

    # Visualize R-peaks in ECG signal **for Jupyter Notebook
    plot_Rpeak_Signal = nk.events_plot(rpeaks['ECG_R_Peaks'], ecg_signal)
    plot_Rpeak_Signal.savefig("rpeaks_signal", dpi = 300)
    

    # Visual R-peak locations for first 5 5 R-peaks
    plot_Rpeak_Head = nk.events_plot(rpeaks['ECG_R_Peaks'][:5], ecg_signal[:20000])
    plot_Rpeak_Head.savefig("rpeaks_head", dpi = 300)
    return rpeaks
    
'''
   Get_TPQS_Peaks()
   The function utilizes the NeuroKit framework to return PQST fiducual peaks.
   -> From https://neurokit2.readthedocs.io/en/latest/_modules/neurokit2/ecg/ecg_peaks.html#ecg_peaks


    - **Cardiac Cycle**: A typical ECG heartbeat consists of a P wave, a QRS complex and a T wave.
      The P wave represents the wave of depolarization that spreads from the SA-node throughout the atria.
      The QRS complex reflects the rapid depolarization of the right and left ventricles. Since the
      ventricles are the largest part of the heart, in terms of mass, the QRS complex usually has a much
      larger amplitude than the P-wave. The T wave represents the ventricular repolarization of the
      ventricles.On rare occasions, a U wave can be seen following the T wave. The U wave is believed
      to be related to the last remnants of ventricular repolarization.

    Parameters
    ----------
    ecg_cleaned : Union[list, np.array, pd.Series]
        The cleaned ECG channel as returned by `ecg_clean()`.
    rpeaks : Union[list, np.array, pd.Series]
        The samples at which R-peaks occur. Accessible with the key "ECG_R_Peaks" in the info dictionary
        returned by `ecg_findpeaks()`.
    sampling_rate : int
        The sampling frequency of `ecg_signal` (in Hz, i.e., samples/second).
        Defaults to 500.
    method : str
        Can be one of 'peak' (default) for a peak-based method, 'cwt' for continuous wavelet transform
        or 'dwt' for discrete wavelet transform.
    show : bool
        If True, will return a plot to visualizing the delineated waves
        information.
    show_type: str
        The type of delineated waves information showed in the plot.
    check : bool
        Defaults to False.

    Returns
    -------
    waves : dict
        A dictionary containing additional information.
        For derivative method, the dictionary contains the samples at which P-peaks, Q-peaks, S-peaks,
        T-peaks, P-onsets and T-offsets occur, accessible with the key "ECG_P_Peaks", "ECG_Q_Peaks",
        "ECG_S_Peaks", "ECG_T_Peaks", "ECG_P_Onsets", "ECG_T_Offsets" respectively.

        For wavelet methods, the dictionary contains the samples at which P-peaks, T-peaks, P-onsets,
        P-offsets, T-onsets, T-offsets, QRS-onsets and QRS-offsets occur, accessible with the key
        "ECG_P_Peaks", "ECG_T_Peaks", "ECG_P_Onsets", "ECG_P_Offsets", "ECG_T_Onsets", "ECG_T_Offsets",
        "ECG_R_Onsets", "ECG_R_Offsets" respectively.

    signals : DataFrame
        A DataFrame of same length as the input signal in which occurences of
        peaks, onsets and offsets marked as "1" in a list of zeros.

    References
    --------------
    - Martínez, J. P., Almeida, R., Olmos, S., Rocha, A. P., & Laguna, P. (2004). A wavelet-based ECG
      delineator: evaluation on standard databases. IEEE Transactions on biomedical engineering,
      51(4), 570-581.
'''
def Get_TPQS_Peaks(ecg_signal,rpeaks):
    # Delineate ECG signal to get TPQS peaks
    _, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=3000)
    pprint.pprint(waves_peak, width=1)

    # Visualize the T-peaks, P-peaks, Q-peaks and S-peaks
    plot_TPQS_Signal = nk.events_plot([waves_peak['ECG_T_Peaks'],
                       waves_peak['ECG_P_Peaks'],
                       waves_peak['ECG_Q_Peaks'],
                       waves_peak['ECG_S_Peaks']], ecg_signal)
    plot_TPQS_Signal.savefig("TPQS_signal", dpi = 300)
    
    # Zooming into the first 3 R-peaks, with focus on T_peaks, P-peaks, Q-peaks and S-peaks
    plot_TPQS_Head = nk.events_plot([waves_peak['ECG_T_Peaks'][:3],
                       waves_peak['ECG_P_Peaks'][:3],
                       waves_peak['ECG_Q_Peaks'][:3],
                       waves_peak['ECG_S_Peaks'][:3]], ecg_signal[:12500])
    plot_TPQS_Head.savefig("TPQS_head", dpi = 300)

    # Delineate the ECG signal and visualizing all peaks of ECG complexes
    _, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=3000, show=True, show_type='peaks')
    

    # Delineate the ECG signal and visualizing all P-peaks boundaries
    signal_peak, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=3000, show=True, show_type='bounds_P')

    # Delineate the ECG signal and visualizing all T-peaks boundaries
    signal_peaj, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=3000, show=True, show_type='bounds_T')
    return waves_peak
    
   
'''
   main()
   This is the main driver
   Argument:
   Return: 
'''
def main():
    #Retrieve ecg data file
    ecg_signal = Get_ECG();
    rPeaks = Get_R_Peaks(ecg_signal)
    waves_peak = Get_TPQS_Peaks(ecg_signal, rPeaks)

if __name__ == '__main__':
    main()






