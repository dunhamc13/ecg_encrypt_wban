#for ecg QRS detection with NeuroKit Framework
#https://github.com/neuropsychology/NeuroKit
import neurokit2 as nk
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
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
    print()
    print()
    print("========          Retrieving ECG Signal Data Set            ========")
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
    print("========          Retrieving R Peaks of QRS complex         ========")
    _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=3000)
    for key in rpeaks:
        print(key, ' : ', rpeaks[key])
    print()

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
    print("=======   Using DWT to retrieve TPQS Peaks and On/Offsets   ========")
    dwt_sig, waves_dwt_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=3000, method="dwt", show=True, show_type="all")
    for key in waves_dwt_peak:
        print(key, ' : ', waves_dwt_peak[key])
    
    def_sig, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=3000, show_type="peaks")

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
    for key in waves_peak:
        if(key == 'ECG_Q_Peaks' or key =='ECG_S_Peaks'):
           print(key, ' : ', waves_peak[key])
    print()
    return waves_dwt_peak, waves_peak
   

'''
   Get_DWT_Avg(dwt_waves)
   This will get the average for dwt_waves
   Argument:
   Return: 
'''
def Get_DWT_Avg(dwt_waves, QS_peaks):
    print("=======               Getting Mean of RX                     =======")
    R_Onset = dwt_waves['ECG_R_Onsets']
    avgRons = np.nanmean(R_Onset)
    avgRons_int = int(avgRons)
    
    R_Ofset = dwt_waves['ECG_R_Offsets']
    avgRoff = np.nanmean(R_Ofset)
    avgRoff_int = int(avgRoff)
    
    t_peak = dwt_waves['ECG_T_Peaks']
    avgTpeak = np.nanmean(t_peak)
    t_peak_int = int(avgTpeak)
    
    t_ons = dwt_waves['ECG_T_Onsets']
    avgTons = np.nanmean(t_ons)
    t_ons_int = int(avgTons)
    
    t_off = dwt_waves['ECG_T_Offsets']
    avgToff = np.nanmean(t_off)
    t_off_int = int(avgToff)
    
    p_peak = dwt_waves['ECG_P_Peaks']
    avgPpeak = np.nanmean(p_peak)
    p_peak_int = int(avgPpeak)
    
    p_ons = dwt_waves['ECG_P_Onsets']
    avgPons = np.nanmean(p_ons)
    p_ons_int = int(avgPons)
    
    p_off = dwt_waves['ECG_P_Offsets']
    avgPoff = np.nanmean(p_off)
    p_off_int = int(avgPoff)
    
    q_peak = QS_peaks['ECG_Q_Peaks']
    avgQpeak = np.nanmean(q_peak)
    q_peak_int = int(avgQpeak)
    
    s_peak = QS_peaks['ECG_S_Peaks']
    avgSpeak = np.nanmean(s_peak)
    s_peak_int = int(avgSpeak)
    return s_peak_int, q_peak_int, p_off_int, p_ons_int, p_peak_int, t_off_int, t_ons_int, t_peak_int, avgRoff_int, avgRons_int
    
'''
   Get_rPeak_Avg(r_peaks)
   This will get the average for dwt_waves
   Argument:
   Return: 
'''
def Get_rPeak_Avg(r_peaks):
    rList = r_peaks.values()
    avgR = 0
    counter = 0
    for x in rList:
        for y in x:
            avgR = avgR + y
            counter = counter + 1
    avgR = avgR / counter
    avgR_int = int(avgR)
    return avgR_int
    
'''
   Get_rPeak_STD(r_peaks)
   This will get the std for dwt_waves
   Argument:
   Return: 
'''
def Get_rPeak_STD(r_peaks):
    rList = r_peaks.values()
    avgR = 0
    length = 0
    for x in rList:
        for y in x:
            length = length + 1

    arr = np.empty([1,length])
    for i in rList:
        index = 0
        for j in i:
            arr[0,index] = j
            index = index + 1
    std = np.std(arr)
    abs_log2_std = int(abs(math.log2(std)))
    return abs_log2_std
'''
   Get_BinaryFeatures(rPeaks, waves_dwt_peak, waves_peak))
   This is the main driver
   Argument:
   Return: 
'''
def Get_BinaryFeatures(rPeaks, waves_dwt_peak, waves_peak):
    print("======= Creating Dictionaries of Mean / STD Binary Features ========")
    avgR = Get_rPeak_Avg(rPeaks)
    stdR = Get_rPeak_STD(rPeaks)
    avgSpeak, avgQpeak, avgPoff, avgPons, avgPpeak,avgToff, avgTons, avgTpeak, avgRoff, avgRons = Get_DWT_Avg(waves_dwt_peak, waves_peak)
    stdSpeak, stdQpeak, stdPoff, stdPons, stdPpeak, stdToff, stdTons, stdTpeak, stdRoff, stdRons = Get_stdRX(waves_dwt_peak, waves_peak)
    dictMean = {"R_Peak" : avgR,"R_Onset": avgRons, "R_Offset": avgRoff, "T_Peak":avgTpeak, "T_Onset":avgTons,"T_Offset":avgToff, "P_Peak":avgPpeak,"P_Onset":avgPons, "P_Offset":avgPoff,"Q_Peak":avgQpeak,"S_Peak":avgSpeak}
    dictSTD = {"R_Peak" : stdR,"R_Onset": stdRons, "R_Offset": stdRoff, "T_Peak":stdTpeak, "T_Onset":stdTons,"T_Offset":stdToff, "P_Peak":stdPpeak,"P_Onset":stdPons, "P_Offset":stdPoff,"Q_Peak":stdQpeak,"S_Peak":stdSpeak}
    return dictMean, dictSTD
 
'''
   exctract_K_Bits(num, k, p)
   takes in RX num
   converts to binary
   drops the least significant bit for best matching
   returns only the std of RX in binary
   Thanks to geeksforgreeks for algorithm
'''
def extract_K_Bits(num,k,p):
  
     # convert number into binary first
     binaryNum = bin(num)
  
     # remove first two characters
     binaryNum = binaryNum[2:]
  
     end = len(binaryNum) - p
     start = end - k + 1
  
     # extract k  bit sub-string
     kBit_SubString = binaryNum[start : end+1]
  
     # convert extracted sub-string into decimal again
     return kBit_SubString
'''
   Get_meanRX(BF)
   This function gets the mean RX
   RX is  RR, RQ, RS, RP, RT
   where RQ = R-Q...
   Argument:
   Return: 
'''
def Get_meanRX(Binary_Features, std):
    print("=======                  Retrieving Mean BF                  =======")
    
    #Get the average Peaks
    r = Binary_Features["R_Peak"]
    rSTD = std["R_Peak"]
    q = Binary_Features["Q_Peak"]
    qSTD = std["Q_Peak"]
    s = Binary_Features["S_Peak"]
    sSTD = std["S_Peak"]
    p = Binary_Features["P_Peak"]
    pSTD = std["P_Peak"]
    t = Binary_Features["T_Peak"]
    tSTD = std["T_Peak"]

    #Get the average On/Offsets
    ro = Binary_Features["R_Onset"]
    roSTD = std["R_Onset"]
    rof = Binary_Features["R_Offset"]
    rofSTD = std["R_Offset"]
    po = Binary_Features["P_Onset"]
    poSTD = std["P_Onset"]
    pof = Binary_Features["P_Offset"]
    pofSTD = std["P_Offset"]
    to = Binary_Features["T_Onset"]
    toSTD = std["T_Onset"]
    tof = Binary_Features["T_Offset"]
    tofSTD = std["T_Offset"]

    #Create Binary Features
    RR = r
    Rro = r - ro
    Rrof = r + r - rof
    RQ = r - q
    RS = r + r - s
    RP = r - p
    Rpo = r - po
    Rpof = r - pof
    RT = r - t
    Rto = r - to
    Rtof = r - tof
    
    #Remove the least significant bit, only use abs log2 std to int bits
    rr_bin = extract_K_Bits(RR,rSTD,2)
    rro_bin = extract_K_Bits(Rro,roSTD,2)
    rrof_bin = extract_K_Bits(Rrof,rofSTD,2)
    rq_bin = extract_K_Bits(RQ,qSTD,2)
    rs_bin = extract_K_Bits(RS,sSTD,2)
    rp_bin = extract_K_Bits(RP,pSTD,2)
    rpo_bin = extract_K_Bits(Rpo,poSTD,2)
    rpof_bin = extract_K_Bits(Rpof,pofSTD,2)
    rt_bin = extract_K_Bits(RT,tSTD,2)
    rto_bin = extract_K_Bits(Rto,toSTD,2)
    rtof_bin = extract_K_Bits(Rtof,tofSTD,2)
    
    print(type(rr_bin))
    #concatenate the bits
    meanRX = rr_bin + rro_bin + rrof_bin + rq_bin + rs_bin + rp_bin + rpo_bin + rpof_bin + rt_bin + rto_bin + rtof_bin
    return meanRX

'''
   Get_stdRX(BF)
   This function gets the std of RX
   RX is  RR, RQ, RS, RP, RT
   where RQ = R-Q...
   Argument:
   Return: 
'''
def Get_stdRX(dwt_waves,QS_Peaks):
    print("=======           Getting Standard Deviation of BF           =======")
    R_Onset = dwt_waves['ECG_R_Onsets']
    stdRons = np.nanstd(R_Onset)
    stdRons_int = int(abs(math.log2(stdRons)))
    
    R_Ofset = dwt_waves['ECG_R_Offsets']
    stdRoff = np.nanstd(R_Ofset)
    stdRoff_int = int(abs(math.log2(stdRoff)))
    
    t_peak = dwt_waves['ECG_T_Peaks']
    stdTpeak = np.nanstd(t_peak)
    stdt_peak_int = int(abs(math.log2(stdTpeak)))
    
    t_ons = dwt_waves['ECG_T_Onsets']
    stdTons = np.nanstd(t_ons)
    stdt_ons_int = int(abs(math.log2(stdTons)))
    
    t_off = dwt_waves['ECG_T_Offsets']
    stdToff = np.nanstd(t_off)
    stdt_off_int = int(abs(math.log2(stdToff)))
    
    p_peak = dwt_waves['ECG_P_Peaks']
    stdPpeak = np.nanstd(p_peak)
    stdp_peak_int = int(abs(math.log2(stdPpeak)))
    
    p_ons = dwt_waves['ECG_P_Onsets']
    stdPons = np.nanstd(p_ons)
    stdp_ons_int = int(abs(math.log2(stdPons)))
    
    p_off = dwt_waves['ECG_P_Offsets']
    stdPoff = np.nanstd(p_off)
    stdp_off_int = int(abs(math.log2(stdPoff)))
    
    q_peak = QS_Peaks['ECG_Q_Peaks']
    stdQpeak = np.nanstd(q_peak)
    stdq_peak_int = int(abs(math.log2(stdQpeak)))
    
    s_peak = QS_Peaks['ECG_S_Peaks']
    stdSpeak = np.nanstd(s_peak)
    stds_peak_int = int(abs(math.log2(stdSpeak)))
    return stds_peak_int, stdq_peak_int, stdp_off_int, stdp_ons_int, stdp_peak_int, stdt_off_int, stdt_ons_int, stdt_peak_int, stdRoff_int, stdRons_int



'''
   Get_RXi(BF,std)
   This function gets the mean RX
   RX is  RR, RQ, RS, RP, RT
   where RQ = R-Q...
   Argument:
   Return: 
'''
def Get_RXi(Binary_Features, std, meanRX):
    print("=======                  Retrieving Mean BF                  =======")
    
    #Get the average Peaks
    r = Binary_Features["R_Peak"]
    rSTD = std["R_Peak"]
    q = Binary_Features["Q_Peak"]
    qSTD = std["Q_Peak"]
    s = Binary_Features["S_Peak"]
    sSTD = std["S_Peak"]
    p = Binary_Features["P_Peak"]
    pSTD = std["P_Peak"]
    t = Binary_Features["T_Peak"]
    tSTD = std["T_Peak"]

    #Get the average On/Offsets
    ro = Binary_Features["R_Onset"]
    roSTD = std["R_Onset"]
    rof = Binary_Features["R_Offset"]
    rofSTD = std["R_Offset"]
    po = Binary_Features["P_Onset"]
    poSTD = std["P_Onset"]
    pof = Binary_Features["P_Offset"]
    pofSTD = std["P_Offset"]
    to = Binary_Features["T_Onset"]
    toSTD = std["T_Onset"]
    tof = Binary_Features["T_Offset"]
    tofSTD = std["T_Offset"]

    #Create Binary Features
    RR = r
    Rro = r - ro
    Rrof = r + r - rof
    RQ = r - q
    RS = r + r - s
    RP = r - p
    Rpo = r - po
    Rpof = r - pof
    RT = r - t
    Rto = r - to
    Rtof = r - tof
    
    #Remove the least significant bit, only use abs log2 std to int bits
    rr_bin = extract_K_Bits(RR,rSTD,2)
    rro_bin = extract_K_Bits(Rro,roSTD,2)
    rrof_bin = extract_K_Bits(Rrof,rofSTD,2)
    rq_bin = extract_K_Bits(RQ,qSTD,2)
    rs_bin = extract_K_Bits(RS,sSTD,2)
    rp_bin = extract_K_Bits(RP,pSTD,2)
    rpo_bin = extract_K_Bits(Rpo,poSTD,2)
    rpof_bin = extract_K_Bits(Rpof,pofSTD,2)
    rt_bin = extract_K_Bits(RT,tSTD,2)
    rto_bin = extract_K_Bits(Rto,toSTD,2)
    rtof_bin = extract_K_Bits(Rtof,tofSTD,2)

    #concatenate the bits
    RXi_Str = rr_bin + rro_bin + rrof_bin + rq_bin + rs_bin + rp_bin + rpo_bin + rpof_bin + rt_bin + rto_bin + rtof_bin
    return RXi_Str

'''
   Get_rPeak(r_peaks)
   This will get the r peak of a wave for RX
   Argument:
   Return: 
'''
def Get_rPeak(r_peaks,BF,idx):
    rList = r_peaks.values()
    avgR = 0
    length = 0
    for x in rList:
        for y in x:
            avgR = avgR + y
            length = length + 1
    arr = np.empty([1,length])
    for i in rList:
        index = 0
        for j in i:
            arr[0,index] = j
            index = index + 1
    rpeak = int(arr[0,idx])
    rMean = BF["R_Peak"]
    rpeak = rpeak - rMean
    return rpeak

'''
   Get_DWTs(dwt_waves,i)
   This will get the average for dwt_waves
   Argument:
   Return: 
'''
def Get_DWTs(dwt_waves, QS_peaks,BF, i):
    print("=======                 Getting RXi                          =======")
    R_Onset = dwt_waves['ECG_R_Onsets']
    RonsMean = BF["R_Onset"]
    Rons = abs(int(R_Onset[i]) - RonsMean)
    
    R_Ofset = dwt_waves['ECG_R_Offsets']
    roffMean = BF["R_Offset"]
    Roff = abs(int(R_Ofset[i]) - roffMean)
    
    t_peak = dwt_waves['ECG_T_Peaks']
    tPmean = BF["T_Peak"]
    Tpeak = abs(int(t_peak[i]) - tPmean)
    
    t_ons = dwt_waves['ECG_T_Onsets']
    tonsMean = BF["T_Onset"]
    Tons = abs(int(t_ons[i]) - tonsMean)
    
    t_off = dwt_waves['ECG_T_Offsets']
    toffMean = BF["T_Offset"]
    Toff = abs(int(t_off[i]) - toffMean)
    
    p_peak = dwt_waves['ECG_P_Peaks']
    pmean = BF["P_Peak"]
    Ppeak = abs(int(p_peak[i]) - pmean)
    
    p_ons = dwt_waves['ECG_P_Onsets']
    ponsMean = BF["P_Onset"]
    Pons = abs(int(p_ons[i]) - ponsMean)
    
    p_off = dwt_waves['ECG_P_Offsets']
    poffMean = BF["P_Offset"]
    Poff = abs(int(p_off[i]) - poffMean)
    
    q_peak = QS_peaks['ECG_Q_Peaks']
    qpMean = BF["Q_Peak"]
    Qpeak = abs(int(q_peak[i]) - qpMean)
    
    s_peak = QS_peaks['ECG_S_Peaks']
    spMean = BF["S_Peak"]
    Speak = abs(int(s_peak[i]) - spMean)
    return Speak, Qpeak, Poff, Pons, Ppeak, Toff, Tons, Tpeak, Roff, Rons
    

'''
   Get_RX(rPeaks, waves_dwt_peak, waves_peak))
   This reaturns a single RX
   Argument:
   Return: 
'''
def Get_RX(rPeaks, waves_dwt_peak, waves_peak, stdBF, meanRX, BF, i):
    print("=======                   Getting RX                        ========")
    Ri = Get_rPeak(rPeaks,BF, i)
    Speak, Qpeak, Poff, Pons, Ppeak, Toff, Tons, Tpeak, Roff, Rons = Get_DWTs(waves_dwt_peak, waves_peak,BF, i)
    dictRX = {"R_Peak" : Ri,"R_Onset": Rons, "R_Offset": Roff, "T_Peak":Tpeak, "T_Onset":Tons,"T_Offset":Toff, "P_Peak":Ppeak,"P_Onset":Pons, "P_Offset":Poff,"Q_Peak":Qpeak,"S_Peak":Speak}
    RXi = Get_RXi(dictRX, stdBF, meanRX)
    return RXi
 

'''
   main()
   This is the main driver
   Argument:
   Return: 
'''
def main():
    #Retrieve ecg data file
    ecg_signal = Get_ECG();

    #Get the R Peaks
    rPeaks = Get_R_Peaks(ecg_signal)

    #Use DWT to get TPQS peaks and all On/Offsets
    waves_dwt_peak, waves_peak = Get_TPQS_Peaks(ecg_signal, rPeaks)

    #Get the Mean, Absolut Value * Log2 of STD rounded to Int for each BF
    BF, stdBF = Get_BinaryFeatures(rPeaks, waves_dwt_peak, waves_peak)

    #Print values
    print("Returning Dictionaries with means and stds")
    for key in BF:
        print(key, ' Mean : ', BF[key])
    print()
    for key in stdBF:
        print(key, ' ABS LOG2 STD as INT : ', stdBF[key])
    print()

    #Make meanRX of BF to begin making BS
    meanRX = Get_meanRX(BF,stdBF)
    print("The meanRX is: " + meanRX)
    print("The length of meanRX is: ", end ='')
    print(len(meanRX))

    #Now make BSx...n
    twofiftysix = 0;
    signal = 1
    BS = ''
    while twofiftysix < 258:
       RXi = Get_RX(rPeaks, waves_dwt_peak, waves_peak, stdBF, meanRX, BF, signal)
       print("BS is " + BS)
       print("The RXi is: " + RXi)
       print("The length of BS is: ", end='')
       print(len(BS))
       print("The length of RXi is: ", end ='')
       print(len(RXi))
       BS = BS + RXi
       print("BS is " + BS)
       print("The length of BS is: ", end='')
       print(len(BS))
       print()
       twofiftysix = len(BS)
    num = int(BS)
    BS = extract_K_Bits(num,256,2)
    print("++++++++++++++  Final BS being writ to file ++++++++++++++++++++++++")
    print(BS)
    print("The length of BS is: ", end='')
    print(len(BS))
    print()
    
    print("Writing to file")
    byte = bytes(BS,"utf8")
    byteList = []
    file = open("ecg.bin", "wb")
    file.write(byte)
    file.close()
    print("Operation Complete")

if __name__ == '__main__':
    main()






