'''
   ECG_BinarySeq.py 
   This program reads in a data set from neurokit2.0 and utilizes the 
   Multi-Fiducial Point Binary Sequence Generation - 11 (MFBSG-11) to 
   create a random binary sequence (BS) utilizing the fiducial points.  
   The BS is then written to a file.
   
   This program is used in conjunction with an encryption proof of concept
   written in c#.  

   The code is structured as follows:
            Component 1: Load Data
            Component 2: ECG Wavelet Detection
            Component 3: Binary Sequence Generation
   
   Backlog:
          1) Modify Component 1 to allow use of larger datasets
          2) Change R peaks type to match TPQS peaks type to reduce
                        -> extra functions due to different type
          3) For proof of concept implement theory of fuzzy committment
   
   This program was created as a class project for 538 with COIL partners
   
'''
################################################################################
################################################################################
#                           Dependencies                                       #
################################################################################
################################################################################
#for ecg QRS detection with NeuroKit Framework
#https://github.com/neuropsychology/NeuroKit
import neurokit2 as nk
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import time
#%matplotlib inline   //use this if in jupyter notebook
#plt.rcParams['figure.figsize'] = [8,5]


################################################################################
################################################################################
#                           Component 1                                        #
################################################################################
################################################################################
'''
   Get_ECG()
   The function loads an ECG signal utilizing the NeuroKit framework.
   -> From https://neurokit2.readthedocs.io/en/
                               latest/_modules/neurokit2/ecg/ecg_peaks.html#data

    Parameters for nk.data function
    ----------
    none :
        The name of the datasets available for inline function are found here:
        <https://neurokit2.readthedocs.io/en/master/datasets.html#>

    Returns
    -------
    ecg_signal : The dataframe containing ECG signals.

'''
def Get_ECG():
    # Retrieve ECG data from NeuroKit
    print()
    print()
    print("========          Retrieving ECG Signal Data Set            ========")
    ecg_signal = nk.data(dataset="ecg_3000hz")['ECG']
    return ecg_signal

################################################################################
################################################################################
#                           Component 2                                        #
################################################################################
################################################################################
'''
   The Structure of Component 2:
                    1. Get_R_Peaks
                    2. Get_TPQS_Peaks
'''
################################################################################
################################################################################
'''
   Get_R_Peaks()

   The function utilizes the NeuroKit framework to return the R fiducual peak.
   The method will find R-peaks in an ECG Signal and return a DataFrame
   the length of the input singal with the R-peaks marked as "1"
   
   -> https://neurokit2.readthedocs.io/en/latest/
                                 _modules/neurokit2/ecg/ecg_peaks.html#ecg_peaks

    Parameters
    ----------
    none : 

    Returns
    -------
    rPeaks : a DataFrame with the rPeaks
'''
def Get_R_Peaks(ecg_signal):
    # Extract R-peak locations from ecg signal file
    print("========          Retrieving R Peaks of QRS complex         ========")
    _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=3000)
    
    #Print to console
    for key in rpeaks:
        print(key, ' : ', rpeaks[key])
    print()

    # Visualize R-peaks in ECG signal **for Jupyter Notebook
    plot_Rpeak_Signal = nk.events_plot(rpeaks['ECG_R_Peaks'], ecg_signal)
    plot_Rpeak_Signal.savefig("rpeaks_signal", dpi = 300)
    

    # Visual R-peak locations for first 5 5 R-peaks  **for Jupyter Notebook
    plot_Rpeak_Head = nk.events_plot(rpeaks['ECG_R_Peaks'][:5], ecg_signal[:20000])
    plot_Rpeak_Head.savefig("rpeaks_head", dpi = 300)
    return rpeaks
    
'''
   Get_TPQS_Peaks()
   
   This function uses the DWT algorithm in the NeuroKit2.0 framework to get the 
   TPQS peaks from an ECG signal.
   
   ->  https://neurokit2.readthedocs.io/en/latest/
                                 _modules/neurokit2/ecg/ecg_peaks.html#ecg_peaks


    Parameters
    ----------
    none : 

    Returns
    -------
    waves_dwt_peak : dictionary of TPQS peaks
    waves_peak : dictionary of TPR onsets and offsets
'''
def Get_TPQS_Peaks(ecg_signal,rpeaks):
    # Delineate ECG signal to get TPQS peaks
    print("=======   Using DWT to retrieve TPQS Peaks and On/Offsets   ========")
    dwt_sig, waves_dwt_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=3000, method="dwt", show=True, show_type="all")

    #Print to console
    for key in waves_dwt_peak:
        print(key, ' : ', waves_dwt_peak[key])
    
    def_sig, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=3000, show_type="peaks")

    # Visualize the T-peaks, P-peaks, Q-peaks and S-peaks  **for JupyterNotebook
    plot_TPQS_Signal = nk.events_plot([waves_peak['ECG_T_Peaks'],
                       waves_peak['ECG_P_Peaks'],
                       waves_peak['ECG_Q_Peaks'],
                       waves_peak['ECG_S_Peaks']], ecg_signal)
    plot_TPQS_Signal.savefig("TPQS_signal", dpi = 300)
    
    # Zooming into the first 3 R-peaks, with focus on
    # T_peaks, P-peaks, Q-peaks and S-peaks  **for JupyterNotebook
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

    #Print to Console
    for key in waves_peak:
        if(key == 'ECG_Q_Peaks' or key =='ECG_S_Peaks'):
           print(key, ' : ', waves_peak[key])
    print()
    
    return waves_dwt_peak, waves_peak
   
################################################################################
################################################################################
#                           Component 3                                        #
################################################################################
################################################################################
'''
   The Structure of Component 2:
                    1. Get_BinaryFeatures -> helper Get_rPeak_Avg
                                          -> helper Get_rPeak_STD
                                          -> Get_DWT_Avg
                                          -> Get_stdRX
                    2. Get_rPeak_Avg
                    3. Get_rPeak_STD
                    4. Get_DWT_Avg
                    5. Get_stdRX
                    6. Get_BS            -> helper Get_RX
                                         -> Get_rPeak
                                         -> Get_DWTs
                                         -> helper Get_RXi
                                         -> helper extract_K_Bits
                    7. Get_RX
                    8. Get_rPeak
                    9. Get_DWTs
                    10. Get_RXi
                    11. extract_K_Bits
'''
################################################################################
################################################################################
'''
    Get_BinaryFeatures(rPeaks, waves_dwt_peak, waves_peak))
    This is the first step in creating the binary features.  A binary feature
    is a vector of different fiducial point intervals.  This function uses
    helper functions to get the averages and standard deviations of each 
    fiducial feature.  These will be used later on for unique bit extraction
    
    Parameters
    ----------
    rPeaks : dictionary of R peaks
    waves_dwt_peak : dictionary of PTQS peaks
    waves_peak : dictionary of PTR onsets and offset

    Returns
    -------
    dictMean : dictionary of fiducial peak averages in the set
    dictSTD : dictionary of standard deviations of fiducial points in the set
'''
def Get_BinaryFeatures(rPeaks, waves_dwt_peak, waves_peak):
    print("======= Creating Dictionaries of Mean / STD Binary Features ========")
    #Get R peak and STD
    avgR = Get_rPeak_Avg(rPeaks)
    stdR = Get_rPeak_STD(rPeaks)
    
    #Get PTQS 
    avgSpeak, avgQpeak, avgPoff, avgPons, avgPpeak,avgToff, avgTons, avgTpeak, avgRoff, avgRons = Get_DWT_Avg(waves_dwt_peak, waves_peak)
    stdSpeak, stdQpeak, stdPoff, stdPons, stdPpeak, stdToff, stdTons, stdTpeak, stdRoff, stdRons = Get_stdRX(waves_dwt_peak, waves_peak)
    dictMean = {"R_Peak" : avgR,"R_Onset": avgRons, "R_Offset": avgRoff, "T_Peak":avgTpeak, "T_Onset":avgTons,"T_Offset":avgToff, "P_Peak":avgPpeak,"P_Onset":avgPons, "P_Offset":avgPoff,"Q_Peak":avgQpeak,"S_Peak":avgSpeak}
    dictSTD = {"R_Peak" : stdR,"R_Onset": stdRons, "R_Offset": stdRoff, "T_Peak":stdTpeak, "T_Onset":stdTons,"T_Offset":stdToff, "P_Peak":stdPpeak,"P_Onset":stdPons, "P_Offset":stdPoff,"Q_Peak":stdQpeak,"S_Peak":stdSpeak}
    return dictMean, dictSTD
 

'''
    Get_rPeak_Avg(r_peaks)
    
    This function will get the average for value for all the R peaks in the set.
    
    Parameters
    ----------
    rPeaks : dictionary of R peaks

    Returns
    -------
    avgR_int : the average value for R in the set
'''
def Get_rPeak_Avg(r_peaks):
    
    #Create list of values
    rList = r_peaks.values()
    
    #Create vars to hold sum and number of iterations
    avgR = 0
    counter = 0
    
    #loop through each item in list, add to sum, keep count of items
    for x in rList:
        for y in x:
            avgR = avgR + y
            counter = counter + 1

    #divide sum by iterations and ensure it is an int
    avgR = avgR / counter
    avgR_int = int(avgR)
    
    return avgR_int

'''
    Get_rPeak_STD(r_peaks)

    This function converts a dataframe to a list to an array TODO: clean this up
    Next, gets the standard deviation in the array and the multiplies by 
    log2, takes the absolute value and casts it to an int.
    
    Parameters
    ----------
    rPeaks : dictionary of R peaks

    Returns
    -------
    abs_log2_std : an integer of the standard deviation multiplied by log2
                  -> with an absolute value cast to integer
'''
def Get_rPeak_STD(r_peaks):

    #Make a list of the values
    rList = r_peaks.values()

    # Vars to hold length of the list
    length = 0

    #loop through list and sum values, keep track of length
    for x in rList:
        for y in x:
            length = length + 1

    #make an empty arr to hold values from list
    arr = np.empty([1,length])

    #make an array from list
    for i in rList:
        index = 0
        for j in i:
            arr[0,index] = j
            index = index + 1

    #now use np standard deviation function
    std = np.std(arr)

    #cast STD to int after absolute value and multiplied by log 2
    abs_log2_std = int(abs(math.log2(std)))
    
    return abs_log2_std
'''
    Get_DWT_Avg(dwt_waves)
    
    This function will go through the dictionaries that hold the fiducial peaks
    and return their average values.
    
    Parameters
    ----------
    dwt_waves : dictionary of TP waves and onsets
    QS_peaks : dictionary of QS peaks

    Returns
    -------
    all means for TPQS peaks, onset, and offsets 
                           -> as applicable (TPR for onset/offset)
'''
def Get_DWT_Avg(dwt_waves, QS_peaks):
    print("=======               Getting Mean of RX                     =======")
    
    #Go through each fiducial point, get the mean, cast it to an int, return int
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
    Get_stdRX(BF)
    
    This function will go through the dictionaries that hold the fiducial peaks
    TPQS and return their standard deviation values.
    
    Parameters
    ----------
    dwt_waves : dictionary of TP waves and onsets
    QS_peaks : dictionary of QS peaks

    Returns
    -------
    all STDs for TPQS peaks, onset, and offsets as 
                               -> applicable (TPR for onset/offset)
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
    Get_BS(rPeaks, waves_dwt_peak, waves_peak))
    
    This function creates a BS and writes it to file.  It uses Get_RX,
    and extract_K_Bits as helper functions

    Parameters
    ----------
    rPeaks : dictionary of R peaks
    waves_dwt_peak : dictionary of TPQS peaks 
    waves_peak : dictionary of TPR onsets and offsets
    stdBF : dictionary of standard deviations of fiducial points
    meanRX : the meanRX
    BF : dictionary of all binary features (RR, RQ.... etc)
    signal : which RX is desired to be created into a BS

    Returns
    -------
    none : 
'''
def Get_BS(rPeaks, waves_dwt_peak, waves_peak, stdBF, BF, signal):
    
    # Vars to hold desired bits / can set to what you want, and empty string
    twofiftysix = 0;
    BS = ''
        
    #get bits while length is less than desired number in twfiftysix
    while twofiftysix < 130:
       
       #Get the BS for RX of ith wave
       RXi = Get_RX(rPeaks, waves_dwt_peak, waves_peak, stdBF, BF, signal)
       
       #concatenate values 
       BS = BS + RXi
       
       #set length of gaurd variable
       twofiftysix = len(BS)
        
    #cast BS to int and extract the bits    
    num = int(BS)
    BS = extract_K_Bits(num,128,2)

    #Console output
    print("++++++++++++++  Final BS"+str(signal)+" being writ to file +++++++++++++++++++++++")
    print(BS)
    print("The length of BS is: ", end='')
    print(len(BS))
    print()
    print("Writing to file")

    #Write the bytes to a file
    byte = bytes(BS,"utf8")
    fileName = "ecg" + str(signal) + ".bin"
    file = open(fileName, "wb")
    file.write(byte)
    file.flush()
    file.close()

    #Complete let user know
    print("Operation Complete for: " + str(signal))

'''
    Get_RX(rPeaks, waves_dwt_peak, waves_peak))
    
    This function goes through all the binary features and their vectors,
    and returns the binary sequence for the ith signal
    
    Parameters
    ----------
    rPeaks : dictionary of R peaks
    waves_dwt_peak : peaks of TPQS
    waves_peak : offsets and onsets
    stdBF : dictionary of standard deviations for each BF
    BF : the binary feature vectors
    i : the ith signal 
    
    Returns
    -------
    RXi : a string of all the concanated bits
'''
def Get_RX(rPeaks, waves_dwt_peak, waves_peak, stdBF, BF, i):
    #print("=======                   Getting RX                        ========")

    #Get the value for the R peak, and all the other fiducial points
    Ri = Get_rPeak(rPeaks,BF, i)
    Speak, Qpeak, Poff, Pons, Ppeak, Toff, Tons, Tpeak, Roff, Rons = Get_DWTs(waves_dwt_peak, waves_peak,BF, i)
    #make dictionary of each fiduical point, for the ith signal, and get BS
    dictRX = {"R_Peak" : Ri,"R_Onset": Rons, "R_Offset": Roff, "T_Peak":Tpeak, "T_Onset":Tons,"T_Offset":Toff, "P_Peak":Ppeak,"P_Onset":Pons, "P_Offset":Poff,"Q_Peak":Qpeak,"S_Peak":Speak}
    RXi = Get_RXi(dictRX, stdBF)
    
    return RXi

'''
    Get_rPeak(r_peaks)
    
    Gets the current R peak and the mean R peak - subtracts mean from curr
    and returns the value
    
    Parameters
    ----------
    rPeaks : dictionary of R peaks
    BF : dictionary of binary features
    idx : the signal to look for
    
    Returns
    -------
    rpeak : the r peak of the ith signal minus the mean of all r peaks
'''
def Get_rPeak(r_peaks,BF,idx):

    #make a list, hold sum, and length of list
    rList = r_peaks.values()
    avgR = 0
    length = 0

    #loop through list to get length
    for x in rList:
        for y in x:
            avgR = avgR + y
            length = length + 1
    arr = np.empty([1,length])

    #make array from list
    for i in rList:
        index = 0
        for j in i:
            arr[0,index] = j
            index = index + 1
    
    #get the value of the ith r peak
    rpeak = int(arr[0,idx])

    #get the mean of r peaks
    rMean = BF["R_Peak"]

    #remove mean from rpeak and return it
    rpeak = rpeak - rMean
    
    return rpeak

'''
    Get_DWTs(dwt_waves,i)
    Gets the current PTQS peaks and PTR onsets and offsets and the correspoding
    means.  Then subtracts mean from curr
    and returns the values
    
    Parameters
    ----------
    dwt_waves : PT peaks, PTR onsets and offsets
    QS_peaks : QS peaks
    BF : all the binary features
    i : the ith signal
    
    Returns
    -------
    all peaks (sans R), onsets, and offsets after removing the mean
'''
def Get_DWTs(dwt_waves, QS_peaks,BF, i):
    print("=======                 Getting RX" +str(i)+"                          =======")

    #Get each fiducial point, get the mean, make it absolute remove mean return int
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
    Get_RXi(BF,std)
    
    This function creates a binary feature vector for RR, RQ, RS, RT, RP,
       -> RRon, RRoff, RTon, RToff, RPon, RPoff.  The vector is the number
       value of the difference between the fiducial points.

    Parameters
    ----------
    Binary_Features : dictionary of all binary features (RR, RQ.... etc)
    std : dictionary of each BFs standard deviation

    Returns
    -------
    RXi_Str : a string of all the concanated bits
'''
def Get_RXi(Binary_Features, std):
    #print("=======                  Retrieving Mean BF                  =======")
    
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
    #Increased STD by mulitple of 10 to achieve 1 IPI cycle for 128 bit BSes
    rr_bin = extract_K_Bits(abs(RR),rSTD*10,2)
    rro_bin = extract_K_Bits(abs(Rro),roSTD*10,2)
    rrof_bin = extract_K_Bits(abs(Rrof),rofSTD*10,2)
    rq_bin = extract_K_Bits(abs(RQ),qSTD*10,2)
    rs_bin = extract_K_Bits(abs(RS),sSTD*10,2)
    rp_bin = extract_K_Bits(abs(RP),pSTD*10,2)
    rpo_bin = extract_K_Bits(abs(Rpo),poSTD*10,2)
    rpof_bin = extract_K_Bits(abs(Rpof),pofSTD*10,2)
    rt_bin = extract_K_Bits(abs(RT),tSTD*10,2)
    rto_bin = extract_K_Bits(abs(Rto),toSTD*10,2)
    rtof_bin = extract_K_Bits(abs(Rtof),tofSTD*10,2)

    #concatenate the bits
    RXi_Str = rr_bin + rro_bin + rrof_bin + rq_bin + rs_bin + rp_bin + rpo_bin + rpof_bin + rt_bin + rto_bin + rtof_bin
    return RXi_Str

'''
   exctract_K_Bits(num, k, p)
    takes in RX num converts to binary drops the least significant bit 
       -> for best matching returns only the std of RX in binary
    Thanks to geeksforgreeks for algorithm
   
    Parameters
    ----------
    num : the number to convert to bits
    k : the number of bits desired
    p : the offset (to skip lsb)
    
    Returns
    -------
    kBit_SubString : a string of k bits of num with LSB dropped 
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
    main()
    
    This is the main driver.  It has three components.  First it opens an 
    ecg data source.  Then it conducts an ECG wavelet detection process.  Then
    it uses that ECG wavelet detection output to create unique binary sequence
    for each wavelet.
    
    Parameters
    ----------
    none : 
    
    Returns
    -------
    none :
'''
def main():
    
    ############################################################################
    #                       Component 1                                        #
    #Retrieve ecg data file
    ecg_signal = Get_ECG();

    ############################################################################
    #                       Component 2                                        #
    #Get the R Peaks
    rPeaks = Get_R_Peaks(ecg_signal)

    #Use DWT to get TPQS peaks and all On/Offsets
    waves_dwt_peak, waves_peak = Get_TPQS_Peaks(ecg_signal, rPeaks)

    ############################################################################
    #                       Component 3                                        #
    #Get the Mean, Absolut Value * Log2 of STD rounded to Int for each BF
    BF, stdBF = Get_BinaryFeatures(rPeaks, waves_dwt_peak, waves_peak)

    #Print values
    print("============ Returning Dictionaries with means and stds ============")
    for key in BF:
        print(key, ' Mean : ', BF[key])
    print()
    for key in stdBF:
        print(key, ' ABS LOG2 STD as INT : ', stdBF[key])
    print()
    
    #Loop through file for as many BSes that you want, less than file length
    # current file is 15 signals long
    signal = 1
    while signal <=10:
        tic = time.perf_counter()
        Get_BS(rPeaks, waves_dwt_peak, waves_peak, stdBF, BF, signal)
        toc = time.perf_counter()
        print(f"Computation Time for BS creation is {toc - tic:0.4f} seconds")
        signal = signal + 1


if __name__ == '__main__':
    main()






