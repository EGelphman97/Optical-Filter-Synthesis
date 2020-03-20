"""
Eric Gelphman
UC San Diego Department of Electrical and Computer Engineering
Last Updated March 20, 2020 Version 1.1.2

Driver program for lattice filter synthesis, if __name__ == "__main__"
is here

Required Packages:
-sys
-designFilter
-latticeFilterSynthesis
-numpy
-scipy.signal: freqz
-matplotlib.pyplot
"""

import sys
import designFilter as dF
import latticeFilterSynthesis as lfs
import numpy as np
from scipy.signal import freqz
import matplotlib.pyplot as plt

    
def calcUnitDelayLength(lamda_start, lamda_end, n_g):
    """
    Function to calculate the unit delay length of the filter
    
    Parameters: lamda_start: smallest wavelength in range of interest, in nm
                  lamda_end: longest wavelength in rane og interest, in nm
                        n_g: group index
                        
    Return:             L_U: Unit delay length L_U in um
    """
    denom = 2.0*((1.0/lamda_start)-(1.0/lamda_end))*n_g
    L_U = 1.0/denom
    L_U = L_U*(1E-03)
    return L_U


def determineWavelengthEndpoints(center_wvlengths, band_wvlength):
    """
    Function to obtain the endpoints of the wavelength interval of interest given the center wavelengths
    
    Parameters: center_wavelengths: ndarray of center wavelengths (for now, at most 2) in format of
                                    [longer wavelength, shorter wavelength]
                     band_wvlength: Length, in nm, of pass/stop bands
                     
    Return: nd array [lambda0, lambda1] where
          lambda0: longest wavelength (smallest frequency) in interval of interest
          lambda1: smallest wavelength (largest frequency) in interval of interest
    """
    lambda0 = 1565#nm
    lambda1 = 1520#nm
    PI = np.pi
    n_bands = center_wvlengths.size
    if n_bands == 1:
        lambda0 = center_wvlengths[0] + 23
        lambda1 = center_wvlengths[0] - 23
    else:
        lambda0 = center_wvlengths[0] + 12
        lambda1 = center_wvlengths[1] - 12
        omega_1 = dF.convertToNormalizedFrequency(lambda0, lambda1, center_wvlengths[0]+(band_wvlength/2.0))
        omega_2 = dF.convertToNormalizedFrequency(lambda0, lambda1, center_wvlengths[1]-(band_wvlength/2.0))
        while omega_1 < 0.05*PI or omega_2 > 0.95*PI:
            lambda0 = lambda0 + 2
            lambda1 = lambda1 - 2
            omega_1 = dF.convertToNormalizedFrequency(lambda0, lambda1, center_wvlengths[0])
            omega_2 = dF.convertToNormalizedFrequency(lambda0, lambda1, center_wvlengths[1])
    return np.array([lambda0, lambda1])        


#Function to recieve the necessary parameters from an input file to design the filter
def receiveFilterParametersFile(filename):
    """
    Function to recieve the necessary parameters from an input file to design the filter

    Parameters: filename: Name of input file

    Return: Formatted list params containing needed values:
            params[0]: Group index n_g
            params[1]: l_c in um
            params[2]: l_c in um
            params[3]: L_2 in um
            params[4]: 'Pass' or 'Stop', indicating filter type
            params[5]: Number of pass or stop bands
            params[6]: First (longer) center wavelength
            params[7]: Second (shorter) center wavelength
            params[8]: Pass/Stop band width, in nm
            params[9]: Max. stopband attenuation, in dB
            params[10]: Filter order
    """
    file1 = open(filename, 'r+')
    params = []
    text = file1.readlines()
    if len(text) < 18:
        print("Error! Not Enough Input Parameters!")
        exit()
    params.append(float(text[9]))#Group index n_g
    line10 = text[10].strip('\n').split(',')
    params.append(float(line10[0]))#l_c
    params.append(float(line10[1]))#l_end
    params.append(float(text[11]))#L_2
    line12 = text[12].strip()
    params.append(line12.strip('%\n'))#Pass or Stop, indicating filter type
    num_bands = int(text[13])#Number of bands (1 or 2 for now)
    params.append(num_bands)
    if num_bands == 1:
        params.append(float(text[14]))
    elif num_bands == 2:
        line14 = text[14].strip('\n').split(',')
        params.append(float(line14[0]))#First center wavelength
        params.append(float(line14[1]))#Second center wavelength
    params.append(float(text[15]))#Pass/Stop band width (in nm)
    params.append(float(text[16]))#Max. stopband attenuation
    params.append(int(text[17]))#Filter order
    return params



def processFilterParametersFile(params):
    """
    Function to determine necessary values to plug into the filter design functions given the
    input parameters. This is to be called after recieveFilterParamtersFile.

    Ex.: params = recieveFilterParametersFile(filename)
         L_U, lengths, bands, max_order, filter_type, endpoints, atten, center_wvlengths, band_wvlength = processFilterParametersFile(params)
         
    Paramters: params: Formatted list of values read from file
    
    Return:         L_U: Unit delay length in um
                lengths: nd array [lc, lend, L2] these are lengths needed for layout
                  bands: List of passbands of filter
                      N: Order of filter
            filter_type: 'Pass' or 'Stop'
              endpoints: nd array of endpoints of desired wavelength interval [lambda_0, lambda_1] lambda_1
                         corresponds to larger frequency
                  atten: Max. stopband attenuation in dB
       center_wvlengths: nd array [lamda_0, lamda_1] lamda_0 is the longer wavelength, in nm, of centers of pass/stop bands
          band_wvlength: Width, in nm, of pass/stop bands
    """
    PI = np.pi
    lc = params[1]
    lend = params[2]
    L2 = params[3]
    lengths = np.array([lc,lend,L2])
    filter_type = params[4]#Pass or Stop
    n_bands = params[5]
    center_wvlengths = np.zeros(1)#Wavelength, in nm, of centers of pass/stop bands
    N = 0#filter order
    L_U = band_wvlength = atten = 0.0#L_U, pass/stop band wavelength in nanometers, attenuation in dB
    endpoints = np.zeros(2)
    if n_bands == 1:
        band_wvlength = params[7]
        endpoints = determineWavelengthEndpoints(np.array([params[6]]), band_wvlength)
        L_U = calcUnitDelayLength(endpoints[1], endpoints[0], params[0])
        center_wvlengths[0] = params[6]
        atten = params[8]
        N = params[9]
    else:
        band_wvlength = params[8]
        endpoints = determineWavelengthEndpoints(np.array([params[6],params[7]]), band_wvlength)
        L_U = calcUnitDelayLength(endpoints[1], endpoints[0], params[0])
        center_wvlengths = np.array([params[6], params[7]])
        atten = params[9]
        N = params[10]
    bands = dF.obtainBandsFromCenterWvlengths(center_wvlengths, endpoints, band_wvlength, filter_type)
    for ii in range(len(bands)):
        print("Band: [" + str(bands[ii][0]) + "," + str(bands[ii][1]) + "]") 
    return L_U, lengths, bands, N, filter_type, endpoints, atten, center_wvlengths, band_wvlength


def processFilterParametersTable(tableFile):
    """
    Function to process data held in a file that contains a table of the wavelengths and attenuation(in dB) at those wavelengths
    
    Parameters: tableFile: Name of file that holds the table
    
    Return:         L_U: Unit delay length in um
                lengths: nd array [lc, lend, L2] these are lengths needed for layout
                  bands:   List of passbands of filter
                      N: Order of filter
            filter_type: 'Pass' or 'Stop'
              endpoints: nd array of endpoints of desired wavelength interval [lambda_0, lambda_1] lambda_1
                         corresponds to larger frequency
                  atten: Max. stopband attenuation in dB
       center_wvlengths: nd array [lamda_0, lamda_1] lamda_0 is the longer wavelength, in nm, of centers of pass/stop bands
          band_wvlength: Width, in nm, of pass/stop bands
    """
    file1 = open(tableFile, 'r+')
    text = file1.readlines()
    n_g = float(text[0])
    line1 = text[1].strip('\n').split(',')
    lc = float(line1[0])
    lend = float(line1[1])
    L2 = float(line1[2])
    lengths = np.array([lc, lend, L2])
    filter_type = text[2].strip('\n')
    num_bands = int(text[3])
    center_wvlengths = np.array([0.0])
    if num_bands == 1:
        center_wvlengths[0] = float(text[4])
    elif num_bands == 2:
        vals = text[4].strip('\n').split(',')
        center_wvlengths = np.array([float(vals[0]), float(vals[1])])   
    band_wvlength = float(text[5])
    atten = float(text[6])
    N = int(text[7])
    lambda1 = float(text[len(text)-1].strip('\n').split(',')[0])#Longest wavelength/lowest frequency
    lambda0 = float(text[8].strip('\n').split(',')[0])#Shortest wavelength/highest frequency
    table = np.zeros((len(text)-8,2))
    last_idx = len(text)-1
    for ii in range(8, len(text)):
        line = text[ii].strip('\n').split(',')#Format is wavlength(nm),magnitude(dB)
        table[last_idx-ii][0] = dF.convertToNormalizedFrequency(lambda1, lambda0, float(line[0]))
        table[last_idx-ii][1] = 10**(float(line[1])/20.0)#Convert from dB to linear scale
    endpoints = np.array([lambda1, lambda0])
    bands = dF.obtainBandsFromCenterWvlengths(center_wvlengths, np.array([lambda1, lambda0]), band_wvlength, filter_type)
    L_U = calcUnitDelayLength(lambda0, lambda1, n_g)
    return L_U, lengths, bands, N, filter_type, endpoints, atten, center_wvlengths, band_wvlength, table

    
def writeLayoutParametersToFile(kappalcs, phis, L_U, L_2, filename, order, method):
    """
    Function to write the layout paramters to a file
    
    Parameters: kappalcs:
                    phis: List of phase terms phi_n for each stage
                     L_U: Unit delay length in um
                     L_2: Length of section that doesn't include delay in the MZI
                filename: Name of output file
                   order: Order of filter
                  method: Method used to design filter 'Kaiser' or 'Parks-McClellan'

    Return: None
    """
    file1 = open(filename, 'r+')#open file
    file1.write("% Filter Design Method: " + method + " Filter Order: " + str(order)+ "\n")
    file1.write("% Unit Delay Length: " + str(L_U) + " um" + "\n")
    alpha = 1E-04 # db per um
    gamma = 10**((-alpha*L_2)/20.0)
    file1.write("% Coupler Length L_2: " + str(L_2) + " um" + "\n")
    file1.write("% Loss Coefficient Gamma per Stage: " + str(gamma) + "\n")
    for ii in range(len(kappalcs)):#Write parameters to file
        if ii != 0:
            file1.write(f"{kappalcs[ii][0]:.6e},{kappalcs[ii][1]:.6e},{phis[ii-1]:6f},\n")
        else:
            file1.write(f"{kappalcs[ii][0]:.6e},{kappalcs[ii][1]:.6e},\n")
    file1.close()#close file


def graphTLambda(A_N, band_wvlength, endpoints, center_wvlengths, atten, filter_type):
    """
    Function to graph the transmission of the filter as a function of wavelength

    Parameters: A_N: coefficient array of polynomial A_N(z) that is the transfer function of the filter
          endpoints: endpoints of wavelength interval of interest

    Return: None
"""
    PI = np.pi
    w, h = freqz(np.flip(A_N))#Need to flip according to numpy/scipy documentation, remember the coef. of term with
    #highest power of z^-1 occupies index 0 of array-see ECE 161B notes
    c = 3.0E8
    f1 = c/endpoints[0]
    f0 = c/endpoints[1]
    wvlength = np.zeros(w.size)
    for ii in range(w.size):
        denom = (1.0/PI)*(w[ii]*(f1-f0)) + f0
        wvlength[ii] = c/denom  
    plt.title('MA/FIR filter frequency response')
    plt.plot(np.flip(wvlength), 20*np.log10(abs(h)), color='b', label='Realized Frequency Response')
    
    #Build plot of "Ideal" T(lambda)
    band_edges = []
    gain_levels = []
    x_min = []
    x_max = []
    y_min = []
    y_max = []
    #Region before first pass/stop band
    x_min.append(endpoints[1])
    x_max.append(center_wvlengths[center_wvlengths.size-1]-(band_wvlength/2.0))
    if filter_type == 'Pass':
        gain_levels.append(-1*atten)
    else:
        gain_levels.append(0)
    #Pass/Stop Bands
    for ii in range(center_wvlengths.size):
        band_edges.append(center_wvlengths[ii]-(band_wvlength/2.0))
        x_min.append(center_wvlengths[ii]-(band_wvlength/2.0))
        band_edges.append(center_wvlengths[ii]+(band_wvlength/2.0))
        x_max.append(center_wvlengths[ii]+(band_wvlength/2.0))
        if filter_type == 'Pass':
            gain_levels.append(0)
            if ii != 0:
                x_min.append(center_wvlengths[ii]+(band_wvlength/2.0))
                gain_levels.append(-1*atten)
                x_max.append(center_wvlengths[ii-1]-(band_wvlength/2.0))       
        else:
            gain_levels.append(-1*atten)
            if ii != 0:
                x_min.append(center_wvlengths[ii]+(band_wvlength/2.0))
                gain_levels.append(0)
                x_max.append(center_wvlengths[ii-1]-(band_wvlength/2.0))       
        y_min.append(-1*atten)
        y_min.append(-1*atten)
        y_max.append(0)
        y_max.append(0)
    #Region after pass/stop bands
    x_min.append(center_wvlengths[0]+(band_wvlength/2.0))
    x_max.append(endpoints[0])
    if filter_type == 'Pass':
        gain_levels.append(-1*atten)
    else:
        gain_levels.append(0)
    plt.vlines(band_edges, y_min, y_max, colors='g', label='Ideal Frequency Response')
    plt.hlines(gain_levels, x_min, x_max, colors ='g')
    plt.legend()
    plt.ylabel('Amplitude [dB]', color='b')
    plt.xlabel('Wavelength [um]')
    plt.show()


def generateTableFile(filename):
    file1 = open(filename, 'r+')#open file
    file1.write("4.0\n")#n_g
    file1.write("2.0,2.1,5.0\n")#Lengths for layout
    file1.write("Stop\n")
    file1.write("2\n")
    file1.write("1547,1537\n")
    file1.write("2.0\n")
    file1.write("50\n")
    file1.write("76\n")
    wv_lengths = np.linspace(1527, 1557, 1024)
    for ii in range(wv_lengths.size):
        if (wv_lengths[ii] >= 1536 and wv_lengths[ii] <= 1538):
            file1.write(str(wv_lengths[ii])+","+str(-60.0)+"\n")
        elif wv_lengths[ii] >= 1546 and wv_lengths[ii] <= 1548:
            file1.write(str(wv_lengths[ii])+","+str(-60.0)+"\n")
        else: 
            file1.write(str(wv_lengths[ii])+","+str(0.0)+"\n")
        
    
def main():
    num_args = len(sys.argv)
    if num_args < 3 or num_args > 4:
        print("Error! Not enough or too many args, format should by synthesisDriver.py [-l or -t] [fileName] [Kaiser or Parks-McClellan if l]")
        print("For a total of 3 args if t, or 4 args if l")
        exit()
    fileName = sys.argv[2]#Input file name
    L_U = 0.0
    atten = 0.0
    band_wvlength = 0.0
    lengths = np.zeros(3)
    bands = []
    max_order = 0
    filter_Type = ''
    method = ''
    endpoints = np.zeros(2)
    center_wvlengths = np.zeros(1)
    table = []
    if sys.argv[1] == '-t':   
        generateTableFile("filterTable1.txt")
        fileName = "filterTable1.txt"
    if sys.argv[1] == '-l':#Input file is in list format
        params = receiveFilterParametersFile(fileName)
        method = sys.argv[3]
        L_U, lengths, bands, N, filter_type, endpoints, atten, center_wvlengths, band_wvlength = processFilterParametersFile(params)
    elif sys.argv[1] == '-t':#Input file is in table format
        L_U, lengths, bands, N, filter_type, endpoints, atten, center_wvlengths, band_wvlength, table = processFilterParametersTable(fileName)
        method = 'Kaiser'
    bands = dF.obtainBandsFromCenterWvlengths(center_wvlengths, endpoints, band_wvlength, filter_type)
    #Design the filter
    A_N = np.zeros(1)
    t_width = 0.0
    if len(table) > 0:
        A_N, t_width = dF.designFIRFilter(N, bands, atten, filter_type, method, table)
    else:
        A_N, t_width = dF.designFIRFilter(N, bands, atten, filter_type, method)
    if A_N[0] > 0.0:
        A_N = -1.0*A_N#Synthesis algorithm works best if leading coefficient is negative
    A_z = np.poly1d(A_N)
    print(A_z)
    if filter_type == 'Stop':
        passbands = dF.obtainPassbandsFromStopbands(bands, t_width)
        bands = passbands
    print(lengths)
    fac = lengths[2]*1E-04#Convert alpha to dB/um as L_2 is in um
    gamma = 10**(-1*fac/20)#Loss coefficient per stage, depends on length L_2 of MZI stage
    kappalcs, phis, B_N = lfs.synthesizeFIRLattice(A_N, N, gamma)
    writeLayoutParametersToFile(kappalcs, phis, L_U, 25.0, "layoutParameters.txt", N, method)
    graphTLambda(A_N, band_wvlength, endpoints, center_wvlengths, atten, filter_type)

if __name__ == '__main__':
    main()
