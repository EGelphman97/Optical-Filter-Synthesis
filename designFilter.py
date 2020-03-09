#Eric Gelphman
#UC San Diego Department of Electrical and Computer Engineering
#March 8, 2020

"""
Python Script that has methods to obtain the transfer function H(z)
Of digital filters as an array of coefficients. These coefs. will then be passed
to latticeFilterSynthesis.py to determine the lattice parameters
Version 1.1.1
"""

import numpy as np
from scipy.signal import kaiserord, freqz, remez
import scipy.special as ss
import matplotlib.pyplot as plt

"""
Function to obtain the normalized angular frequency omega, 0 <= omega <= pi value from a given (continuous-time) wavelength value lamda
Parameters:  lamda0 = longest wavelength in range of interest, in nm
             lamda1 = shortest wavelength in range of interest, in nm
             lamda = wavelength you want to find normalized frequency for, in nm
Return: Normalized frequency omega, 0 <= omega <= pi
"""
def convertToNormalizedFrequency(lamda0, lamda1, lamda):
    c = 3.0E8#The speed of light
    lamda_0 = lamda0*(1.0E-9)
    lamda_1 = lamda1*(1.0E-9)
    lamda_ = lamda*(1.0E-9)
    f1 = c/lamda_1
    f0 = c/lamda_0
    f = c/lamda_
    omega = np.pi*(f-f0)/(f1-f0)
    return omega

"""
Function to perform a boolean operation on a list of stopbands to get corresponding list of passbands
Parameters: stopbands: list of tuples of corner (normalized) frequencies of stopbands
            t_width:   transition bandwidth (normalized frequency)
Return: list of tuples of passbands with corresponding gains
"""
def obtainPassbandsFromStopbands(stopbands, t_width):
    passbands = []
    if stopbands[0][0] == 0.0:
        if len(stopbands) == 1:
            passbands.append((stopbands[0][1]+t_width, np.pi, 1.0))
    else:
        passbands.append((0.0, stopbands[0][0]-t_width, 1.0))
        if stopbands[0][1] != np.pi and len(stopbands) == 1:
            passbands.append((stopbands[0][1]+t_width, np.pi, 1.0))
    if len(stopbands) == 2:
        passbands.append((stopbands[0][1]+t_width,stopbands[1][0]-t_width, 1.0))
        if stopbands[1][1] != np.pi:
            passbands.append((stopbands[1][1]+t_width, np.pi, 1.0))
    return passbands

"""
Function to obtain the bands(pass or stop), in units of normalized frequency, from the center wavelengths (in nm)
Parameters: center_wvlength: ndarray of center wavelengths, longest wavelength/lowest frequency is in position 0 of array
            band_wvlength:    desired width of pass (or stop) band, in nm
            endpoints: endpoints of wavelength of interest, longest wavelength/lowest frequency is in position 0 of array
            typef:        string that indicates type of filter, this parameter is either "Pass" or "Stop"
Return: List of tuples representing band (normalized) frequency intervals
"""
def obtainBandsFromCenterWvlengths(center_wvlengths, endpoints, band_wvlength, typef):
    bands = []
    for ii in range(center_wvlengths.size):
        lambda_lower = center_wvlengths[ii] - (band_wvlength/2.0)
        lambda_upper = center_wvlengths[ii] + (band_wvlength/2.0)
        omega_lower = convertToNormalizedFrequency(endpoints[0], endpoints[1], lambda_upper)
        omega_upper = convertToNormalizedFrequency(endpoints[0], endpoints[1], lambda_lower)
        if typef == "Pass":
            bands.append((omega_lower, omega_upper, 1.0))
        else:
            bands.append((omega_lower, omega_upper, 0.0))
    return bands

"""
Function to generate the "ideal" Frequency response H(e^jw) of filter for use in my custom Kaiser window function
Parameters: 
                  bands: List of tuples (or any array-like object)
                t_width: Transition width of exponential taper
                      N: ifft length(>= 1024)
            filter_type: 'Pass' or 'Stop'
Return: data: matrix of wavelengths and amplitude at that wavelength in dB
"""
def generateH_ejw(bands, t_width, N, filter_type, plot=False):
    PI = np.pi
    atten_linear = 10**(-300/20.0)
    if filter_type == 'Pass':
        H_ejw = np.zeros(N)#"Ideal" Filter transfer function in frequency domain
    else:
        H_ejw = np.ones(N)
    mag_val = np.log(atten_linear)#Needed to fill H_ejw array - is natural log of stopband amplitude
    omegas = np.linspace(0, PI, N)
    if filter_type == 'Pass':
        for ii in range(N):
            for band in bands:
                #In passband
                if omegas[ii] >= band[0] and omegas[ii] <= band[1]:
                    H_ejw[ii] = 1.0
                #In transition band before passband, do rising exponential taper
                elif omegas[ii] >= band[0] - t_width and omegas[ii] < band[0]:
                    omega1 = band[0] - t_width
                    omega2 = band[0]
                    B = mag_val/(omega1 - omega2)
                    H_ejw[ii] = np.exp(B*(omegas[ii]-omega2))
                #In transition band after passband, do decaying exponential taper
                elif omegas[ii] > band[1] and omegas[ii] <= band[1] + t_width:
                    omega1 = band[1]
                    omega2 = band[1] + t_width
                    B = mag_val/(omega2 - omega1)
                    H_ejw[ii] = np.exp(B*(omegas[ii]-omega1))
    else:
        for ii in range(N):
            for band in bands:
                #If in stopband, 
                if omegas[ii] >= band[0] and omegas[ii] <= band[1]:
                    H_ejw[ii] = atten_linear
                #In transition band before stopband, do decaying exponential taper
                elif omegas[ii] >= band[0] - t_width and omegas[ii] < band[0]:
                    omega1 = band[0] - t_width
                    omega2 = band[0] 
                    B = (atten/(-20.0))/(omega2 - omega1)
                    H_ejw[ii] = np.exp(B*(omegas[ii]-omega1))
                #In transition band after stopband, do rising exponential taper
                elif omegas[ii] > band[1] and omegas[ii] <= band[1] + t_width:
                    omega1 = band[1]
                    omega2 = band[1] + t_width
                    B = (atten/(-20.0))/(omega1 - omega2)
                    H_ejw[ii] = np.exp(B*(omegas[ii]-omega2))
    if plot==True:
        plt.title('Ideal filter frequency response')
        plt.scatter(omegas, H_ejw)
        plt.ylabel('Amplitude [linear]', color='g')
        plt.xlabel('Wavelength [um]')
        plt.show()
    return H_ejw
            
"""
Function to determine the coefficients of a multiband FIR filter with corresponding passbands (or stopbands) and gains using a Kaiser window

Parameters:   bands: list of 3-tuples (omega_p1, omega_p2, G) were omega_p1 <= omega <= omega_p2 is one band of the multiband filter,
                     G is the gain in the band in linear scale
              t_width: min. transition width for any band in the multiband filter
              num_coefs: Order of filter plus one(number of taps)
              beta: Kaiser parameter beta, depends on attenuation, determined using one of numpy's built-in functions
              filterType:'Pass" or 'Stop' indicating whether it is a bandpass or bandstop filter
              plot: default is True, set to false to NOT plot the frequency response
              
Return:       coefs: ndarray which holds filter coeffients, index refers to power of z^-1
              order: order of filter 
"""
def designFIRFilterGKaiser(bands, t_width, num_coefs, beta, filterType, plot=True):
    PI = np.pi
    order = num_coefs - 1
    #Kaiser window function designs Type I or II filters by default, need to check that necessary conditions are met
    if order % 2 == 1:#A Type II FIR linear phase system must have a zero at omega = PI
        if filterType == 'Pass' and bands[len(bands)-1][1] == PI:#If gain of filter is not supposed to be 0 at omega = PI
            num_coefs = num_coefs + 1#Increase order by 1 to make it a Type I
            order = order + 1
        elif filterType == 'Stop' and bands[len(bands)-1][1] != PI:
            num_coefs = num_coefs + 1#Increase order by 1 to make it a Type I
            order = order + 1
    H_ejw = generateH_ejw(bands, t_width, 2048, filterType)
    h_d = np.fft.irfft(H_ejw)#Desired filter response, in time domain
    #Make sure order is even(we need to shift by order/2)
    order = num_coefs - 1
    if order % 2 != 0:
        order = order + 1
        num_coefs = num_coefs + 1
    #Build coef array
    alpha = order/2
    h_d = np.roll(h_d, int(alpha))#Need to shift by order/2 so Kaiser window captures most of the energy
    coefs = np.zeros(num_coefs)
    for ii in range(num_coefs):
        kaiser_window_coef = ss.i0(beta*np.sqrt(1.0 - ((ii-alpha)/alpha)**2))/ss.i0(beta)
        coefs[ii] = h_d[ii]*kaiser_window_coef
    
    if plot:
        w, h = freqz(coefs,worN=2048)
        plt.title('Kaiser Window filter frequency response')
        plt.plot(w, 20*np.log10(abs(h)), 'b')
        plt.ylabel('Amplitude [dB]', color='b')
        plt.xlabel('Frequency [rad/sample]')
        plt.show()
    
    return np.flip(coefs), order#Need to flip so indices match in numpy's poly1d class, highest power of z^-1 is in index 0 of coefficent array

"""
Function to determine the coefficients of a multiband equiripple FIR filter with corresponding passbands and gains using the Parks-McClellan algorithm,
implemented using scipy.signals's remez()

Parameters:   order: Order of filter
              bands: list of 3-tuples (omega_p1, omega_p2, G) were omega_p1 <= omega <= omega_p2 is one passband of the multiband filter, G is the passband gain in linear scale
              plot: default is True, set to false to NOT plot the frequency response
              
Return:       coefs: ndarray which holds filter coeffients, index refers to power of z^-1
              order: order of filter 
"""
def designFIRFilterPMcC(order, t_width, bands, plot=True):
    PI = np.pi
    freq = []#Frequency points
    gain = []#Gain of filter for bands in freq, size of this list should be exactly half the size of freq
    
    #Build lists freq and gain
    for ii in range(len(bands)):
        if bands[ii][0] != 0.0:
            freq.append(bands[ii][0]-t_width)
            gain.append(0.0)
        freq.append(bands[ii][0])
        gain.append(bands[ii][2])
        freq.append(bands[ii][1])
        if bands[ii][1] != PI:
            freq.append(bands[ii][1]+t_width)
    if 0.0 not in freq:
        freq.insert(0,0.0)
    if PI not in freq:
        freq.append(PI)
        gain.append(0.0)

    weight = []#Weighting function to plug into remez exchnage algorithm
    for ii in range(len(gain)):
        if gain[ii] == 0.0:
            weight.append(10.0)
        else:
            weight.append(1.0)

    #Design the filter
    coefs = remez(order+1, freq, gain, weight, fs=2.0*PI)
    if plot:
        w, h = freqz(coefs)
        plt.title('Equiripple filter frequency response')
        plt.plot(w, 20*np.log10(abs(h)), 'b')
        plt.ylabel('Amplitude [dB]', color='b')
        plt.xlabel('Frequency [rad/sample]')
        plt.show()
    order = coefs.size - 1
    return coefs, order

"""
Function to determine the coefficients of a multiband FIR filter that is optimal in the least-squares sense using
Dr. T. Nguyen's and Dr. P. Vaidyanathan's eigenfilter method

Parameters:   order: Order of filter. This must be even, so numtaps = order + 1 is odd
              bands: list of 3-tuples (omega_p1, omega_p2, G) were omega_p1 <= omega <= omega_p2 is one passband of the multiband filter, G is the passband gain in linear scale
              plot: default is True, set to false to NOT plot the frequency response
              
Return:       coefs: ndarray which holds filter coeffients, index refers to power of z^-1
              order: order of filter 
"""
def designFIRFilterEigen(M, t_width, bands, plot=True):
    PI = np.pi
    Q = np.zeros((M+1,M+1))
    omegas = np.array([bands[0][0]-t_width, bands[0][0], bands[0][1], bands[0][1]+t_width, bands[1][0]-t_width, bands[1][0], bands[1][1], bands[1][1]+t_width])
    alpha = 0.1
    beta = 0.1
    gamma_1 = 0.8/3
    gamma_2 = 0.8/3
    gamma_3 = 0.0/3
    for ii in range(M+1):
        for jj in range(M+1):
            if ii == 0 and jj == 0:
                Q[ii][jj] = (1.0/PI)*(alpha*omegas[0] + gamma_1*(omegas[4]-omegas[3]) + beta*(PI-omegas[7]))
            else:
                x = ii+jj
                y = ii-jj
                Integral_1 = (1.0/x)*np.sin(x*omegas[0]) 
                Integral_2 = (1.0/x)*(np.sin(x*omegas[4]) - np.sin(x*omegas[3])) 
                Integral_3 = (-1.0/x)*np.sin(x*omegas[7]) 
                omega_01 = (omegas[2]+omegas[1])/2.0
                omega_02 = (omegas[6]+omegas[5])/2.0
                term1 = term2 = term3 = term4 = 0.0
                if ii != 0:
                    term2 = (np.cos(jj*omega_01)/ii)*(np.sin(ii*omegas[2])-np.sin(ii*omegas[1]))
                    term4 = (np.cos(jj*omega_01)/ii)*(np.sin(ii*omegas[6])-np.sin(ii*omegas[5]))
                if jj != 0:
                    term1 = (np.cos(ii*omega_01)/jj)*(np.sin(jj*omegas[2])-np.sin(jj*omegas[1]))
                    term3 = (np.cos(ii*omega_01)/jj)*(np.sin(jj*omegas[6])-np.sin(jj*omegas[5]))               
                Integral_4 = np.cos(ii*omega_01)*np.cos(jj*omega_01)*(omegas[2]-omegas[1]) - term1 - term3 + (1.0/x)*(np.sin(x*omegas[2])-np.sin(x*omegas[1])) 
                Integral_5 = np.cos(ii*omega_02)*np.cos(jj*omega_02)*(omegas[6]-omegas[5]) - term2 - term4 + (1.0/x)*(np.sin(x*omegas[6])-np.sin(x*omegas[5])) 
                if y != 0:
                    Integral_1 = Integral_1 + (1.0/y)*np.sin(y*omegas[0])
                    Integral_2 = Integral_2 + (1.0/y)*(np.sin(x*omegas[4]) - np.sin(y*omegas[3]))
                    Integral_3 = Integral_3 - (1.0/y)*np.sin(y*omegas[7])
                    Integral_4 = Integral_4 + (1.0/y)*(np.sin(y*omegas[2])-np.sin(y*omegas[1]))
                    Integral_5 = Integral_5 + (1.0/y)*(np.sin(x*omegas[6])-np.sin(y*omegas[5]))
                Q[ii][jj] = (1.0/PI)*(alpha*Integral_1 + gamma_1*Integral_2 + beta*Integral_3 + gamma_2*Integral_4 + gamma_3*Integral_5)
    lambdas, vs = np.linalg.eigh(Q)
    coefs = vs[0]
    if plot:
        w, h = freqz(coefs)
        plt.title('Eigenfilter frequency response')
        plt.plot(w, 20*np.log10(abs(h)), 'b')
        plt.ylabel('Amplitude [dB]', color='b')
        plt.xlabel('Frequency [rad/sample]')
        plt.show()
    A_N = np.flip(coefs)                                       
    order = coefs.size - 1
    return A_N, order 
    
"""
Function to iteratively design a FIR/MA filter using the Kaiser window method
Parameters: N_max: Maximum orderder allowed for filter
     center_freqs: Center (normalized) frequencies of the pass (or stop) bands, lower frequency is in index 0
            bands: List of tuples representing passband intervals, in units of normalized frequency
       filterType: Type of filter, is either "Pass" or "Stop"
Return: A_N: Coefficent array for filter, coef. of Z^-N term is in position 0 in array
          N: Filter order
    t_width: Transition band width, in normalized frequency
"""
def designKaiserIter(N_max, bands, atten, filterType):
    PI = np.pi
    max_twidth = 0.15*PI#What max. t_width is for single-band filter
    if len(bands) == 2:
        DSP_band_gap = bands[1][0] - bands[0][1]#Define max. transition width in terms of gap between passbands (or stopbands
        max_twidth = DSP_band_gap/4.0
    t_width = max_twidth/6.0
    num_coefs, beta = kaiserord(atten, t_width)#Determine parameter beta of Kaiser window
    N = num_coefs - 1
    while N > N_max:
        if t_width < max_twidth:
            t_width = 1.1*t_width
            num_coefs, beta = kaiserord(atten, t_width)#Determine parameter beta of Kaiser window
            N = num_coefs - 1
        else:
            atten = atten - 2.5
            num_coefs, beta = kaiserord(atten, t_width)#Determine parameter beta of Kaiser window
            N = num_coefs - 1
    A_N, N = designFIRFilterGKaiser(bands, t_width, num_coefs, beta, filterType, plot=False)
    return A_N, N, t_width


    
"""
Function to obtain the FIR/MA transfer function A_N(z) of the filter
Paramters: center_wvlengths: Wavelengths, in nm, of center of the pass/stop bands as numpy nd array
                  endpoints: nd array that has the 
              band_wvlength: Width, in nm, of the pass/stop bands
                filter_type: 'Pass" or 'Stop'
                      atten: Stopband attenuation in dB
                    t_width: Transition bandwidth, in units of normalized frequency
Return: A_N: Coefficients of transfer function polynomial A_N(z) coefficient of highest power of z^-1 is in index 0 of array      
"""
def frequencySamplingWithKaiserWindow(H_ejw, atten, t_width, plot=True):
    num_coefs, beta = kaiserord(atten, t_width)
    h_d = np.fft.irfft(H_ejw)#Desired filter response, in time domain
    #Make sure order is even(we need to shift by order/2)
    order = num_coefs - 1
    if order % 2 != 0:
        order = order + 1
        num_coefs = num_coefs + 1
    #Build coef array
    alpha = order/2
    h_d = np.roll(h_d, int(alpha))#Need to shift by order/2 so Kaiser window captures most of the energy
    coefs = np.zeros(num_coefs)
    for ii in range(num_coefs):
        kaiser_window_coef = ss.i0(beta*np.sqrt(1.0 - ((ii-alpha)/alpha)**2))/ss.i0(beta)
        coefs[ii] = h_d[ii]*kaiser_window_coef
    if plot == True:
        w, h = freqz(coefs)
        plt.title('Frequency-Sampled Kaiser Filter frequency response')
        plt.plot(w, 20*np.log10(abs(h)), 'b')
        plt.ylabel('Amplitude [dB]', color='b')
        plt.xlabel('Frequency [rad/sample]')
        plt.show()
    A_N = np.flip(coefs)#Term corresponding to highest power of z^-1 occupies index 0 in array
    return A_N, order

"""        
def main():
    PI = np.pi
    A_N = frequencySamplingWithKaiserWindow(np.array([1547, 1538]), np.array([1555,1532]), 4, 'Pass', 50, 0.03*PI)
    
if __name__ == '__main__':
    main()
"""


