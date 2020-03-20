"""
Eric Gelphman
UC San Diego Department of Electrical and Computer Engineering
Last Updated March 20, 2020 Version 1.1.2

Python Script that has methods to obtain the transfer function H(z)
Of digital filters as an array of coefficients. These coefs. will then be
passed to latticeFilterSynthesis.py to determine the lattice parameters

Required Imports:
-numpy
-scipy.signal: freqz, remez (only need these two functions, don't
               need whole package)
-scipy.special: io (modified Bessel function of 1st kind)
-matplotlib.pyplot
"""

import numpy as np
from scipy.signal import freqz, remez
from scipy.special import i0
import matplotlib.pyplot as plt

def convertToNormalizedFrequency(lamda0, lamda1, lamda):
    """
    Function to obtain the normalized angular frequency omega, 0 <= omega <=
    pi value from a given (continuous-time) wavelength value lamda
    
    Parameters:  lamda0 = longest wavelength in range of interest, in nm
                 lamda1 = shortest wavelength in range of interest, in nm
                 lamda = wavelength you want to find normalized frequency for, in nm
    Return: Normalized frequency omega, 0 <= omega <= pi
    """
    c = 3.0E8#The speed of light
    lamda_0 = lamda0*(1.0E-9)
    lamda_1 = lamda1*(1.0E-9)
    lamda_ = lamda*(1.0E-9)
    f1 = c/lamda_1
    f0 = c/lamda_0
    f = c/lamda_
    omega = np.pi*(f-f0)/(f1-f0)
    return omega


def obtainPassbandsFromStopbands(stopbands, t_width):
    """
    Function to perform a boolean operation on a list of stopbands to get
    corresponding list of passbands
    
    Parameters: stopbands: list of tuples of corner (normalized) frequencies
                           of stopbands
                t_width:   transition bandwidth (normalized frequency)
    Return: list of tuples of passbands with corresponding gains
    """
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

def obtainBandsFromCenterWvlengths(center_wvlengths, endpoints, band_wvlength, typef):
    """
    Function to obtain the bands(pass or stop), in units of normalized frequency, from
    the center wavelengths (in nm)
    
    Parameters: center_wvlength: ndarray of center wavelengths, longest wavelength/
                                 lowest frequency is in position 0 of array
                  band_wvlength: desired width of pass (or stop) band, in nm
                      endpoints: endpoints of wavelength of interest, longest wavelength/
                                 lowest frequency is in position 0 of array
                          typef: string that indicates type of filter, this parameter is
                                 either "Pass" or "Stop"
    Return: List of tuples representing band (normalized) frequency intervals
    """
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


def generateH_ejw(bands, t_width, N, filter_type, plot=False):
    """
    Function to generate the "ideal" Frequency response H(e^jw) of filter for use in
    Kaiser window function
    
    Parameters: 
                  bands: List of tuples (or any array-like object)
                t_width: Transition width of exponential taper
                      N: ifft length(>= 1024)
            filter_type: 'Pass' or 'Stop'
    Return: data: matrix of wavelengths and amplitude at that wavelength in dB
    """
    PI = np.pi
    atten_linear = 10**(-300/20.0)#Do exponential taper to very small value (nearly 0) -300 db is nearly 0
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
                    break
                #In transition band before passband, do rising exponential taper
                elif omegas[ii] >= band[0] - t_width and omegas[ii] < band[0]:
                    omega1 = band[0] - t_width
                    omega2 = band[0]
                    B = mag_val/(omega1 - omega2)
                    H_ejw[ii] = np.exp(B*(omegas[ii]-omega2))
                    break
                #In transition band after passband, do decaying exponential taper
                elif omegas[ii] > band[1] and omegas[ii] <= band[1] + t_width:
                    omega1 = band[1]
                    omega2 = band[1] + t_width
                    B = mag_val/(omega2 - omega1)
                    H_ejw[ii] = np.exp(B*(omegas[ii]-omega1))
                    break
    else:
        atten = 300#Attenuation in db
        for ii in range(N):
            for band in bands:
                #If in stopband, 
                if omegas[ii] >= band[0] and omegas[ii] <= band[1]:
                    H_ejw[ii] = atten_linear
                    break
                #In transition band before stopband, do decaying exponential taper
                elif omegas[ii] >= band[0] - t_width and omegas[ii] < band[0]:
                    omega1 = band[0] - t_width
                    omega2 = band[0] 
                    B = (atten/(-20.0))/(omega2 - omega1)
                    H_ejw[ii] = np.exp(B*(omegas[ii]-omega1))
                    break
                #In transition band after stopband, do rising exponential taper
                elif omegas[ii] > band[1] and omegas[ii] <= band[1] + t_width:
                    omega1 = band[1]
                    omega2 = band[1] + t_width
                    B = (atten/(-20.0))/(omega1 - omega2)
                    H_ejw[ii] = np.exp(B*(omegas[ii]-omega2))
                    break
    if plot==True:
        plt.title('Ideal filter frequency response')
        plt.scatter(omegas, H_ejw)
        plt.ylabel('Amplitude [linear]', color='g')
        plt.xlabel('Wavelength [um]')
        plt.show()
    return H_ejw
            

def designFIRFilterGKaiser(bands, t_width, num_coefs, beta, filterType, table=None, plot=True):
    """
    Function to determine the coefficients of a multiband FIR filter with corresponding passbands
    (or stopbands) and gains using a Kaiser window

    Parameters:   bands: list of 3-tuples (omega_p1, omega_p2) where omega_p1 <= omega <= omega_p2
                         is one band of the multiband filter
                t_width: min. transition width for any band in the multiband filter
              num_coefs: Order of filter plus one(number of taps) The filter order needs to be even, so this should be odd
                         This is because we want the filter to be a Type I linear phase system, which has no restriction on the
                         location of its zeros 
                   beta: Kaiser parameter beta, depends on attenuation, determined using one of numpy's
                         built-in functions
             filterType: 'Pass" or 'Stop' indicating whether it is a bandpass or bandstop filter
                  table: If input file is in table format. Default is none
                   plot: default is True, set to false to NOT plot the frequency response
              
    Return:       coefs: ndarray which holds filter coeffients, index refers to power of z^-1
    """
    PI = np.pi
    order = num_coefs - 1
    H_ejw = np.zeros(1)
    if table is not None:
        H_ejw = np.array(table[:,1])
    else:
        H_ejw = generateH_ejw(bands, t_width, 4096, filterType)
    h_d = np.fft.irfft(H_ejw)#Desired filter response, in time domain
    #Build coef array
    alpha = order/2
    h_d = np.roll(h_d, int(alpha))#Need to shift by order/2 so Kaiser window captures most of the energy
    coefs = np.zeros(num_coefs)
    for ii in range(num_coefs):
        kaiser_window_coef = i0(beta*np.sqrt(1.0 - ((ii-alpha)/alpha)**2))/i0(beta)
        coefs[ii] = h_d[ii]*kaiser_window_coef
    
    if plot:
        w, h = freqz(coefs,worN=4096)
        plt.title('Kaiser Window filter frequency response')
        plt.plot(w, 20*np.log10(abs(h)), 'b')
        plt.ylabel('Amplitude [dB]', color='b')
        plt.xlabel('Frequency [rad/sample]')
        plt.show()
    
    return np.flip(coefs)#Need to flip so indices match in numpy's poly1d class, highest power of z^-1 is in index 0 of coefficent array


def designFIRFilterParksMcClellan(order, t_width, bands, filterType, plot=True):
    """
    Function to determine the coefficients of a multiband equiripple FIR filter with corresponding
    passbands and gains using the Parks-McClellan algorithm, implemented using scipy.signals's remez()

    Parameters:   order: Order of filter, must be even for Type I linear phase systems with no restriction on
                         location of zeros
                  bands: list of 3-tuples (omega_p1, omega_p2) were omega_p1 <= omega <= omega_p2 is
                         one passband of the multiband filter
             filterType: 'Pass" or 'Stop", indicating filter type
                   plot: default is True, set to false to NOT plot the frequency response
              
        Return:   coefs: ndarray which holds filter coeffients, index refers to power of z^-1
                  order: order of filter 
    """
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
        if filterType == 'Pass':
            gain.append(0.0)
        else:
            gain.append(1.0)

    weight = []#Weighting function to plug into remez exchnage algorithm
    for ii in range(len(gain)):
        if gain[ii] == 0.0:
            weight.append(10.0)
        else:
            weight.append(1.0)

    #Design the filter
    coefs = remez(order+1, freq, gain, weight, fs=2.0*PI)
    print(coefs)
    if plot:
        w, h = freqz(coefs)
        plt.title('Equiripple filter frequency response')
        plt.plot(w, 20*np.log10(abs(h)), 'b')
        plt.ylabel('Amplitude [dB]', color='b')
        plt.xlabel('Frequency [rad/sample]')
        plt.show()
    order = coefs.size - 1
    return np.flip(coefs)


def designFIRFilter(N, bands, atten, filterType, method, table=None):
    """
    Function to iteratively design a FIR/MA filter using the Kaiser window method
    Parameters: N: Filter order, this needs to be even
         center_freqs: Center (normalized) frequencies of the pass (or stop) bands, lower frequency is
                       in index 0
                bands: List of tuples representing passband intervals, in units of normalized frequency
           filterType: Type of filter, is either "Pass" or "Stop"
               method: Method used to design filter - is either "Kaiser" or "ParksMcClellan"
                table: Table of frequency response values if input file is in table format
    Return: A_N: Coefficent array for filter, coef. of Z^-N term is in position 0 in array
              N: Filter order
        t_width: Transition band width, in normalized frequency
    """
    PI = np.pi
    A_N = np.zeros(1)
    t_width = 0.0
    if method == 'Kaiser':
        if atten < 21.0:
            beta = 0.0
        elif atten >= 21.0 and atten <= 50.0:
            beta = 0.5842*(atten - 21.0)**(0.4) + 0.07886*(atten - 21.0)
        else:
            beta = 0.1102*(atten - 8.7)
        t_width = (atten - 8.0)/(2.285*N)
        A_N = designFIRFilterGKaiser(bands, t_width, N + 1, beta, filterType, table=None, plot=False)
    else:
        del_s = 10**(-atten/20.0)
        del_p = 10**(1-(atten/20.0))
        t_width = (1.0/(14.6*N))*((2.0*np.pi)*(-20.0*np.log10(np.sqrt(del_s*del_p))-13.0))
        A_N = designFIRFilterParksMcClellan(N, t_width, bands, filterType)
    return A_N, t_width

"""    
def main():
    PI = np.pi
    M = 40
    t_width = 0.06*PI
    center_wvlengths = np.array([1540])
    bands = obtainBandsFromCenterWvlengths(center_wvlengths, np.array([1560, 1520]), 8, 'Pass')
    A_N, N = designFIRFilterEigen(M, t_width, bands, plot=True)
    #A_N = (np.array([1547, 1538]), np.array([1555,1532]), 4, 'Pass', 50, 0.03*PI)
    
if __name__ == '__main__':
    main()
"""



