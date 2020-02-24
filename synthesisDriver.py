"""
Eric Gelphman
UC San Diego Department of Electrical and Computer Engineering
February 23, 2020

Driver program for lattice filter synthesis
Version 1.0.1
"""

import designFilter as dF
import latticeFilterSynthesis as lfs
import numpy as np
from scipy import integrate
from scipy.signal import freqz
import matplotlib.pyplot as plt


"""
Function to calculate the unit delay length of the filter
Parameters: lamda_start = smallest wavelength in range of interest, in nm
            lamda_end = longest wavelength in rane og interest, in nm
            n_g = group index
Return: L_U in um
"""
def calcUnitDelayLength(lamda_start, lamda_end, n_g):
    denom = 2.0*((1.0/lamda_start)-(1.0/lamda_end))*n_g
    L_U = 1.0/denom
    L_U = L_U*(1E-03)
    return L_U

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

#Function to calculate max. passband attenuation/passband insertion loss
#A_N is the transfer function of the FIR filter as a numpy poly1d object
#bands is a list of passbands
def calcInsertionLoss(A_N, bands):
    """
    Find Insertion loss in each passband by numerically integrating |A_N(e^jw)| over each passband to find average value,
    then calculate the insertion loss in that passband. Then, add to list and take max.
    """
    insertionLossEachBand = []
    N = A_N.coef.size - 1
    for band in bands:
        numPoints = 150
        freqpoints = np.linspace(band[0],band[1],num=numPoints)
        magA_z_vals = np.zeros(numPoints)
        for ii in range(freqpoints.size):
            val = np.absolute(np.polyval(A_N,np.exp(freqpoints[ii]*1j)**(-N)))
            magA_z_vals[ii] = val
        avg_val = (1.0/(band[1]-band[0]))*integrate.simps(magA_z_vals,freqpoints,even='avg')
        il = 20.0*np.log10(avg_val/band[2])
        insertionLossEachBand.append(il)
    result = np.amax(insertionLossEachBand)
    return result#Return value in dB


#Function to recieve the necessary parameters from an input file to design the filter
def receiveFilterParameters(filename):
    file1 = open(filename, 'r+')
    params = []
    text = file1.readlines()
    if len(text) < 14:
        print("Error! Not Enough Input Parameters!")
        exit()
    params.append(float(text[7]))#Group index n_g
    line8 = text[8].strip('\n').split(',')
    params.append(float(line8[0]))#l_c
    params.append(float(line8[1]))#l_end
    params.append(float(text[9]))#L_2
    line10 = text[10].strip()
    params.append(line10.strip('%\n'))#Pass or Stop, indicating filter type
    num_bands = int(text[11])#Number of bands (1 or 2 for now)
    params.append(num_bands)
    if num_bands == 1:
        params.append(float(text[12]))
    elif num_bands == 2:
        line12 = text[12].strip('\n').split(',')
        params.append(float(line12[0]))#First center wavelength
        params.append(float(line12[1]))
    params.append(float(text[13]))#Max. filter order
    return params

#Function to determine necessary values to plug into the filter design functions given the input parameters
def processFilterParameters(params):
    PI = np.pi
    params = receiveFilterParameters("filterDesignParameters.txt")
    lamda_0 = 1565#nm
    lamda_1 = 1520#nm
    print(str(lamda_0))
    print(str(lamda_1))
    L_U = calcUnitDelayLength(lamda_1,lamda_0,params[0])
    lc = params[1]
    print(str(lc))
    lend = params[2]
    print(str(lend))
    L2 = params[3]
    print(str(L2))
    filter_type = params[4]#Pass or Stop
    print(filter_type)
    bands = []#Passbands of multiband filter
    n_bands = params[5]
    center_freqs = np.zeros(1)
    max_order = 0.0
    if n_bands == 1:
        center_freqs[0] = convertToNormalizedFrequency(lamda_0, lamda_1, params[6])
        max_order = params[7]
    else:
        omega_c1 = convertToNormalizedFrequency(lamda_0, lamda_1, params[6])
        omega_c2 = convertToNormalizedFrequency(lamda_0, lamda_1, params[7])
        center_freqs = np.array([omega_c1, omega_c2])
        max_order = params[8]
    print(center_freqs)
    print(str(max_order))
    return L_U, filter_type, center_freqs             

#Function to write the layout paramters to a file
def writeLayoutParametersToFile(kappalcs, phis, L_U, L_2, filename, insertionLoss, order, method):
    file1 = open(filename, 'r+')#open file
    file1.write("Filter Design Method: " + method + " Filter Order: " + str(order)+ "\n")
    file1.write("Insertion Loss: " + str(insertionLoss) + " dB" + "\n")
    file1.write("Unit Delay Length: " + str(L_U) + " um" + "\n")
    alpha = 1E-04 # db per um
    gamma = 10**((-alpha*L_2)/20.0)
    file1.write("Coupler Length L_2: " + str(L_2) + " um" + "\n")
    file1.write("Loss Coefficient Gamma per Stage: " + str(gamma) + "\n")
    #file1.write("Overall Gamma: " + str(float(order*gamma)) + "\n")
    for ii in range(len(kappalcs)):#Write parameters to file
        if ii != 0:
            file1.write(f"%{kappalcs[ii][0]:.6e},{kappalcs[ii][1]:.6e},{phis[ii-1]:6f}\n")
        else:
            file1.write(f"%{kappalcs[ii][0]:.6e},{kappalcs[ii][1]:.6e}\n")
    file1.close()#close file

def main():
    PI = np.pi
    params = receiveFilterParameters("filterDesignParameters.txt")
    L_U, filter_type, center_freqs = processFilterParameters(params)
    """
    A_N, N = designOpticalFilter(method, bands, t_width, spec)
    if A_N[0] > 0.0:
        A_N = -1.0*A_N
    print(A_N)
    """

if __name__ == '__main__':
    main()
