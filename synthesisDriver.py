"""
Eric Gelphman
UC San Diego Department of Electrical and Computer Engineering
February 18, 2020

Driver program for lattice filter synthesis
Version 1.0.0
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
    if len(text) < 11:
        print("Error! Not Enough Input Parameters!")
        exit()
    line0 = text[0].strip('\n').split(':')
    method = line0[1].strip()
    if method == "Kaiser" or method == "Parks-McClellan":
        params.append(method)#Filter design method
    else:
        print("Error! Input Kaiser or Parks-McClellan for filter design method")
    line1 = text[1].strip('\n').split(':')
    params.append(float(line1[1]))#Wavelength lamda_min, in nm
    line2 = text[2].strip('\n').split(':')
    params.append(float(line2[1]))#Wavelength lamda_max, in nm
    line3 = text[3].strip('\n').split(':')
    params.append(float(line3[1]))#Group index ng
    line4 = text[4].strip('\n').split(':')
    small_ls = line4[1].split(',')#l_c and l_end
    params.append(float(small_ls[0]))#l_c
    params.append(float(small_ls[1]))#l_end
    line5 = text[5].strip('\n').split(':')
    params.append(float(line5[1]))#L2
    line6 = text[6].strip()
    params.append(line6.strip('%\n'))#Pass or Stop, indicating filter type
    line7 = text[7].strip('\n').split(':')
    num_bands = int(line7[1])#Number of bands (1 or 2 for now)
    params.append(num_bands)
    if num_bands == 1:
        line8 = text[8].strip('\n').split(":")
        line8mod = line8[1].strip('[]').split(',')
        params.append([float(line8mod[0]),float(line8mod[1])])
    elif num_bands == 2:
        line8 = text[8].strip('\n').split(":")
        intervals = line8[1].split(';')
        interval1 = intervals[0].strip('[]').split(',')
        params.append([float(interval1[0]),float(interval1[1])])#Firsrt pass/stop band
        interval2 = intervals[1].strip('[]').split(',')
        params.append([float(interval2[0]),float(interval2[1])])
    line9 = text[9].strip('\n').split(':')
    params.append(float(line9[1]))#transition bandwidth
    line10 = text[10].strip('\n').split(':')
    if method == 'Kaiser':
        params.append(float(line10[1]))#Attenuation in dB
    else:
        params.append(int(line10[1]))#Filter order
    return params

#Function to determine necessary values to plug into the filter design functions given the input parameters
def processFilterParameters(params):
    PI = np.pi
    params = receiveFilterParameters("filterDesignParameters.txt")
    method = params[0]
    lamda_0 = params[2]#nm
    lamda_1 = params[1]#nm
    print(method)
    print(str(lamda_0))
    print(str(lamda_1))
    L_U = calcUnitDelayLength(lamda_1,lamda_0,params[3])
    lc = params[4]
    print(str(lc))
    lend = params[5]
    print(str(lend))
    L2 = params[6]
    print(str(L2))
    
    bands = []#Passbands of multiband filter
    n_bands = params[8]
    
    #Determine transition bandwidth
    if n_bands == 1:
        omega_prime = convertToNormalizedFrequency(lamda_0, lamda_1, params[9][1]-params[10])
    else:
        omega_prime = convertToNormalizedFrequency(lamda_0, lamda_1, params[9][1]-params[11])
    t_width = omega_prime - convertToNormalizedFrequency(lamda_0, lamda_1, params[9][1])#Transition bandwidth = passband_edge + t_width - passband_edge
    print(str(t_width))
    
    if params[7] == "Pass":
        lamda_p1 = params[9][0]
        lamda_p2 = params[9][1]
        omega_p1 = convertToNormalizedFrequency(lamda_0, lamda_1, lamda_p1)
        omega_p2 = convertToNormalizedFrequency(lamda_0, lamda_1, lamda_p2)
        bands.append((omega_p1,omega_p2,1.0))
        if n_bands == 2:
            lamda_p3 = params[10][0]
            lamda_p4 = params[10][1]
            omega_p3 = convertToNormalizedFrequency(lamda_0, lamda_1, lamda_p3)
            omega_p4 = convertToNormalizedFrequency(lamda_0, lamda_1, lamda_p4)
            bands.append((omega_p3,omega_p4,1.0))
    elif params[7] == "Stop":
        stopbands = []
        lamda_s1 = params[9][0]
        lamda_s2 = params[9][1]
        omega_s1 = convertToNormalizedFrequency(lamda_0, lamda_1, lamda_s1)
        omega_s2 = convertToNormalizedFrequency(lamda_0, lamda_1, lamda_s2)
        stopbands.append((omega_s1,omega_s2))
        if n_bands == 2:
            lamda_s3 = params[10][0]
            lamda_s4 = params[10][1]
            omega_s3 = convertToNormalizedFrequency(lamda_0, lamda_1, lamda_s3)
            omega_s4 = convertToNormalizedFrequency(lamda_0, lamda_1, lamda_s4)
            stopbands.append((omega_s3,omega_s4))
        bands = dF.obtainPassbandsFromStopbands(stopbands, t_width)#Need passbands to plug into filter design functions
    for ii in range(len(bands)):
        print("[" + str(bands[ii][0]) + "," + str(bands[ii][1]) + "]" + "\n")
    spec = 0.0
    if method == "Kaiser":
        if n_bands == 1:
            spec = params[11]
        else:
            spec = params[12]
    else:
        if n_bands == 1:
            spec = int(params[11])
        else:
            spec = int(params[12])
    print(str(spec))
    return method, L_U, bands, t_width, spec

#Function to choose the appropriate filter design method once the imputs have been processed
def designOpticalFilter(method, bands, t_width, spec):
    if method == "Kaiser":
        return dF.designFIRFilterKaiser(spec, bands, t_width, plot=False)
    else:
        return dF.designFIRFilterPMcC(spec, t_width, bands, plot=False)
              

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
        if ii != 0 and ii < 10:
            file1.write(f"% Stage {ii}  : kappa: {kappalcs[ii][0]:.6e} L_c: {kappalcs[ii][1]:.6e} um  phi: {phis[ii-1]:6f} \n")
        elif ii >= 10:
            file1.write(f"% Stage {ii} : kappa: {kappalcs[ii][0]:.6e} L_c: {kappalcs[ii][1]:.6e} um  phi: {phis[ii-1]:6f} \n")
        else:
            file1.write(f"% Stage {ii}: kappa: {kappalcs[ii][0]:.6e} L_c: {kappalcs[ii][1]:.6e} um \n")
    file1.close()#close file

def main():
    PI = np.pi
    params = receiveFilterParameters("filterDesignParameters.txt")
    method, L_U, bands, t_width, spec = processFilterParameters(params)
    """
    A_N, N = designOpticalFilter(method, bands, t_width, spec)
    if A_N[0] > 0.0:
        A_N = -1.0*A_N
    print(A_N)
    """

if __name__ == '__main__':
    main()
