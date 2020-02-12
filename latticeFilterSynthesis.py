#Eric Gelphman
#UC San Diego Department of Electrical and Computer Engineering
#February 11, 2020

#Implementation of Madsen and Zhao's Optical FIR Lattice Filter Design Algorithm
#Version 1.0.5

import designFilter as dF
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
Function to obtain the normalized angular frequency omega, 0 <= omega <= pi value from a given (continous-time) wavelength value lamda
Parameters:  lamda0 = longest wavelength in range of interest, in m
             lamda1 = shortest wavelength in range of interest, in m
             lamda = wavelength you want to find normalized frequency for, in m
Return: Normalized frequency omega, 0 <= omega <= pi
"""
def convertToNormalizedFrequency(lamda0,lamda1, lamda):
    c = 3.0E8#The speed of light
    f1 = c/lamda1
    f0 = c/lamda0
    f = c/lamda
    FSR = 2*(f1-f0)
    omega = (2.0*np.pi*f)/FSR
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


"""
Function to find the reflection coefficients gamma for each stage of the lattice filter using the Levinson algorithm,
a variation of which was used in ECE 161B to calculate the lattice filter coefficents given the FIR transfer function H(z)
Paramters: A_N: Polynomial(represented by NumPy's poly1d class) in z^-1 of degree N that is the transfer function of the
                digital filter we want to implement
Return:    array of reflection coeffients gamma. The array index is the same as the stage index               
"""
def levinsonM161B(A_N):
    size = A_N.coef.size
    N = A_N.coef.size - 1#Order of function A_N(z)
    gammas = np.zeros(A_N.coef.size)
    gammas[N] = -1.0*A_N.coef[N]#gamma_N = A_N[N] = coef. of z^-N term
    #print(A_N.coef)
    for ii in range(N,0,-1):
        A_N_1 = np.zeros(ii)
        for jj in range(ii):
            beta = 1.0/(1.0 - (np.abs(gammas[ii])**2))
            A_N_1[jj] = beta*(A_N.coef[jj]+(gammas[ii]*np.conj(A_N.coef[ii-jj])))
            #print(jj)
            #print(str(A_N.coef[jj]) + " " + str(A_N.coef[ii-jj]))
            #print("A_N_1[" + str(jj) + "] : " + str(A_N_1[jj]))
        gammas[ii-1] = -1.0*A_N_1[A_N_1.size-1]
        A_N = np.poly1d(A_N_1)
    return gammas
    
            

"""
Function to find the polynomial B_N(z) which is needed to find the 2x2 transfer function of the optical filter
Parameters: A: Polynomial(represented by NumPy's poly1d class) in z^-1 of degree N that also is part of the 2x2 transfer function
Return:     Polynomial(poly1d) in z^-1 of degree N B_N(z)
"""
def findBPoly(A):
    phase_arr = np.zeros(A.coef.size)
    phase_arr[0] = 1.0#only coef. of Z^(-N) term is nonzero for phase term
    bbr_coef = -1.0*np.polymul(A.coef,np.flip(A.coef))
    bbr = np.poly1d(np.polyadd(bbr_coef,phase_arr))#Polynomial B_N(z)B_NR(z)
    roots = bbr.roots#Find roots of B_N(z)B_NR(z)
    b_roots = roots[0:roots.size-1:2]#B(z)BR(z) has n roots, n is event, B(z) has n/2 roots
    B_tild = np.poly1d(b_roots, True)#Construct polynomial from its roots
    alpha = np.sqrt((-A.coef[0]*A.coef[A.coef.size - 1])/(B_tild.coef[0]*B_tild.coef[B_tild.coef.size - 1]))#Scale factor
    B = alpha*B_tild#Build B_N(z) by scaling B_tild(z) by alpha
    """
    print("B: ")
    print(np.poly1d(B))
    print()
    """
    return np.poly1d(B)

"""
Function to compute the power coupling ratio (kappa_N)^2 of stage N of the optical filter
Parameters: A_N: Polynomial(represented by NumPy's poly1d class) in z^-1 of degree N that is part of the 2x2 transfer function
            B_N: Polynomial(poly1d) in z^-1 of degree N that is part of the 2x2 transfer function
Return:     kappa_N_sq: Coupling coefficient of stage N of the filter
"""
def calcPowerCouplingRatio(A_N, B_N):
    arr = B_N.coef
    beta = np.absolute(B_N.coef[B_N.coef.size-1]/A_N.coef[A_N.coef.size-1])**2
    kappa = beta/(1.0 + beta)
    return kappa

"""
Function to compute the cross-over length and other lengths needed for layout
Parameters: kappa: power coupling coefficient(Prof. Mookherjea's kappa^2, Madsden and Zhao's eta)
            lamda_0: wavelength, in um, of longest wavelength in filter spectrum
            lamda_1: wavelength, in um, of shortest wavelength in filter spectrum
"""
def calcLengths(kappa, lamda_0, lamda_1):
    L_c = 15.85 #Length in um, from graph on slide 9
    psy = (2.0/np.pi)*np.arcsin(np.sqrt(kappa))
    lc_lend = psy*L_c
    return L_c, lc_lend
    

"""
Function to synthesize an FIR optical lattice filter using the algorithm outlined in Section 4.5 of Madsden and Zhao
Paramters: A_N: Polynomial(represented by NumPy's poly1d class) in z^-1 of degree N that is part of the 2x2 transfer function
           N:   Filter order
           lamda_0: wavelength, in um, of longest wavelength in filter spectrum
           lamda_1: wavelength, in um, of shortest wavelength in filter spectrum
Return: kappalcs: List of power coupling coefficients kappa_n's, Lc's and lc + lend for each stage, as well as c_n and s_n list index is same as n
                  Format is: kappalcs[n] = (kappa_n, L_c_n, lc + lend of stage n, c_n, s_n)
        phi_l:   List of phase terms phi_n list index is n-1
"""
def synthesizeFIRLattice(A_N, N, lamda_0, lamda_1):
    gamma = 1.0
    phi_total = 0.0
    B_N = findBPoly(A_N)
    phi_l = []#List of phi_n's
    kappalcs = []#List of kappas, Lc's, s_n's, c_n's. This is what we want to return
    n = N
    while n >= 0:
        #print(A_N)
        #print(B_N)
        kappa = calcPowerCouplingRatio(A_N,B_N)#Find kappa
        L_c, lc_lend = calcLengths(kappa, lamda_0*1000, lamda_1*1000)#Find lengths of structures we need for layout, convert wavelengths to micrometers
        c_n = np.sqrt(1.0-kappa)
        s_n = np.sqrt(kappa)
        kappalcs.insert(0,(kappa, L_c, lc_lend, c_n, s_n))
        if n > 0:
            B_N1_arr = np.polyadd(-s_n*A_N,c_n*B_N)#Step-down recursion relation for B polynomial of stage N-1, this is an ndArray
            B_N1 = np.poly1d(B_N1_arr[0:B_N1_arr.size-1])#Reduce order by 1          
            A_N1_tild = np.polyadd(c_n*A_N,s_n*B_N)
            phi_n = -(np.angle(A_N1_tild[A_N1_tild.size-1])+np.angle(B_N1.coef[B_N1.coef.size-1]))
            phi_l.insert(0,phi_n)
            A_N1_tild = gamma*np.exp(1j*phi_n)*A_N1_tild
            A_N1 = np.poly1d(A_N1_tild[1:A_N1_tild.size])#Build polynomial A_N1(z), and reduce order by 1
        n = n - 1
        A_N = A_N1
        B_N = B_N1
    return kappalcs, phi_l

"""
Function to get the 2x2 transfer function of the filter given the power coupling ratio kappa for each stage
Parameters: kappas: array of kappa_n's 0 <= n <= N    N = filter order
            phis:   array of phi_n's 1<= n <= N
Return: Polynomials A_N(z), B_N(z), A_N_R(z), B_N_R(z) that form 2x2 transfer function of filter
"""
def fromKappasGetTransferFunction(kappas, phis):
    A_N_1 = np.poly1d([np.sqrt(1.0-kappas[0])])#A_0(z) = c_0
    B_N_1 = np.poly1d([np.sqrt(kappas[0])])#B_0(z) = s_0
    for ii in range(1,kappas.size):
        c_n = np.sqrt(1.0-kappas[ii])
        s_n = np.sqrt(kappas[ii])
        poly1arr = np.zeros(2,dtype=complex)
        num1 = c_n*np.exp(-1j*phis[ii-1])
        poly1arr[0] = num1
        poly2arr = -s_n*B_N_1
        A_N = np.polyadd(np.polymul(np.poly1d(poly1arr),A_N_1),np.poly1d(poly2arr))
        
        poly3arr = np.zeros(2,dtype=complex)
        num2 = s_n*np.exp(-1j*phis[ii-1])
        poly3arr[0] = num2
        poly4arr = c_n*B_N_1
        B_N = np.polyadd(np.polymul(np.poly1d(poly3arr),A_N_1),np.poly1d(poly4arr))

        A_N_1 = A_N
        B_N_1 = B_N

    for ii in range(A_N_1.coef.size):
        if np.imag(A_N_1.coef[ii]) < 1.0E-16:
            A_N_1.coef[ii] = np.real(A_N_1.coef[ii])
        if np.imag(B_N_1.coef[ii]) < 1.0E-16:
            B_N_1.coef[ii] = np.real(B_N_1.coef[ii])
    A_N_R = np.poly1d(np.conj(np.flip(A_N_1.coef)))
    B_N_R = np.poly1d(np.conj(np.flip(B_N_1.coef)))
    return A_N_1, B_N_1, A_N_R, B_N_R
    
        

#Function to write the layout paramters to a file
def writeLayoutParametersToFile(kappalcs, L_U, filename, insertionLoss, order, method):
    file1 = open(filename, 'r+')#open file
    file1.write("Filter Design Method: " + method + " Filter Order: " + str(order)+ "\n")
    file1.write("Insertion Loss: " + str(insertionLoss) + " dB" + "\n")
    file1.write("Unit Delay Length: " + str(L_U) + " um" + "\n")
    L_c = kappalcs[0][1]
    L_2 = 5.0*L_c
    alpha = 1E-04 # db per um
    gamma = 10**((-alpha*L_2)/20.0)
    file1.write("Cross-Over Length L_c: " + str(L_c) + " um" + "\n")
    file1.write("Coupler Length L_2: " + str(L_2) + " um" + "\n")
    file1.write("Loss Coefficient Gamma: " + str(gamma) + "\n")
    for ii in range(len(kappalcs)):#Write paramters to file
        file1.write("kappa: " + str(kappalcs[ii][0]) + " lc + lend: " + str(kappalcs[ii][2]) + " c_" +str(ii) + ": "
                    + str(kappalcs[ii][3]) + " s_" +str(ii) + ": " + str(kappalcs[ii][4]) + "\n")
    file1.close()#close file
    
    
def main():
    PI = np.pi
    lamda_0 = 1565#nanometers
    lamda_1 = 1520#nanometers
    lamda_0 = lamda_0*(1E-03)#microns
    lamda_1 = lamda_1*(1E-03)#microns
    #bands = [(0.3*PI,0.45*PI,1.0)]
    bands = [(0.3*PI, 0.4*PI, 1.0), (0.65*PI, 0.75*PI,0.75)]
    A_N, N = dF.designFIRFilterPMcC(29, 0.05*PI, bands, plot=False)
    A_N = np.flip(A_N)#So indices match in numpy's poly1d  class
    A_z = np.poly1d(A_N)
    print(A_z)
    """
    N = 2
    A_N = np.poly1d([-0.25, 0.5*np.cos(PI/6.0), -0.25])
    A_z = np.poly1d(A_N)
    print(A_z)
    #gammas = levinsonM161B(A_z)
    #print(gammas)
    A_z = np.poly1d(A_N)
    """
    kappalcs, phis = synthesizeFIRLattice(A_z, N, lamda_0, lamda_1)
    #L_U = calcUnitDelayLength(lamda_1, lamda_0, 5.5772)
    #insertionLoss = calcInsertionLoss(A_z, bands)
    #writeLayoutParametersToFile(kappalcs, L_U, "layoutParameters.txt", insertionLoss, N, "Parks-McClellan")
    kappas = np.zeros(len(kappalcs))
    for ii in range(len(kappalcs)):
        kappas[ii] = kappalcs[ii][0]
    #kappas = np.array([0.1464, 0.5, 0.8536])
    #phis = np.array([np.pi, 0.0])
    A_N, B_N, A_N_R, B_N_R = fromKappasGetTransferFunction(kappas, phis)
    print(np.poly1d(A_N))
    """
    w, h = freqz(np.flip(B_N.coef))
    plt.title('Equiripple filter frequency response')
    plt.plot(w, 20*np.log10(abs(h)), 'b')
    plt.ylabel('Amplitude [dB]', color='b')
    plt.xlabel('Frequency [rad/sample]')
    plt.show()
    """
                 
if __name__ == '__main__':
    main()
    
        
    
    
    
    
