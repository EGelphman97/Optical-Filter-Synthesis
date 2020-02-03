#Eric Gelphman
#UC San Diego Department of Electrical and Computer Engineering
#February 2, 2020

#Implementation of Madsen and Zhao's Optical FIR Lattice Filter Design Algorithm
#Version 1.0.3

import designFilter as dF
import numpy as np
from scipy import integrate

#Function to calculate max. passband attenuation/passband insertion loss
#A_N is the transfer function of the FIR filter as a numpy poly1d object
#bands is a list of passbands
def calcInsertionLoss(A_N, bands):
    overall_max = []
    for band in bands:
        #Find maximum deviation from ideal value in passband by numerically integrating |A_N(z)| over the passband
        freqpoints = np.linspace(band[0],band[1],num=100)
        magA_z_vals = np.zeros(100)
        for ii in freqpoints:
            N = A_N.coef.size
            val = np.absolute(np.polyval(A_N,np.exp(ii*1j)**(-N)))
            magA_z_vals[int(ii)] = val
        avg_val = (1.0/(band[1]-band[0]))*integrate.simps(magA_z_vals,freqpoints, even='avg')
        overall_max.append(np.absolute(band[2]-avg_val))
    result = np.amin(overall_max)
    return 20.0*np.log10(result)#Return value in dB

#Function to determine if a frequency is in the stopband
def inStopBand(bands, t_width, freq):
    result = True
    for band in bands:
        if band[0]-t_width <= freq and band[1]+t_width >= freq:
            result = False
    return result

#Function to calculate stopband extinction
def calcStopbandExtinction(A_N, bands, t_width):
    #Find maximum magnitude of transfer function in stopband by numerically integrating |A_N(z)| over the stopband
    freqpoints = np.linspace(0.0,np.pi,num=1200)
    magA_z_vals = []#Values of A_N(z) actually in stopband
    #stopbandPoints = []#Frequency points actually in stopband
    for ii in freqpoints:
        #If point is actually in the stopband
        if inStopBand(bands, t_width, ii):
            N = A_N.coef.size
            val = np.absolute(np.polyval(A_N,np.exp(ii*1j)**(-N)))**2
            magA_z_vals.append(val)
            #stopbandPoints.append(ii)
    print(magA_z_vals)
    avg_val = np.average(np.array(magA_z_vals)) 
    return -20.0*np.log10(avg_val)#Return value in dB
     
    
    

"""
Function to find the polynomial B_N(z) which is needed to find the 2x2 transfer function of the optical filter
Parameters: A: Polynomial(represented by NumPy's poly1d class) in z^-1 of degree N that also is part of the 2x2 transfer function
            phi_total: Total phase shift of filter
Return:     Polynomial(poly1d) in z^-1 of degree N B_N(z)
"""
def findBPoly(A):
    phase_arr = np.zeros(A.coef.size)
    phase_arr[0] = 1.0#only coef. of Z^(-N) term is nonzero for phase term
    #print(np.poly1d(phase_arr))
    A_R = np.poly1d(np.flip(A.coef))
    bbr_coef = -1.0*np.polymul(A.coef,np.flip(A.coef))
    bbr = np.poly1d(np.polyadd(bbr_coef,phase_arr))#Polynomial B_N(z)B_NR(z)
    roots = bbr.roots#Find roots of B_N(z)B_NR(z)
    b_roots = roots[0:roots.size-1:2]#B(z)BR(z) has n roots, n is event, B(z) has n/2 roots
    B_tild = np.poly1d(b_roots, True)#Construct polynomial from its roots
    #print(B_tild)
    alpha = np.sqrt((-A.coef[0]*A.coef[A.coef.size - 1])/(B_tild.coef[0]*B_tild.coef[B_tild.coef.size - 1]))#Scale factor
    
    #print(B_tild.coef[0]*B_tild.coef[B_tild.coef.size - 1])
    B = alpha*B_tild#Build B_N(z) by scaling B_tild(z) by alpha
    #print("B: ")
    #print(np.poly1d(B))
    #print()
    return np.poly1d(B)

"""
Function to compute the coupling coefficient (kappa_N)^2 of stage N of the optical filter
Parameters: A_N: Polynomial(represented by NumPy's poly1d class) in z^-1 of degree N that is part of the 2x2 transfer function
            B_N: Polynomial(poly1d) in z^-1 of degree N that is part of the 2x2 transfer function
Return:     kappa_N_sq: Coupling coefficient of stage N of the filter
"""
def calcCouplingCoef(A_N, B_N):
    arr = B_N.coef
    beta = np.absolute(B_N.coef[B_N.coef.size-1]/A_N.coef[A_N.coef.size-1])**2
    kappa = beta/(1.0 + beta)
    return kappa

"""
Function to synthesize an FIR optical lattice filter using the algorithm outlined in Section 4.5 of Madsden and Zhao
Paramters: A_N: Polynomial(represented by NumPy's poly1d class) in z^-1 of degree N that is part of the 2x2 transfer function
           N:   Filter order
Return: kappa_l: List of coupling coefficients kappa_n_sq list index is same as n
        phi_l:   List of phase terms phi_n list index is n-1
"""
def synthesizeFIRLattice(A_N, N):
    gamma = 1.0
    phi_total = 0.0
    B_N = findBPoly(A_N)
    kappa_l = []#List of kappa_n's
    phi_l = []#List of phi_n's
    lcsgamma = []#List of Lc's, s_n's, c_n's, and gammas
    n = N
    while n >= 0:
        #print(A_N)
        #print(B_N)
        kappa = calcCouplingCoef(A_N,B_N)
        L_c = (2.0*np.pi)/kappa
        kappa_l.insert(0,kappa)
        c_n = np.sqrt(1.0-kappa)
        s_n = np.sqrt(kappa)
        gamma = 10**(-L_c*(10**(-4))/20.0)
        lcsgamma.insert(0,(L_c,c_n,s_n,gamma))
        if n > 0:
            B_N1_arr = np.polyadd(-s_n*A_N,c_n*B_N)#Step-down recursion relation for B polynomial of stage N-1, this is an ndArray
            B_N1 = np.poly1d(B_N1_arr[0:B_N1_arr.size-1])#Reduce order by 1
            A_N1_tild = np.polyadd(c_n*A_N,s_n*B_N)
            phi_n = -(np.angle(A_N1_tild[A_N1_tild.size-1])+np.angle(B_N1.coef[B_N1.coef.size-1]))
            phi_l.insert(0,phi_n)
            A_N1_tild = np.exp(1j*phi_n)*A_N1_tild
            A_N1 = np.poly1d(A_N1_tild[1:A_N1_tild.size])#Build polynomial A_N1(z), and reduce order by 1
        n = n - 1
        A_N = A_N1
        B_N = B_N1
    return kappa_l, phi_l, lcsgamma

#Function to write the layout paramters to a file
def writeLayoutParametersToFile(kappas, lcsg, filename, insertionLoss, extinction, order, method):
    if len(kappas) != len(lcsg):
        print("Error! Array Dimension of Layout Paramters Mismatch!")
        print(len(kappas))
        print(len(lcsg))
        exit()
    file1 = open(filename, 'r+')#open file
    file1.write("Filter Design Method: " + method + " Filter Order: " + str(order)+ "\n")
    file1.write("Insertion Loss: " + str(insertionLoss) + " dB" + "\n")
    file1.write("Extinction: " + str(extinction) + " dB" + "\n")
    for ii in range(len(kappas)):#Write paramters to file
        file1.write("kappa: " + str(kappas[ii]) + " L_c: " + str(lcsg[ii][0]) + " c_" +str(ii) + ": " + str(lcsg[ii][1]) +
                    " s_" +str(ii) + ": " + str(lcsg[ii][2]) + " gamma_" +str(ii) + ": " + str(lcsg[ii][3]) +"\n")
    file1.close()#close file
    
    
def main():
    PI = np.pi
    #bands = [(0.3*PI,0.45*PI,1.0)]
    bands = [(0.3*PI, 0.4*PI, 1.0), (0.65*PI, 0.75*PI,0.75)]
    A_N, N = dF.designFIRFilterPMcC(25, 0.03*PI, bands, plot=True)
    print("Order: " + str(N))
    #N = 2
    #A_N = np.poly1d([-0.25,0.25*2.0*np.cos(np.pi/6),-0.25])
    A_z = np.poly1d(A_N)
    kap_l, ph_l, lcsg = synthesizeFIRLattice(A_z, N)
    insertionLoss = calcInsertionLoss(A_z, bands)
    extinction = calcStopbandExtinction(A_z, bands, 0.03*PI)
    writeLayoutParametersToFile(kap_l, lcsg, "layoutParameters.txt", insertionLoss, extinction, N, "Parks-McClellan")
    
    #N = 2
    #A_N = np.poly1d([-0.25,0.25*2.0*np.cos(np.pi/6),-0.25])
    #kap_l, ph_l = synthesizeFIRLattice(A_N, N)
    
                 
if __name__ == '__main__':
    main()
    
        
    
    
    
    
