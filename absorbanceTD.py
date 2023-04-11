#!/usr/bin/env python3
# -*- coding: utf_8 -*-

"""
Python library for evaluation of principal components of unpolarized absorbance measurements
using both procedures of Sambridge et al. (2008) and Jackson et al. (2018)

M. Sambridge and A. Jackson, 2018.
"""

from __future__ import print_function
from __future__ import unicode_literals
from pyevtk.hl import gridToVTK

import time
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

#CSV likelihood
# df = pd.read_csv('likelihood.csv')
# df.to_csv('likelihood.csv')

# Absorbance library routines

def pdf(Q, a1, a2, a3):
    """Probability density function, PDF, eqns (32)-(33) of Jackson et al. (2018)"""

    if not a1 >= a2 > a3:
        raise ValueError('Function pdf: Input error in arguments. It should be a1 >= a2 > a3')
    if Q <= a3:
        return 0.0
    if Q >= a1:
        return 0.0
    if Q > a2:
        return 1.0/np.pi/np.sqrt((a1-a2)*(Q-a3))*float(mp.ellipk((a2-a3)*(a1-Q)/(a1-a2)/(Q-a3)))
    return 1.0/np.pi/np.sqrt((a2-a3)*(a1-Q))*float(mp.ellipk((a1-a2)*(Q-a3)/(a2-a3)/(a1-Q)))

def G(c, b, p):
    """G function, eqns (28)-(29) of Jackson et al. (2018)"""

    return 2*(c-b)*np.sqrt(1.0/(p+1.0)/(b+c))*float(mp.ellippi(2.0*b/(b+c), 2.0*(b+c*p)/(b+c)/(p+1.0)))

def cdf(Q, a1, a2, a3):
    """Cummulative density function, CDF, eqns (28)-(30) of Jackson et al. (2018)"""

    if not a1 >= a2 >= a3:
        raise ValueError('Function cdf: Input error in arguments. It should be a1>=a2>=a3')
    if Q <= a3:
        return 0.0
    if Q >= a1:
        return 1.0
    if Q == a2:
        return 1.0/np.pi*2.0*np.arctan((a2-a3)/(a1-a2))
    if Q > a2:
        return 1.0/np.pi/np.sqrt(2*a1-a2-a3)*G(2*Q-a2-a3, a3-a2, (a2-a3)/(2*a1-a2-a3))
    return 1.0 - 1.0/np.pi/np.sqrt(a1+a2-2*a3)*G(a1+a2-2*Q, a2-a1, (a1-a2)/(a1+a2-2*a3))

def pdfsmooth(Q, a1, a2, a3, delta):
    """evaluate smoothed PDF for a single datum and multiple absorbance values"""

    return (cdf(Q+delta, a1, a2, a3)-cdf(Q-delta, a1, a2, a3))/(2.0*delta)

def opt_func(x, data, delta):
    """evaluate -ve log(likelihood)"""
    if (x[0] - x[1] <= x[2]) & (x[2] <= x[1]) & (x[1] <= x[0]):
        out = np.zeros(len(data))
        for i, v in enumerate(data):
            out[i] = pdfsmooth(v, x[0], x[1], x[2], delta)
            if out[i] == 0.0:
                #print (' i =', i)
                return np.finfo(np.float32).max # return barrier function : 0 when BL wanted!!!!
        return -np.sum(np.log(out)) # pylint: disable=invalid-unary-operand-type
    #  " bug workaround
    #print (' point infeasible')
    return np.finfo(np.float32).max # return barrier function

def checkPDF(x, data, delta):
    """check if PDF is zero because of data distribution"""

    if (x[0] - x[1] <= x[2]) & (x[2] <= x[1]) & (x[1] <= x[0]):
        for v in data:
            if pdfsmooth(v, x[0], x[1], x[2], delta) == 0.0:
                return True # return barrier function
        return False
    return True

def abs_ParamSearchSize(x, y, z):
    """Calculate number of feasible combinations of principal absorbances in grid"""

    xv, yv, zv = np.meshgrid(x, y, z)
    bl = (xv - yv <= zv) & (zv <= yv) & (yv <= xv) # locations of grid points that satisfy constraints
    return len(xv[bl])

def abs_ParamSearch(x, y, z, data, delta):
    """perform grid search to maximize log(likelihood)"""

    xv, yv, zv = np.meshgrid(x, y, z)
    bl = (xv - yv <= zv) & (zv <= yv) & (yv <= xv) #& (xv <= yv + zv) # locations of grid points that satisfy constraints !! ZV NOT Z??!!
    #a1b, a2b, a3b, best = xv[bl][0], yv[bl][0], zv[bl][0], np.sum(np.log(pdfsmooth(data, xv[bl][0], yv[bl][0], zv[bl][0], delta)))
    gridToVTK("bl", x, y, z, cellData = {'bl':1*bl})
    a1b, a2b, a3b, best = 0.0, 0.0, 0., -np.finfo(np.float32).max
    loglikegrid = np.zeros((len(x), len(y), len(z)))
    for idx,i in enumerate(x):
        for idy,j in enumerate(y):
            for idz,k in enumerate(z):
                p = [i,j,k]
                loglikegrid[idx][idy][idz] = -opt_func(p, data, delta)
    #print('le max est: ',np.max(loglikegrid))
    for v in zip(xv[bl], yv[bl], zv[bl]):           # v = coordinates of possible data
        loglike = -opt_func(v, data, delta)
        #print(loglike)
        if loglike > best:
            best, a1b, a2b, a3b = loglike, v[0], v[1], v[2]
            print('loglike', loglike, 'a1', v[0], 'a2', v[1], 'a3', v[2])
    return best, a2b+a3b-a1b, a1b+a3b-a2b, a1b+a2b-a3b, a1b+a2b+a3b, a1b, a2b, a3b, loglikegrid

def gridsearchRect(data, delta, n):
    """Perform grid search for principal absorbances over a 3-D rectangle"""

    # set up Cartesian search grid for parameters (a1, a2, a3)
    a2_start = 0.9*np.min(data)  # set limits of a2
    a2_stop = 1.1*np.max(data)   # set limits of a2
    a3_start = 0.0               # set lower limit of a3 using eqn. (41) Jackson et al. (2018)
    a3_stop = a2_stop            # set upper limit of a3 using eqn. (40) Jackson et al. (2018)
    a1_start = a2_start          # set lower limit of a1 using eqn. (41) Jackson et al. (2018)
    a1_stop = 2.0*a2_stop        # set upper limit of a1 using eqn. (40) Jackson et al. (2018)
    x = np.linspace(a1_start, a1_stop, n[0]) # a1 discretized values
    y = np.linspace(a2_start, a2_stop, n[1]) # a2 discretized values
    z = np.linspace(a3_start, a3_stop, n[2]) # a3 discretized values
    BC = [[a1_start,a1_stop],[a2_start,a2_stop],[a3_start,a3_stop]]
    np.savetxt("BC.txt", BC)

    print('\nPerforming grid search on grid', n[0], 'x', n[1], 'x', n[2],
    '\nDiscretization intervals delta a1 =', (a1_stop-a1_start)/n[0],
    'delta a2 =', (a2_stop-a2_start)/n[1],
    'delta a3 =', (a3_stop-a3_start)/n[2],
    '\nwith', abs_ParamSearchSize(x, y, z), 'feasible points...\n')
    start_time = time.time()
    sol = abs_ParamSearch(x, y, z, data, delta)
    print("--- {:g} seconds compute time ---".format(time.time() - start_time))
    s = sol[8]
    gridToVTK("likelihood", x, y, z, cellData = {'likelihood':s})

    #print(s)
    #np.savetxt("Likelihood.csv",s,fmt='%.4e')#, delimiter=",")
    # with open('Likelihood.csv','wb') as f:
    # for a in s:
	# np.savetxt(f, a)#, fmt='%10d')
	# f.write('\n')
    #export 3Darray to CSV:
    s_reshaped = s.reshape(s.shape[0],-1)
    print('shape is ',s.shape)
    np.savetxt("Likelihood.txt", s_reshaped)

    return sol

def gridsearch(data, delta, n):
    """Perform grid search for principal absorbances within model feasible region"""

    a1b, a2b, a3b, best = 0.0, 0.0, 0., -np.finfo(np.float32).max

    y = np.linspace(0.9*np.min(data), 1.1*np.max(data), n[1]+1) # a2 discretized values
    print('\nPerforming grid search on grid', n[0], 'x', n[1], 'x', n[2],
          '\nDiscretization of a2 axis: a2 start', 0.9*np.min(data),
          ' a2_stop', 1.1*np.max(data),
          ' delta a2', y[1]-y[0])
    start_time = time.time()
    loglikegrid = np.zeros(np.array(n)+1)
    # for yv in y:
    #     z = np.linspace(0.0, yv, n[2]+1)
    #     for zv in z:
    #         x = np.linspace(yv, yv+zv, n[0]+1)
    #         for xv in x:
    for iy, yv in enumerate(y):
        z = np.linspace(0.0, yv, n[2]+1)
        for iz, zv in enumerate(z):
            x = np.linspace(0, yv+zv, n[0]+1) #0 instead of yv??????????????????????????????
            for ix, xv in enumerate(x):
    # for yv in y:
    #     z = np.linspace(0.0, yv, n[2]+1)
    #     for zv in z:
    #         x = np.linspace(yv, yv+zv, n[0]+1)
    #         for xv in x:
                loglike = -opt_func([xv, yv, zv], data, delta)
                loglikegrid[ix][iy][iz] = loglike
                #loglikegrid.append(loglike)
                #print(i, j, k, loglike, ' a1 ', xv, ' a2 ', yv, ' a3 ', zv, ' like', np.exp(loglike))
                if loglike > best:
                    best, a1b, a2b, a3b = loglike, xv, yv, zv
                    print('New best loglike', loglike, 'a1', xv, 'a2', yv, 'a3', zv, 'like', np.exp(loglike))
    print("--- {:g} seconds compute time ---".format(time.time() - start_time))
    # loglikegrid = loglikegrid.reshape(n[0]+1,n[1]+1,n[2]+1)
    gridToVTK("likelihood_fromgridsearch", x, y, z, cellData = {'likelihood_fromgridsearch':loglikegrid})
    like_reshaped = loglikegrid.reshape(loglikegrid.shape[0],-1)
    np.savetxt("LikelihoodFromGridSearch.txt", like_reshaped)
    np.savetxt("shapelikelihood.txt", loglikegrid.shape)
    return best, a2b+a3b-a1b, a1b+a3b-a2b, a1b+a2b-a3b, a1b+a2b+a3b, a1b, a2b, a3b

def randomfeasible(data, _):
    """Randomly generate feasible principal absorbances"""

    x = np.random.rand(3)
    d1, d2 = 0.9*np.min(data), 1.1*np.max(data)
    a2 = ((1.-x[0])*(d1**3)+x[0]*(d2**3))**(1/3.)
    a3 = a2*np.sqrt(x[1])
    a1 = (a2+a3)*x[2] + (1.-x[2])*a2

    #d3 = np.min(data)-delta
    #d4 = np.max(data)+delta
    #a3 = np.min([a2, d3])*np.sqrt(x[1])
    #a1 = (a2+a3)*x[2] + (1.-x[2])*np.max([a2, d4])

    return [a1, a2, a3]

def randomfeasibleloop(data, delta):
    """
    Randomly generate set of principal absorbances
    feasible w.r.t model and data constraints
    """

    a = randomfeasible(data, delta)
    count = 0
    while (checkPDF(a, data, delta)) and (count < 1000):
        a = randomfeasible(data, delta)
        count += 1
    return a

def checkfeasible(a):
    """check model feasibility of principal absorbances"""

    return (a[0] - a[1] <= a[2]) & (a[2] <= a[1]) & (a[1] <= a[0])

def initialsimplex(data, delta):
    """Randomly generate feasible initial simplex for principal absorbances"""

    np.random.seed(61254557)
    out = np.zeros([4, 3])
    out = [randomfeasibleloop(data, delta) for _ in out]

    return out

def localsearch(data, delta):
    """Perform local optimisation for principal absorbances"""

    mthd = 'Powell'
    mthd = 'Nelder-Mead'
    if mthd == 'Nelder-Mead':
        print('\nOptimisation by Nelder-Mead simplex algorithm')
        np.random.seed(61254557)
        #xsimp = initialsimplex(data, delta) # initialize starting simplex using random feasible points
        #print(' initial simplex:', xsimp)
        x0 = [np.max(data), 3.0*np.mean(data)-np.min(data)-np.max(data), np.min(data)]
        # # initialize starting point to Sambridge et al. (2008) solution
        x0 = randomfeasibleloop(data, delta) # initialize starting point to random feasible
        start_time = time.time()
        res = minimize(opt_func, x0, args=(data, delta), method='Nelder-Mead', options={'xtol': 0.01, 'disp': True})
        # options={'xtol': 0.01, 'disp': True, 'initial_simplex': xsimp}) # call used if initializing simplex
        end_time = time.time()
    elif mthd == 'Powell':
        print('\nOptimisation by Powell algorithm')
        #x0 = [np.max(data), 3.0*np.mean(data)-np.min(data)-np.max(data), np.min(data)]
        # # initialize starting point to Sambridge et al. (2008) solution
        np.random.seed(61254557)
        x0 = randomfeasibleloop(data, delta) # initialize starting point to random feasible
        #x0 = randomfeasible(data) # initialize starting point to random feasible
        #print(x0)
        start_time = time.time()
        res = minimize(opt_func, x0, args=(data, delta), method='Powell', options={'xtol': 0.01, 'disp': True})
        end_time = time.time()
    else:
        raise ValueError("Function localsearch: Unknown 'mthd' specified.")
    print("--- {:g} seconds compute time ---".format(end_time - start_time))
    out = [res.fun, res.x[1]+res.x[2]-res.x[0], res.x[0]+res.x[2]-res.x[1],
           res.x[0]+res.x[1]-res.x[2], res.x[0]+res.x[1]+res.x[2], res.x[0], res.x[1], res.x[2]]
    return out

def plotCond(data, delta, sol):
    """plot conditional PDFs for principal absorbances similar to Figure 9 of Jackson et al. (2018)"""

    plt.rcParams["figure.figsize"] = (8.0, 3.0)
    # These plot limits can be adjusted for increased resolution
    a2_start = 0.9*np.min(data)  # set limits of a2 for plot
    a2_stop = 1.1*np.max(data)   # set limits of a2 for plot
    a3_start = 0.0               # set lower limit of a3 using eqn. (41) Jackson et al. (2018) for plot
    a3_stop = a2_stop            # set upper limit of a3 using eqn. (40) Jackson et al. (2018) for plot
    a1_start = a2_start          # set lower limit of a1 using eqn. (41) Jackson et al. (2018) for plot
    a1_stop = 2.0*a2_stop        # set upper limit of a1 using eqn. (40) Jackson et al. (2018) for plot
    a1 = np.linspace(a1_start, a1_stop, 100) # a1 discretized values
    a2 = np.linspace(a2_start, a2_stop, 100) # a2 discretized values
    a3 = np.linspace(a3_start, a3_stop, 100) # a3 discretized values

    _, (ax1, ax2, ax3) = plt.subplots(1, 3)
    data1 = list(map(lambda x: np.exp(-opt_func([x, sol[1], sol[2]], data, delta)), a1))
    ax1.plot(a1, data1) # plot similar to 9a of Jackson et al. (2018)
    ax1.set_title('Conditional PDF for $a_1$')
    ax1.set_xlabel('$a_1$')
    ax1.set_ylabel('P($a_1|Q, a_2, a_3$)')
    #
    data1 = list(map(lambda x: np.exp(-opt_func([sol[0], x, sol[2]], data, delta)), a2))
    ax2.plot(a2, data1) # plot similar to 9b of Jackson et al. (2018)
    ax2.set_title('Conditional PDF for $a_2$')
    ax2.set_xlabel('$a_2$')
    ax2.set_ylabel('P($a_2|Q, a_1, a_3$)')
    #
    data1 = list(map(lambda x: np.exp(-opt_func([sol[0], sol[1], x], data, delta)), a3))
    ax3.plot(a3, data1) # plot similar to 9c of Jackson et al. (2018)
    ax3.set_title('Conditional PDF for $a_3$')
    ax3.set_xlabel('$a_3$')
    ax3.set_ylabel('P($a_3|Q, a_1, a_2$)')
    plt.tight_layout()
    plt.show()
    plt.savefig("Figure2.pdf")
    #
    return

def Atoa(A):
    """utility routine to convert big A absorbance to little a absorbance"""
    return 0.5*(A[2]+A[1]), 0.5*(A[2]+A[0]), 0.5*(A[0]+A[1]), np.sum(A)

def atoA(a):
    """utility routine to convert big A absorbance to little a absorbance"""
    return a[1]+a[2]-a[0], a[0]+a[2]-a[1], a[1]+a[0]-a[2], np.sum(a)

def PlotPQ():
    """Plot P(Q) figures similar to Figure 5.2 of Jackson et al. (2018)"""
    plt.subplot(1, 3, 1)
    data1 = list(map(lambda x: pdfsmooth(x, 3.0, 2.5, 1.0, 0.01), np.arange(1-0.01, 3+0.01, 0.01)))
    plt.plot(np.arange(1-0.01, 3+0.01, 0.01), data1) # plot 5.1 of Jackson et al. (2018)
    plt.title('$\\Delta=0.01$')
    plt.xlabel('Q')
    plt.ylabel('P(Q)')
    #
    data2 = list(map(lambda x: pdfsmooth(x, 3.0, 2.5, 1.0, 0.05), np.arange(1-0.01, 3+0.01, 0.01)))
    plt.subplot(1, 3, 2)
    plt.plot(np.arange(1-0.01, 3+0.01, 0.01), data2) # plot 5.2 of Jackson et al. (2018)
    plt.title('$\\Delta=0.05$')
    plt.xlabel('Q')
    plt.ylabel('P(Q)')
    #
    data3 = list(map(lambda x: pdfsmooth(x, 3.0, 2.5, 1.0, 1.0), np.arange(0.01, 4+0.01, 0.02)))
    plt.subplot(1, 3, 3)
    plt.plot(np.arange(0.01, 4+0.01, 0.02), data3) # plot 5.3 of Jackson et al. (2018)
    plt.title('$\\Delta=1.0$')
    plt.xlabel('Q')
    plt.ylabel('P(Q)')

    plt.tight_layout()
    plt.show()
    plt.savefig("Figure1.pdf")

def main():
    """
    Main body of script. The above routines are suitable to be imported into your own python calling script
                         with a call like `import absorbance'
                         The example calling script below is only executed if this file is itself evaluated by a pythion interpretor.
    """
    plt.rcParams["figure.figsize"] = (8.0, 3.0)

    # Parameters for user to edit as necessary
    datatype = 'Excel'                          # choose source of data, either 'Testset'= specified below or 'Excel' read in from file
    optmeth = 'Grid'                            # Type of optimisation method 'Grid' or 'Simplex',
    #optmeth = 'Simplex'                         # Type of optimisation method 'Grid' or 'Simplex',
                                                # NB: Simplex is fast, grid is very slow but foolproof with large na1, na2, na3.
    Ndsets = [10, 20, 60, 300, 450, 600, 750, 1000] # List of data set sizes to perform optimisation over (only used if datatype = 'Excel')
    Ndsets = [10, 20]                           # List of data set sizes to perform optimisation over (only used if datatype = 'Excel')
    Ndsets =  [20]
    delta = 1                                 # data error size as described in Jackson et al. (2018) > 25, 25, 25 for more accuracy
    res = 20
    na1, na2, na3 = res, res, res                     # Set discretization levels for grid search (only used if optmeth='Grid')
                                                # These values should be > 30, 30, 30 for serious application.
    DoPlotPQ = True                               # Plot P(Q) figures similar to Figure 5.2 of Jackson et al. (2018)
    PlotConditionals = True                     # Plot conditional probability distributions P(a1|Q, a2, a3) for each principal absorbance
                                                # similar to Figure 9 of Jackson et al. (2018)
    # End of section where users can edit control parameters

    #
    # calculate P(Q) using formulae for three cases of observational error and plot them
    #
    # Here is an example of the pdf for a=(3.0, 2.5, 1.0) and delta=0.01
    if DoPlotPQ:  # set to True to see plot similar to Figure 5.2 of Jackson et al. (2018); False to ignore
        PlotPQ()
    #
    # Set up test data corresponding to that used in Jackson et al. (2018)
    if datatype == 'Testset': # small demonstration dataset
        data_all = np.array([44.15967, 49.92802, 43.72898, 34.96102, 37.11223, 48.20369, 44.02242, 32.64254,
                             39.17191, 39.30958, 41.48838, 60.79360, 27.77383, 62.43976, 42.75821, 30.67905,
                             61.95585, 25.06300, 33.80495, 42.13757, 36.95285, 29.79391, 26.66335])
        Ndsets = [len(data_all)]
#       Atruth = [1.89, 45.6, 80.94]
#       atruth = Atoa(Atruth)
    elif datatype == 'Excel':
        # read in absorbance data from excel file here (Default is synthetic data of Jackson et al. (2018))
        exceldata = pd.read_excel('Absorbance_testdata.xlsx',sheet_name = 'Abs_data_uniform_errors_1000') # Excel filename and sheet for data
        data_all = exceldata.loc[:, 'Delta=1.0'] # read excel column of Absorbance values
    else:
        raise ValueError("Function main: Unknown 'datatype' specified.")

    # Find principal absorbances which maximize the likelihood defined in Jackson et al. (2018)

    for nv in Ndsets:                                           # list of data set sizes to consider
        dataO = np.array(data_all[:nv])                         # select data subset
        data = dataO[np.argsort(-np.abs(dataO-np.mean(dataO)))] # sort data
        print('\nTest data: Number of absorbance values =', len(data),
              '\nResults using Sambridge et al. (2008)',
              '\n1 =', np.max(data), 'a2 =', 3.0*np.mean(data)-np.min(data)-np.max(data), 'a3 =', np.min(data),
              '\nA_a =', 3.0*np.mean(data)-2.0*np.max(data),
              ' A_b =', 2.0*(np.min(data)+np.max(data))-3.0*np.mean(data),
              ' A_c =', 3.0*np.mean(data)-2.0*np.min(data))

        if optmeth == 'Simplex': # optimsation using Nealder-Mead simplex algorithm
            sol = localsearch(data, delta)  # minimize -ve log likelihood
            print('\nResults of local optimisation for maximum Likelihood of Jackson et al. (2018)')
        elif optmeth == 'Grid': # optimsation using grid search over chosen discretization
            sol = gridsearch(data, delta, [na1, na2, na3])   # maximize log likelihood   					#!!!!!!
            print('\nResults of grid search for maximum Likelihood of Jackson et al. (2018)')
        else:
            raise ValueError("Function main: Unknown 'optmeth' specified")
        print('A_a =', sol[1], ' A_b =', sol[2], ' A_c =', sol[3], ' A_tot =', sol[4], ' log Like', sol[0])

    # Plot conditional PDF of each of (a1, a2, a3) through maximum likelihood values of the other two.
    # Similar plot to Figure 9 of Jackson et al. (2018) from which error estimates in principal absorbances can be estimated.

    if PlotConditionals:
        print('Building conditional plot of Likelihood through solution...')
        plotCond(data, delta, sol[5:8]) # plot conditional PDFs for principal absorbances through solution.
        
    

if __name__ == "__main__":
    main()
