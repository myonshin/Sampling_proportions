'''
@Author: Myong Jong Shin (myonshin AT iu DOT edu)
Last Edited on 3:21 PM 3/23/2025
Copyright Reserved
'''

import numpy as np
import scipy.stats as stats
import random
random.seed(2400)

def Wald_CI(nn, kk, confidence_level):
# This function replicates 'Wald confidence interval', Eq 1 in Agresti & Coull(1998,TAS)

    p_hat = kk/nn
    se = np.sqrt(p_hat*(1-p_hat)/nn)
    l_bound = p_hat - stats.norm.ppf(1-(1-confidence_level)/2)*se
    u_bound = p_hat + stats.norm.ppf(1-(1-confidence_level)/2)*se
    return l_bound,u_bound

def Wilson_CI(nn, kk, confidence_level):
# AKA the score test interval
    p_hat = kk/nn
    se_hat_sq = p_hat*(1-p_hat)/nn
    crit = stats.norm.ppf(1-(1-confidence_level)/2)
    omega = nn/(nn + crit**2)
    A = p_hat + crit**2/(2*nn)
    B = crit * np.sqrt(se_hat_sq + crit**2/(4*nn**2))

    l_bound = omega*(A-B)
    u_bound = omega*(A+B)

    return l_bound,u_bound

def Exact_CI(nn, kk, confidence_level):
# AKA Clopper-Pearson interval
    crit_1 = stats.f.ppf(q = (1-confidence_level)/2, dfn = 2*kk, dfd = 2*(nn-kk+1))
    crit_2 = stats.f.ppf(q = 1-(1-confidence_level)/2, dfn = 2*(kk+1), dfd = 2*(nn-kk))
    
    if kk == 0:
        l_bound = 0.0
    else:
        l_bound = (1+(nn-kk+1)/(kk*crit_1)) **(-1)
    if kk == nn:
        u_bound = 1.0
    else:
        u_bound = (1+(nn-kk)/((kk+1)*crit_2))**(-1)
    
    return l_bound, u_bound

def Wald_continuity_corr(nn, kk, confidence_level):
    p_hat = kk / nn
    se = np.sqrt(p_hat * (1 - p_hat) / nn)
    l_bound = p_hat - stats.norm.ppf(1 - (1 - confidence_level) / 2)*se - 0.5/nn
    u_bound = p_hat + stats.norm.ppf(1 - (1 - confidence_level) / 2)*se + 0.5/nn
    
    return l_bound, u_bound

def Score_continuity_corr(nn, kk, confidence_level):
    # method 1,2,3 and 4 all gives same answerl score_continuity_corr 0.965467303581;
    crit = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    '''
    #method 1
    l_bound = np.max( [0, kk-0.5+0.5*crit**2 - crit*np.sqrt(kk-0.5 - (kk-0.5)**2/nn + 0.25*crit**2) /(nn+crit**2 )] )
    u_bound = np.min( [1, ( kk+0.5+0.5*crit**2 + crit*np.sqrt(kk+0.5 - (kk+0.5)**2/nn + 0.25*crit**2) )/(nn+crit**2 )] )
    
    #method 2
    p_hat = kk / nn
    l_bound = np.max( [0, ( 2*kk+crit**2 -1 -crit*np.sqrt(crit**2-1/nn+4*kk*(1-p_hat)+4*p_hat-2) )/(2*nn+2*crit**2 )] )
    u_bound = np.min( [1, ( 2*kk+crit**2 +1 +crit*np.sart(crit**2-1/nn+4*kk*(1-p_hat)-4*p_hat+2) )/(2*nn+2*crit**2 )] )
    '''
    #method 3
    p_hat = kk / nn
    l_bound = np.max( [0, ( 2*kk+crit**2 -1 -crit*np.sqrt(crit**2-2-1/nn+4*p_hat*nn*(1-p_hat)+4*p_hat) )/( 2*nn+2*crit**2 )] )
    u_bound = np.min( [1, ( 2*kk+crit**2 +1 +crit*np.sqrt(crit**2+2-1/nn+4*p_hat*nn*(1-p_hat)-4*p_hat) )/( 2*nn+2*crit**2 )] )
    '''
    #method 4
    crit = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    omega = nn/(nn + crit**2)
    SE_hat_sq_1 = (kk/nn - 0.5/nn)*(1- (kk/nn - 0.5/nn))/nn
    A_1 = (kk/nn - 0.5/nn) + crit**2 / (2 * nn)
    B_1 = crit * np.sqrt(SE_hat_sq_1 + crit**2 / (4 * nn**2))
    SE_hat_sq_u = (kk/nn + 0.5/nn)* (1-(kk/nn + 0.5/nn))/nn
    A_u = (kk/nn + 0.5/nn) + crit**2 / (2 * nn)
    B_u = crit * np.sqrt(SE_hat_sq_u + crit**2 / (4 * nn**2))
    
    l_bound = np.max([0, omega * (A_1 - B_1)])
    u_bound = np-min([1, omega * (A_u + B_u)])
    '''
    return l_bound, u_bound


CIs = {
    Wald_CI: "Wald_CI ",
    Wilson_CI: "Wilson_CI ",
    Exact_CI: "Exact_CI ",
    Wald_continuity_corr: "Wald_continuity_corr ",
    Score_continuity_corr: "Score_continuity_corr "
}

def indicator(X, l_bound, u_bound):
    if l_bound <= X <= u_bound:
        return 1
    else:
        return 0

def bin_prob(n,k,p):
    binom_dist = stats.binom(n,p)
    prob = binom_dist.pmf(k)
    return prob

def CnP_functions(support,CI_func):
    CnP = 0
    for success in range(NN+1):
        l_bound, u_bound = CI_func(NN, success, confidence_level = 0.95)
        CnP = CnP + indicator(support, l_bound, u_bound)*bin_prob(NN, success, support)
    return CnP





#%% Table 1 of Agresti & Coull(1998,TAS) per chosen confidence interval formula (in this code, Wald)

NN = 30 # Number of experiments; change this value to replicate different columns of TABLE 1
sim  = 1000 # Integrand grid

# Integration method 1
# Calculates coverage probabilities

for CI_func, name in CIs.items():
    Cn_1 = 0
    for support in np.linspace(0,1,sim):
        Cn_1 = Cn_1 + CnP_functions(support,CI_func)*(1/sim)
    print(name, Cn_1)

'''
# Integration method 2
import scipy.integrate as integrate
Cn_2, error = integrate.quad(CnP_function,0,1)
print(Cn_2)

# Integration method 3
def trapezoidal_rules(f,a,b,n):
    h = (b-a)/n
    integral = (f(a) + f(b))/2
    for i in range(1,n):
        x = a + i*h
        integral += f(x)
    integral *= h
    return integral
Cn_3 = trapezoidal_rules(CnP_function, 0, 1, sim)
print(Cn_3)
'''