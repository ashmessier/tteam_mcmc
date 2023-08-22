import batman 
import numpy as np
import scipy.stats as stats
from pathlib import Path
import os
import emcee
import sys

# function to find exoplanet name from filename 
def find_exoplanetname(filename):
    
    """ Extracts exoplanet name in the correct format to reference the exoplanet archive table. Only currently 
    working for exoplanets in the following catalogs: "TOI", "XO", "WASP","TRES","Quatar","OGLE","HAT","K2","CoRoT","HD", "GJ"

    Returns:
        string: exoplanet archive formatted exoplanet name 
    """
    # returns part of string before a "b" character 
    filename = filename.upper() # uppercases whole string 
    filename = filename.replace("-", "") # gets rid of dashes 
    planetname = filename.split("B", 1)[0] # splits on B character, takes first part 
    
    # finds first index of a number, returns that index 
    numbers = [index for index, char in enumerate(planetname) if char.isdigit()]

    # Split the string into parts based on the numeric part
    words = planetname[:numbers[0]]
    numbers = planetname[numbers[0]:]
    
    with_dash = ["TOI", "XO", "WASP","TRES","Quatar","OGLE","HAT","K2","CoRoT"]
    no_dash = ["HD", "GJ"]
    
    if words in no_dash:
        return f"{words} {numbers} b"
    if words in with_dash:
        return f"{words}-{numbers} b"
    if words == "K":
        numbers = planetname.split("-", 1)
        return f"K2-{numbers} b"
    if words == "HATP":
        return f"HAT-P-{numbers} b"
    
def get_priors(prior_dict, scale=1):
    """Modifies priors from [t0, period, RpRs, aRs, i, e, w] -> [t), log(period), RpRs, log(ars), cosi, esinw, ecosw], 
    adds priors for u1, u2 (limb darkening priors), and airmass trend.

    Args:
        prior_dict (dict): dictionary with keys corresponding to the above values and associated errors from the NASA exoplanet archive. 
        scale (float): value to increase/decrease uncertainties by to improve fitting quality. Default is no scaling. 

    Returns:
        array:9x2 array compatable with the lc, lnp, and run_mcmc functioss containing initial parameters, priors to use in lc model fitting.
    """
    
    # assigns variables to values to perform modifications to
    priors = np.zeros((7, 2))
    scale = 10
    priors[0] = [prior_dict["pl_tranmid"], prior_dict["pl_tranmiderr1"] * scale]  # T0
    priors[1] = [prior_dict["pl_orbper"], prior_dict["pl_orbpererr1"] * scale] # per
    priors[2] = [prior_dict["pl_ratror"], prior_dict["pl_ratrorerr1"] * scale] # Rp/R*
    priors[3] = [prior_dict["pl_ratdor"], prior_dict["pl_ratdorerr1"] * scale] # a/R*
    priors[4] = [prior_dict["pl_orbincl"], prior_dict["pl_orbinclerr1"] * scale]   # i
    priors[5] = [prior_dict["pl_orbeccen"], prior_dict["pl_orbeccenerr1"] * scale]   # e
    priors[6] = [prior_dict["pl_orblper"], prior_dict["pl_orblpererr1"] * scale]  # w

    # modified parameters
    ecosw = priors[5][0] * np.cos(priors[6][0] * np.pi / 180)  # e cos w
    esinw = priors[5][0] * np.sin(priors[6][0] * np.pi / 180)  # e sin w
    cosi = np.cos(priors[4][0] * np.pi / 180)
    log10per = np.log10(priors[1][0])
    log10ars = np.log10(priors[3][0])

    # errors
    sig_esinw = np.sqrt((np.sin(priors[6][0] * np.pi / 180)) ** 2 * (priors[5][1] * np.pi / 180) ** 2 + (
                priors[5][0] * np.pi / 180) ** 2 * (np.cos(priors[6][0] * np.pi / 180)) ** 2 * (
                                    priors[6][1] * np.pi / 180) ** 2)
    sig_ecosw = np.sqrt((np.cos(priors[6][0] * np.pi / 180)) ** 2 * (priors[5][1] * np.pi / 180) ** 2 + (
                priors[5][0] * np.pi / 180) ** 2 * (np.sin(priors[6][0] * np.pi / 180)) ** 2 * (
                                    priors[6][1] * np.pi / 180) ** 2)
    sig_cosi = np.sin(cosi) * priors[4][1]
    sig_log10per = 1 / (np.log(10)) * priors[1][1] / priors[1][0]
    sig_log10ars = 1 / (np.log(10)) * priors[3][1] / priors[3][0]

    # creates modified priors array
    priors_mod = np.zeros((10, 2))
    priors_mod[0] = priors[0] # t0
    priors_mod[1] = [log10per, sig_log10per]  # orbital period
    priors_mod[2] = priors[2] # Rp/R* 
    priors_mod[3] = [log10ars, sig_log10per]  # a/R*
    priors_mod[4] = [cosi, sig_cosi]  # i
    priors_mod[5] = [esinw, sig_esinw]  # e
    priors_mod[6] = [ecosw, sig_ecosw]  # w
    priors_mod[7] = [0.3, 0.01]  # limb darkening u1 (AiJ initial guess )
    priors_mod[8] = [0.3, 0.01]  # limb darkening u2 
    priors_mod[9] = [0.01, 0.01] # slope from airmass 
    
    return priors_mod 

# lightcurve model with batman 
def lc(pars, data):
    
    """Creates a lightcurve model.
    Inputs: 
    pars (list): parameter guesses in the form pars = [t0, log(period), RpR*, log(a/R*), cosi, esinw, ecosw, u1, u2, airmass_slope]
    data (dictionary): containing an entry "time" of BJD_TBD time values spanning the duration of transit observation

    Returns:
        array: array of flux values corresponding to transit lightcurve predicted by input parameters of same length of time array. 
    """
    
    params = batman.TransitParams()  # object to store transit parameters
    params.t0 = pars[0]  # time of inferior conjunction
    params.per = 10 ** pars[1]  # orbital period
    params.rp = pars[2] # rprs
    params.a = 10 ** pars[3]  # semi-major axis (in units of stellar radii)
    params.inc = np.arccos(np.fabs(pars[4])) * (180 / np.pi)  # orbital inclination (in degrees)
    params.ecc = np.sqrt(pars[6] ** 2 + pars[5] ** 2)  # eccentricity
    params.w = np.arctan2(pars[5], pars[6]) * (180 / np.pi)  # longitude of periastron (in degrees)
    params.limb_dark = "quadratic"  # limb darkening model
    params.u = [pars[7], pars[8]]
     
    m = batman.TransitModel(params, np.array(data["time"]))  # initializes model
    model = m.light_curve(params)

    return model

def make_model(pars, data):
    ramp = 1 + pars[9] * (data["time"] - np.median(data["time"]))
    return lc(pars, data) * ramp

# log likelihood function 
def lnp(pars, priors, data):
    
    """ Calculates log likelihood value for a set of input parameters to the "lc" function. For use in minimization and MCMC model fitting.

    Args:
        pars (list): parameter guesses in the form pars = [t0, log(period), RpR*, log(a/R*), cosi, esinw, ecosw, u1, u2]
        priors (list): Prior measurements of system parameters, extracted from Exoplanet Archive, with associated uncertainty. 
        follows same format as pars array, but as a (9, 2) shape array to fit errors. 
        data (dict): Dictionary containing time, flux, and error measurements of a transit observation. Each entry is an numpy array. 

    Returns:
        float: value representing relative likelihood that input set of pars represents the bestfit parameters to the data given the data and 
        prior measurements. The larger the value, the more likely the set of parameters. 
    """
    if pars[4] < 0: return -np.inf # chaces lnp function away from cosi < 1 bc that is illegal 
    
    model = make_model(pars, data)

    # Calculate the log-likelihood
    log_prob_data = 0.0
    log_prob_data += np.sum(stats.norm.logpdf(data["flux"] - model, loc=0, scale=data["error"]))

    # Calculate the log-likelihood of prior values
    log_prob_prior = 0.0
    log_prob_prior += stats.norm.logpdf(pars[0], loc=priors[0][0], scale=priors[0][1]) # t0
    log_prob_prior += stats.norm.logpdf(pars[1], loc=priors[1][0], scale=priors[1][1]) # log period
    log_prob_prior += stats.norm.logpdf(pars[2], loc=priors[2][0], scale=priors[2][1]) # RpRs
    log_prob_prior += stats.norm.logpdf(pars[3], loc=priors[3][0], scale=priors[3][1]) # log_ars
    log_prob_prior += stats.norm.logpdf(pars[4], loc=priors[4][0], scale=priors[4][1]) # cosi
    log_prob_prior += stats.norm.logpdf(pars[5], loc=priors[5][0], scale=priors[5][1]) # esinw
    log_prob_prior += stats.norm.logpdf(pars[6], loc=priors[6][0], scale=priors[6][1]) # ecosw
    log_prob_prior += stats.norm.logpdf(pars[7], loc=priors[7][0], scale=priors[7][1]) # U1
    log_prob_prior += stats.norm.logpdf(pars[8], loc=priors[8][0], scale=priors[8][1]) # U2
    #log_prob_prior += stats.norm.logpdf(pars[9], loc=priors[9][0], scale=priors[9][1]) # airmass slope

    # Combine log-likelihoods
    log_likelihood_value = log_prob_data + log_prob_prior

    return log_likelihood_value

def minimize_lnp(pars, priors, data):
    
    """ log likelihood function used in scipy's "minimize" function. 
    Args:
        pars (list): parameter guesses in the form pars = [t0, log(period), RpR*, log(a/R*), cosi, esinw, ecosw, u1, u2]
        priors (list): Prior measurements of system parameters, extracted from Exoplanet Archive, with associated uncertainty. 
        follows same format as pars array, but as a (9, 2) shape array to fit errors. 
        data (dict): Dictionary containing time, flux, and error measurements of a transit observation. Each entry is an numpy array. 

    Returns:
        float: inverse of the log likelihood value calculated via the function "lnp". 
    """
    return -1 * lnp(pars, priors, data)
    
def run_mcmc(pars, priors, data):
    """ runs MCMC model fitting on data. 

    Args:
        pars (list): parameter guesses in the form pars = [t0, log(period), RpR*, log(a/R*), cosi, esinw, ecosw, u1, u2]
        priors (list): Prior measurements of system parameters, extracted from Exoplanet Archive, with associated uncertainty. 
        follows same format as pars array, but as a (9, 2) shape array to fit errors. 
        data (dict): Dictionary containing time, flux, and error measurements of a transit observation. Each entry is an numpy array. 

    Returns:
        numpy array: Array of size (nwalkers, len(pars)) representing walker positions per each parameter at the end of the MCMC run. Additionally saves 
        flat sample as .npy file for reference 
        """
        
    nburn = int(input("Number of burnin (reccomend 500): "))
    nprod = int(input("Number of product runs (reccomend 2000): "))
    ndim = len(pars)
    nwalkers = 2*ndim

    pos = np.empty((nwalkers, ndim))
    pos_errscale = 1
    for i, par in enumerate(pars):
        pos[:, i] = np.random.normal(par, priors[i][1]/pos_errscale, nwalkers) # priors errscale used to be /10

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnp,
                                    args=(priors, data))
    pos, _, _ = sampler.run_mcmc(pos, nburn, progress=True)  # runs mcmc on burnin values
    sampler.reset()
    pos, _, _ = sampler.run_mcmc(pos, nprod, progress=True)  # runs from positions of burnin values

    # creates flat_sample array for corner plot generation
    flat_sample = sampler.get_chain(discard=0, thin=1, flat=True)  # flattens list of samples
    
    # checks to see if too few points to create valid contours was printed 
    
    
    return flat_sample

# gets parameters from flatsample
def flatsample_pars(flat_sample):
    
    """ Calculates final parameters, uncertainties from a flatsample array in the form (nwalkers, len(pars)). Final parameters
    are equal to the median of the final walker array per each parameter. Uncertainties reperesent 1 sigma range of distribution of final walker positions. 

    Returns:
        mcmc_pars (array): array in the form [t0, log(period), RpR*, log(a/R*), cosi, esinw, ecosw, u1, u2] representing bestfit parameters from MCMC fitting
        mcmc_stds (array): array in the form [t0, log(period), RpR*, log(a/R*), cosi, esinw, ecosw, u1, u2] representing uncertainties on parameters from MCMC fitting 
    """
    # values from mcmc fitting
    T0 = flat_sample[:, 0]
    log_period = flat_sample[:, 1]
    RpRs = flat_sample[:, 2]
    log_a = flat_sample[:, 3]
    cosi = flat_sample[:, 4]
    esinw = flat_sample[:, 5]
    ecosw = flat_sample[:, 6]
    u1 = flat_sample[:, 7]
    u2 = flat_sample[:, 8]
    airmass_slope = flat_sample[:, 9]
    depth = RpRs ** 2 * 1000 # in ppt 
    np.column_stack((flat_sample, depth))

    # initializes lists to iterate over to find median values, STDs from flat_sample -> get best fit parameters from mcmc + error
    value_list = [T0, log_period, RpRs, log_a, cosi, esinw, ecosw, u1, u2, airmass_slope, depth]
    mcmc_pars = []
    mcmc_stds = []

    # iterates over lists to calculate median value, std
    for parameter in value_list:
        median_parameter = np.median(parameter)
        mcmc_pars.append(median_parameter)

        sig1 = np.percentile(parameter, 16)  # one sigma away from median on left
        sig2 = np.percentile(parameter, 84)  # "", right
        diff_sig1 = median_parameter - sig1  # std on left
        diff_sig2 = sig2 - median_parameter  # "", right
        avg_std = (diff_sig1 + diff_sig2) / 2  # average spread (accounts for small skew?)
        mcmc_stds.append(avg_std)

    return mcmc_pars, mcmc_stds