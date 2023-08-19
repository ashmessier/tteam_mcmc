import batman 
import numpy as np
import scipy.stats as stats
from pathlib import Path
import os
import emcee

# lightcurve model with batman 
def lc(pars, data):
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

# log likelihood function 
def lnp(pars, priors, data):
    model = lc(pars, data)

    # Calculate the log-likelihood
    log_prob_data = 0.0
    log_prob_data += np.sum(stats.norm.logpdf(data["flux"] - model, loc=0, scale=data["error"]))

    # Calculate the log-likelihood of prior values
    log_prob_prior = 0.0
    log_prob_prior += stats.norm.logpdf(pars[0], loc=priors[0][0], scale=priors[0][1]) # t0
    log_prob_prior += stats.norm.logpdf(pars[1], loc=priors[1][0], scale=priors[1][1]) # log period
    log_prob_prior += stats.norm.logpdf(pars[2], loc=priors[2][0], scale=priors[2][1]) # RpRsk
    log_prob_prior += stats.norm.logpdf(pars[3], loc=priors[3][0], scale=priors[3][1]) # log_ars
    log_prob_prior += stats.norm.logpdf(pars[4], loc=priors[4][0], scale=priors[4][1]) # cosi
    log_prob_prior += stats.norm.logpdf(pars[5], loc=priors[5][0], scale=priors[5][1]) # esinw
    log_prob_prior += stats.norm.logpdf(pars[6], loc=priors[6][0], scale=priors[6][1]) # ecosw
    log_prob_prior += stats.norm.logpdf(pars[7], loc=priors[7][0], scale=priors[7][1]) # U1
    log_prob_prior += stats.norm.logpdf(pars[8], loc=priors[8][0], scale=priors[8][1]) # U2

    # Combine log-likelihoods
    log_likelihood_value = log_prob_data + log_prob_prior

    return log_likelihood_value

def minimize_lnp(pars, priors, data):
    return -1 * lnp(pars, priors, data)
    
def run_mcmc(pars, priors, data):
    nburn = 500
    nprod = 2000
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

    np.save("flatsample", flat_sample)  # saves flatsamples array

    return flat_sample

# gets parameters from flatsample
def flatsample_pars(flat_sample):
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
    depth = RpRs ** 2 * 1000000
    np.column_stack((flat_sample, depth))

    # initializes lists to iterate over to find median values, STDs from flat_sample -> get best fit parameters from mcmc + error
    value_list = [T0, log_period, RpRs, log_a, cosi, esinw, ecosw, u1, u2, depth]
    mcmc_pars = []
    mcmc_stds = []

    # iterates over lists to calculate median value, std
    for i, parameter in enumerate(value_list):
        median_parameter = np.median(parameter)
        mcmc_pars.append(median_parameter)

        sig1 = np.percentile(parameter, 16)  # one sigma away from median on left
        sig2 = np.percentile(parameter, 84)  # "", right
        diff_sig1 = median_parameter - sig1  # std on left
        diff_sig2 = sig2 - median_parameter  # "", right
        avg_std = (diff_sig1 + diff_sig2) / 2  # average spread (accounts for small skew?)
        mcmc_stds.append(avg_std)

    return mcmc_pars, mcmc_stds