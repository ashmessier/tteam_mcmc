import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.stats import sigma_clip
from tt_mcmc_functions import *
from scipy.optimize import minimize
import corner
from matplotlib import rc
import sys
import os
import shutil
rc("font",**{"family":"serif", "serif":["Times"]})
rc("text", usetex=True)

# loads in observation dataframe 
# filename = input("Enter exoplanet filename (csv): ")

run = True

# these will be input for final project, but are set for now
filename = "TOI1516b-2022sep23_measurements.xls - TOI1516b-2022sep23_measurements.xls.csv"
exoplanet_name = find_exoplanetname(filename)

# makes exoplanet name into possible directory name - remove spaces -> _
updated_exoplanet_name = exoplanet_name.replace(" ", "_")

# creates new directory for specific planet 
planet_dir = Path(f"{updated_exoplanet_name}")
planet_dir.mkdir(parents=True, exist_ok=True)

# moves data file into the new directory
source_path = filename  # Replace with the actual source file path
destination_path = f'{updated_exoplanet_name}/'  # Replace with the actual destination directory path
file_path = os.path.join(destination_path, source_path)

# moves data file into new diretory if not already there 
if os.path.exists(file_path) == False:
    shutil.move(source_path, destination_path)

# read in data 
df = pd.read_csv(f"./{updated_exoplanet_name}/{filename}")

 # normalizes data using mean of baseline (first 30 points?)
norm_flux = df["rel_flux_T1"] / np.mean(df["rel_flux_T1"][:30])
norm_err = df["rel_flux_err_T1"] / np.mean(df["rel_flux_T1"][:30])

# runs sigma clip to 3 sigma to remove outliers 
sigmaclip = sigma_clip(norm_flux, 3, masked=True)
outlier_mask = sigmaclip.mask

# initilaizes data dictionary with useful columns 
data = {"time":df["BJD_TDB"][~outlier_mask], 
        "flux":norm_flux[~outlier_mask],
        "error":norm_err[~outlier_mask],
        "airmass": df["AIRMASS"][~outlier_mask]}
if run: 
# exoplanet archive prior formatting ======================================================

    # grabs columns of exoplanet archive corresponding to target 
    exoplanet_archive = pd.read_csv("exoplanet_archive.csv")
    
    # finds columns corresponding to planet name 
    data_mask = exoplanet_archive["pl_name"].isin([exoplanet_name])
    data_cols = exoplanet_archive[data_mask]
    del data_cols["pl_name"] # deletes planet name column, now extraneous 

    # takes the average of all data columns to get best informed parameters/priors
    pars_df = data_cols.mean() # usse average for all 
    pars_df["pl_tranmid"] = data_cols["pl_tranmid"].median() # EXCEPT transit center time; takes median bc can't be average 
    print(pars_df["pl_tranmid"])

    # list of key values 
    all_keys = [key for key in pars_df.keys() if key[0] == "p"]

    # returns keys where data_cols["pl_name"] has a nan value 
    nan_keys = [key for key in pars_df.index if np.isnan(pars_df[key])]
    non_nan_keys = [key for key in all_keys if key not in nan_keys]

    # dictionary of prior guesses IF exoplanet table value nan
    nan_prior_dict = {key : exoplanet_archive[key].median() for key in all_keys[1:]}    
    if "pl_tranmid" in nan_keys : nan_prior_dict["pl_tranmid"] = data["time"][np.argmin(data["flux"])] # initial guess for transit center time at expected center time 
    
    # assigns values to all keys from either exoplanet archive or priors dictionary 
    prior_dict_1 = {key: pars_df[key] for key in non_nan_keys}
    prior_dict = {key: nan_prior_dict[key] for key in nan_keys}
    prior_dict.update(prior_dict_1)
    prior_dict["pl_tranmid"] = data["time"][np.argmin(data["flux"])]
    
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
    priors_mod[9] = [0.01, 0.01] # slope from airmass (arbitrary?)

    # sets initial parameters as values from exoplanet archive
    pars = priors_mod[:,0] 

    # runs nelder-meade minimization on model, data to get set of initial parameters
    minimize_results = minimize(minimize_lnp, pars, args=(priors_mod, data), method="Nelder-Mead")
    minimize_pars = minimize_results.x
    
    #  runs mcmc on all parameters 
    flat_sample = run_mcmc(minimize_pars, priors_mod, data)
    np.save(f"{updated_exoplanet_name}/flatsample", flat_sample)  # saves flatsamples array to specific planet file 

else:
    flat_sample = np.load(f"{updated_exoplanet_name}/flatsample.npy")

 # plots cornerplot 
labels = ["T0", "log_period", "RpRs", "log_ars", "cosi",
            "esinw", "ecosw", "u1", "u2", "airmass_slope"]
fig = corner.corner(flat_sample, labels=labels, show_titles=True)
plt.tight_layout()
fig_path = planet_dir / f"{updated_exoplanet_name}_cornerplot.png"
plt.savefig(fig_path)

# extracts final parameters, saves to text file 
mcmc_pars, mcmc_stds = flatsample_pars(flat_sample)
labels.append("depth (ppt)")

output_file_path = f'{planet_dir}/final_parameters.txt'
with open(output_file_path, 'w') as file:
    for i, label in enumerate(labels): 
        file.write(f"{label}: {mcmc_pars[i]} +/- {mcmc_stds[i]} \n")
    
# calculates residuals, RMS 
final_lc = make_model(mcmc_pars, data) # includes airmass 
detrended_lc = lc(mcmc_pars, data) # just lc 
airmass_detrended_data = data['flux'] / (1 + mcmc_pars[9] * (data["time"] - np.median(data["time"]))) # removes airmass trend from data 
residuals = final_lc - data["flux"]  # calculates residuals
RMS = np.sqrt(((residuals) ** 2).mean())
print("RMS of residuals:", RMS)

# calculates depth as difference between baseline and transit center 
baseline = detrended_lc.max()
center = detrended_lc.min()
depth_absolute = (baseline - center)/baseline * 1000
depth_abs_err = np.std(residuals) * 1000
print(f"absolute depth: {depth_absolute} +/- {depth_abs_err}")
print(f"Rp/R* depth: {mcmc_pars[10]} +/- {mcmc_stds[10]}")
    
# plots final lightcurve w all the other stuff on the plot AiJ uses <3 
depth = mcmc_pars[10] / 1000
plt.figure(figsize=(10, 10))

plt.scatter(data["time"], airmass_detrended_data, color="blue", s=4, label="Airmass Detrended Flux")
plt.errorbar(data["time"], airmass_detrended_data, yerr=data["error"], color="blue", alpha=0.1)
plt.plot(data["time"], detrended_lc, color="blue")

plt.scatter(data["time"], residuals + 0.945, s=4, color="fuchsia", label=f"residuals, RMS={round(RMS, 4)}")

plt.ylim(1 - 14 * depth, 1 + 8 * depth)
plt.legend(loc="upper center", fontsize=15)

plt.xlabel("BJD_{TBD}")
plt.ylabel("Relative Flux")

fig_path = planet_dir / f"{updated_exoplanet_name}_lightcurve.png"
plt.savefig(fig_path)

# future work

# - make sure this work with other exoplanets (specifically one where there are missing archive inputs)
# - make prettier plots with final model, residuals 
# - figure out how to interpert RMSE of residuals 
# - Ultimae goal - some sort of package?? where you can just grab it from github, install packages, and run to process a lightcurve, get priors
#       Figure out how to give inputs in command line when run package via python lightcurve_fit.py:
#           - nburn, nprod
#       Figure out more convenient pipeline for getting data from drive -> ask someone? 
# - Use this to determine minimum transit depth we can observe -> go from depth (ppm) to milimags? 