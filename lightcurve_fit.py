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
# print("")
# print("Enter exoplanet Archive planet name \n for example, HATP22 -> HAT-P-22: \n see archive for reference: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=PS&constraint=default_flag=1&constraint=disc_facility+like+%27%25TESS%25%27")
# print("")
# exoplanet_name = input("Name: ")

run = True

# these will be input for final project, but are set for now
filename = "HATP22B-2023apr02.csv"
exoplanet_name = "HAT-P-22 b"

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

 # normalizes data using mean of baseline (first 50 points?)
norm_flux = df["rel_flux_T1"] / np.mean(df["rel_flux_T1"][:15])
norm_err = df["rel_flux_err_T1"] / np.mean(df["rel_flux_T1"][:15])

# runs sigma clip to 3 sigma to remove outliers 
sigmaclip = sigma_clip(norm_flux, 3, masked=True)
outlier_mask = sigmaclip.mask

# initilaizes data dictionary with useful columns 
data = {"time":df["BJD_TDB"][~outlier_mask], 
        "flux":norm_flux[~outlier_mask],
        "error":norm_err[~outlier_mask],
        "airmass": df["AIRMASS"][~outlier_mask]}
if run: 
    # grabs columns of exoplanet archive corresponding to target 
    exoplanet_archive = pd.read_csv("exoplanet_archive.csv")

    data_mask = exoplanet_archive["pl_name"].isin([exoplanet_name])
    data_cols = exoplanet_archive[data_mask]
    del data_cols["pl_name"] # deletes planet name column, now extraneous 

    # takes the average of all data columns to get best informed parameters/priors
    pars_df = data_cols.mean() # usse average for all 
    pars_df["pl_tranmid"] = data_cols["pl_tranmid"].median() # EXCEPT transit center time; takes median bc can't be average 

    # assigns variables to values to perform modifications to
    priors = np.zeros((7, 2))
    priors[0] = [pars_df["pl_tranmid"], pars_df["pl_tranmiderr1"]]  # T0
    priors[1] = [pars_df["pl_orbper"], pars_df["pl_orbpererr1"]] # per
    priors[2] = [pars_df["pl_ratror"], pars_df["pl_ratrorerr1"]] # Rp/R*
    priors[3] = [pars_df["pl_ratdor"], pars_df["pl_ratdorerr1"]] # a/R*
    priors[4] = [pars_df["pl_orbincl"], pars_df["pl_orbinclerr1"]]   # i
    priors[5] = [pars_df["pl_orbeccen"], pars_df["pl_orbeccenerr1"]]   # e
    priors[6] = [pars_df["pl_orblper"], pars_df["pl_orblpererr1"]]  # w

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
    priors_mod = np.zeros((9, 2))
    priors_mod[0] = priors[0] # t0
    priors_mod[1] = [log10per, sig_log10per]  # orbital period
    priors_mod[2] = priors[2] # Rp/R* 
    priors_mod[3] = [log10ars, sig_log10per]  # a/R*
    priors_mod[4] = [cosi, sig_cosi]  # i
    priors_mod[5] = [esinw, sig_esinw]  # e
    priors_mod[6] = [ecosw, sig_ecosw]  # w
    priors_mod[7] = [0.53, 0.01]  # limb darkening u1 (sunlike, johnsonV)
    priors_mod[8] = [0.25, 0.01]  # limb darkening u2 

    # sets initial parameters as values from exoplanet archive
    pars = priors_mod[:,0] 

    # runs nelder-meade minimization on model, data to get set of initial parameters
    minimize_results = minimize(minimize_lnp, pars, args=(priors_mod, data), method="Nelder-Mead")
    minimize_pars = minimize_results.x
    
    #  runs mcmc on all parameters 
    flat_sample = run_mcmc(minimize_pars, priors_mod, data)

else:
    flat_sample = np.load("flatsample.npy")

 # plots cornerplot 
labels = ["T0", "log_period", "RpRs", "log_ars", "cosi",
            "esinw", "ecosw", "u1", "u2"]
fig = corner.corner(flat_sample, labels=labels, show_titles=True)
plt.tight_layout()
fig_path = planet_dir / f"{updated_exoplanet_name}_cornerplot.png"
plt.savefig(fig_path)

# extracts final parameters, saves to text file 
mcmc_pars, mcmc_stds = flatsample_pars(flat_sample)

output_file_path = f'{planet_dir}/final_parameters.txt'
with open(output_file_path, 'w') as file:
    for i, label in enumerate(labels): 
        file.write(f"{label}: {mcmc_pars[i]} +/- {mcmc_stds[i]}")
    
# calculates residuals, RMS 
final_lc = lc(mcmc_pars, data)
residuals = final_lc - data["flux"] 
RMS = np.sqrt(((residuals) ** 2).mean())
print("RMS of residuals:", RMS)
    
# plots final lightcurve w all the other stuff on the plot AiJ uses <3 
depth = mcmc_pars[9] / 1e6

plt.figure(figsize=(10, 10))
plt.scatter(data["time"], data["flux"], color="blue", s=4, label="rel_flux_T1")
plt.errorbar(data["time"], data["flux"], yerr=data["error"], color="blue", alpha=0.1)
plt.plot(data["time"], final_lc, color="blue")
plt.scatter(data["time"], residuals + 0.945, s=4, color="fuchsia", label=f"residuals, RMS={round(RMS, 4)}")

plt.ylim(1 - 14 * depth, 1 + 6 * depth)
plt.legend(loc="upper center", fontsize=15)

plt.xlabel("BJD_{TBD}")
plt.ylabel("Relative Flux")

fig_path = planet_dir / f"{updated_exoplanet_name}_lightcurve.png"
plt.savefig(fig_path)

# future work
# - make sure this work with other exoplanets (specifically one where there are missing archive inputs)
# - figure out how to detrend airmass (a mystery)
# - make prettier plots with final model, residuals 
# - figure out how to interpert RMSE of residuals 
# - Determine best limb darkening priors (0.3, 0.3 on AIJ?)
# - Determine which priors are best kept on or off for running MCMC/minimization
# - make process of assigning priors/pars smoother 
# - Ultimae goal - some sort of package?? where you can just grab it from github, install packages, and 
#       run to process a lightcurve, get priors
#       Also update with image processing?? if i ever ever learn that but this seems difficult-ish