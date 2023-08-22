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
import warnings
warnings.filterwarnings("ignore")

# formatting data ==============================================================================
    # loads in observation dataframe 
#filename = input("Enter exoplanet filename (csv): ")
filename = "HAT-P-62-b_measurements.xls - HAT-P-62-b_measurements.xls.csv"

# converts filename to exoplanet archive safe filename 
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
if len(pars_df) == 0: print("Planet not in exoplanet archive, initial parameters may be incorrect.")

# list of key values 
all_keys = [key for key in pars_df.keys() if key[0] == "p"]

# dictionary of keys -> terms 
keys_to_names = {'pl_orbper':"Period", 'pl_orbeccen': "Eccentricity (e)", 
                 'pl_orbincl':"Inclination (i)", 'pl_tranmid': "Transit Center Time (T0)", 'pl_ratdor':"Scaled Semimajor Axis (a/R*)", 
                 'pl_ratror' : "Scaled Planet Radius (Rp/R*)",  'pl_orblper':"Argument of Periastum (w)"}

# returns keys where data_cols["pl_name"] has a nan value 
nan_keys = [key for key in pars_df.index if np.isnan(pars_df[key])]
non_nan_keys = [key for key in all_keys if key not in nan_keys]

# prints which pars are referenced from where: 
nan_keys_noerr = [key for key in nan_keys if key[-1]!="1"]
non_nan_keys_noerr = [key for key in non_nan_keys if key[-1]!="1"]

nan_names = [keys_to_names[key] for key in nan_keys_noerr]
non_nan_names = [keys_to_names[key] for key in non_nan_keys_noerr]

print("Priors referenced directly from Exoplanet Archive:")
print("")
for item in non_nan_names: print(item)
print("")
print("Priors assumed from Exoplanet Archive: ")
print("")
for item in nan_names: print(item)
print("")

# dictionary of prior guesses IF exoplanet table value nan
nan_prior_dict = {key : exoplanet_archive[key].median() for key in all_keys[1:]}    
if "pl_tranmid" in nan_keys : nan_prior_dict["pl_tranmid"] = data["time"][np.argmin(data["flux"])] # initial guess for transit center time at expected center time 

# assigns values to all keys from either exoplanet archive or priors dictionary 
prior_dict_1 = {key: pars_df[key] for key in non_nan_keys}
prior_dict = {key: nan_prior_dict[key] for key in nan_keys}
prior_dict.update(prior_dict_1)

# gets priors from dictionary 
priors_mod = get_priors(prior_dict, scale=10)

# sets initial parameters as values from exoplanet archive
pars = priors_mod[:,0] 

# runs nelder-meade minimization on model, data to get set of initial parameters
minimize_results = minimize(minimize_lnp, pars, args=(priors_mod, data), method="Nelder-Mead")
minimize_pars = minimize_results.x

final_lc = make_model(minimize_pars, data) # includes airmass 
depth_absolute = (final_lc.max() - final_lc.min())
# if depth is 0, means that lightcurve is flat, and transit center time is incorrect, check after minimization
if depth_absolute == 0:
    # rerun mcmc with different transit center time
    print("Rerunning using manual transit center time...")
    prior_dict["pl_tranmid"] = data["time"][np.argmin(data["flux"])] # sets transit center time equal to time location of minimum center time? 
    # this overrides the value from the table (may revisit)

    # gets priors from dictionary 
    priors_mod = get_priors(prior_dict, scale=10)

    # sets initial parameters as values from exoplanet archive
    pars = priors_mod[:,0] 

    # runs nelder-meade minimization on model, data to get set of initial parameters
    minimize_results = minimize(minimize_lnp, pars, args=(priors_mod, data), method="Nelder-Mead")
    minimize_pars = minimize_results.x
    final_lc = make_model(minimize_pars, data) # includes airmass 
    depth_absolute = (final_lc.max() - final_lc.min())

    if depth_absolute == 0:
        print("No valid guess for transit center time, aborting attempt.")
        sys.exit()

# runs mcmc
print(f"Running MCMC for {exoplanet_name}...")
flat_sample = run_mcmc(minimize_pars, priors_mod, data)
    
# Cornerplot error dealing-with ========================================================================================================================
    # plots cornerplot 
labels = ["T0", "log_period", "RpRs", "log_ars", "cosi",
            "esinw", "ecosw", "u1", "u2", "airmass_slope"]

    # extracts final parameters, saves to text file 
mcmc_pars, mcmc_stds = flatsample_pars(flat_sample)
# calculates absolute depth 

# chatGPT code to catch warning 
import logging 
# Configure the logging system to capture the specific warning message
logging.basicConfig(level=logging.INFO)

# Define a custom handler to capture the warning message
class MyWarningHandler(logging.Handler):
    def __init__(self, alternative_behavior_executed, rerun_marker):
        super().__init__()
        self.alternative_behavior_executed = alternative_behavior_executed
        self.rerun_marker = rerun_marker

    def emit(self, record):
        if not self.alternative_behavior_executed and "Too few points to create valid contours" in record.getMessage():
            self.alternative_behavior_executed = True
            self.rerun_marker[0] = True
            
def run_corner_plot(alternative_behavior_executed, rerun_marker):
    warning_handler = MyWarningHandler(alternative_behavior_executed, rerun_marker)
    logging.getLogger().addHandler(warning_handler)

    fig = corner.corner(flat_sample, labels=labels, show_titles=True)
    plt.tight_layout()
    fig_path = planet_dir /"cornerplot.png"
    plt.savefig(fig_path)

    logging.getLogger().removeHandler(warning_handler)
    
alternative_behavior_executed = False
rerun_marker = [False]

# Run the corner plot
run_corner_plot(alternative_behavior_executed, rerun_marker)

# If the rerun marker is set, rerun the specific code
if rerun_marker[0]:
    # Define a custom handler to capture the warning message for the second plot generation
    class MyWarningHandlerSecondPlot(logging.Handler):
        def emit(self, record):
            global alternative_behavior_executed
            if not alternative_behavior_executed and "Too few points to create valid contours" in record.getMessage():
                # Code to execute when the specific warning occurs
                print(f"Second cornerplot did not converge correctly, verify visually at tteam_mcmc/{updated_exoplanet_name}/cornerplot.png")
                
                keep_going = str(input("Is the cornerplot viable? y/n: "))
                if keep_going == "y":
                    print("Continuing...") 
                else: 
                    print("Aborting run...")
                    sys.exit()
                alternative_behavior_executed = True  # Set the flag to True

    # Create the custom warning handler for the second plot generation
    warning_handler_second_plot = MyWarningHandlerSecondPlot()

    # Add the handler to the root logger
    logging.getLogger().addHandler(warning_handler_second_plot)

    # Run the code that generates the second corner plot
    print("Too few points for valid contours. Rerunning MCMC with higher burnin, production run.")
    flat_sample = run_mcmc(minimize_pars, priors_mod, data)
    fig = corner.corner(flat_sample, labels=labels, show_titles=True)
    plt.tight_layout()
    fig_path = planet_dir /"cornerplot.png"
    plt.savefig(fig_path)
    print("Using current flatsample as bestfit sample...")
    print("")

    # Clean up by removing the custom warning handler for the second plot generation
    logging.getLogger().removeHandler(warning_handler_second_plot)
    alternative_behavior_executed = False

# phew that was not fun ========================================================================================================================
# continuing 

# calculates absolute depth 
final_lc = make_model(mcmc_pars, data) # includes airmass 
detrended_lc = lc(mcmc_pars, data) # just lc 
airmass_detrended_data = data['flux'] / (1 + mcmc_pars[9] * (data["time"] - np.median(data["time"]))) # removes airmass trend from data 
residuals = final_lc - data["flux"]  # calculates residuals
RMS = np.sqrt(((residuals) ** 2).mean())
print(f"RMS of residuals: {RMS}")

    # calculates depth as difference between baseline and transit center (what do i do w this)
baseline = detrended_lc.max()
center = detrended_lc.min()
depth_absolute = (baseline - center)/baseline * 1000
depth_abs_err = np.std(residuals) * 1000    

    # saves final parameters to a text file
labels.append("depth (ppt)") # use this now after cornerplot generation 
output_file_path = f'{planet_dir}/final_parameters.txt'
with open(output_file_path, 'w') as file:
    for i, label in enumerate(labels): 
        file.write(f"{label}: {mcmc_pars[i]} +/- {mcmc_stds[i]} \n")
    file.write(f"absolute depth (baseline-center)/baseline: {depth_absolute} +/- {depth_abs_err}")
    file.write(f"RMS of residuals: {RMS}")
    
    # plots final lightcurve w all the other stuff on the plot AiJ uses <3 
depth = mcmc_pars[10] / 1000 # gatheres depth value for plot spacing 
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
print(f"Plots and final parameters saved to direcory 'tteam_mcmc/{updated_exoplanet_name}'")

# future work
# Rerun with a bunch of datasets to pick out possible errors 
# figure out why transit depth is sorta off + which transit depth calculation is used by astro_swarthmore so I can better compare values/errors 
# - make prettier plots with final model, residuals , other properties 
# - figure out how to interpert RMSE of residuals 
# - Ultimae goal - some sort of package?? where you can just grab it from github, install packages, and run to process a lightcurve, get priors
#       Figure out how to give inputs in command line when run package via python lightcurve_fit.py:
#           - nburn, nprod
#       Figure out more convenient pipeline for getting data from drive -> ask someone? 
# - Use this to determine minimum transit depth we can observe -> go from depth (ppm) to milimags? 