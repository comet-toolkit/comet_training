import xarray as xr
import numpy as np
from punpy import MeasurementFunction, MCPropagation
from matheo.band_integration import band_integration
import pandas as pd
import obsarray
from obsarray.templater.dataset_util import DatasetUtil
import matplotlib.pyplot as plt
import os

ds_refl_L9 = xr.load_dataset("example_L9_20220606.nc")
ds_refl_L9_rcn = xr.load_dataset("example_L9_20220606_rcn.nc")

bands_L9=["B1","B2","B3","B4","B5","B6"]
wav_L9 = [442.98244284,482.58889933,561.33224557,654.60554515,864.5708545,1609.09056245]

refl_L9 = np.array([np.mean(ds_refl_L9[band].values) for band in bands_L9])
u_refl_L9 = np.array([np.std(ds_refl_L9[band].values) for band in bands_L9])
refl_L9_rcn = np.array([np.mean(ds_refl_L9_rcn[band].values) for band in bands_L9])
u_refl_L9_rcn = np.array([np.std(ds_refl_L9_rcn[band].values) for band in bands_L9])

class BandIntegrateL9(MeasurementFunction):
    # your measurement function
    def meas_function(self, reflectance, wavelength):
        """
        Function to perform S2A band integration on reflectance

        :param reflectance: reflectance spectrum
        :param wavelength: wavelengths
        """
        refl_band, band_centres = band_integration.spectral_band_int_sensor(
            d=reflectance,
            wl=wavelength,
            platform_name="Landsat-9",
            sensor_name="OLI",
            u_d=None,
        )
        return refl_band[:6]

    def get_argument_names(self):
        """
        Function that returns the argument names of the meas_func, as they appear in the digital effects table (used to find right variable in input data).

        :return: List of argument names
        """
        return ["reflectance", "wavelength"]

    def get_measurand_name_and_unit(self):
        """
        Function that returns the measurand name and unit of the meas_func. These will be used to store in the output dataset.

        :return: tuple(measurand name, measurand unit)
        """
        return "band_reflectance", ""

ds_HYP = xr.open_dataset("HYPERNETS_L_GHNA_L2A_REF_20220606T0900_20231226T1435_v2.0.nc")  # read digital effects table

bad_flags=["pt_ref_invalid", "half_of_scans_masked", "not_enough_dark_scans", "not_enough_rad_scans",
           "not_enough_irr_scans", "no_clear_sky_irradiance", "variable_irradiance",
           "half_of_uncertainties_too_big", "discontinuity_VNIR_SWIR", "single_irradiance_used"]
flagged = DatasetUtil.get_flags_mask_or(ds_HYP["quality_flag"], bad_flags)
id_series_valid = np.where(~flagged)[0]
ds_HYP = ds_HYP.isel(series=id_series_valid)

vza=0.7
vaa=38.1
vzadiff=(ds_HYP["viewing_zenith_angle"].values - vza)
vaadiff=(np.abs(ds_HYP["viewing_azimuth_angle"].values - vaa%360))
angledif_series = vzadiff** 2 + vaadiff ** 2
id_series = np.where(angledif_series**2 == np.min(angledif_series**2))[0][0]
ds_HYP = ds_HYP.isel(series=id_series)

prop = MCPropagation(50,parallel_cores=1)
band_int_L8 = BandIntegrateL9(prop, use_err_corr_dict=True)
ds_HYP_L8 = band_int_L8.propagate_ds_total(ds_HYP)

wav_HYP_full = ds_HYP["wavelength"].values
refl_HYP_full = ds_HYP["reflectance"].values
u_refl_HYP_full = refl_HYP_full*np.sqrt(ds_HYP["u_rel_systematic_reflectance"].values**2+ds_HYP["u_rel_random_reflectance"].values**2)/100

refl_HYP = ds_HYP_L8["band_reflectance"].values
u_refl_HYP = refl_HYP*ds_HYP_L8["u_tot_band_reflectance"].values/100


rcn_data = np.genfromtxt("GONA01_2022_157_v00.09.input", dtype="str", delimiter="\t", skip_header=4)
if np.all(rcn_data[:,-1]==''):
    rcn_data=rcn_data[:,:-1]
wav_rcn = rcn_data[12:223, 0].astype(float)
times_rcn = rcn_data[2, 1:]

refl_rcn = rcn_data[12:223, 3].astype(float)
u_refl_rcn = np.abs(rcn_data[229:440, 3].astype(float))

refl_rcn_band, band_centres, u_refl_rcn_band  = band_integration.spectral_band_int_sensor(
            d=refl_rcn,
            wl=wav_rcn,
            platform_name="Landsat-8",
            sensor_name="OLI",
            u_d=u_refl_rcn,
        )
refl_rcn_band, u_refl_rcn_band = refl_rcn_band[:6], u_refl_rcn_band[:6]

bias = ((refl_L9/refl_HYP) - 1) * 100
u_bias = (
    np.sqrt(
        (u_refl_L9 / refl_L9) ** 2
        + (u_refl_HYP / refl_HYP) ** 2
    )
    * 100
)

bias_rcn = ((refl_L9_rcn/refl_rcn_band) - 1) * 100
u_bias_rcn = (
    np.sqrt(
        (u_refl_L9_rcn / refl_L9_rcn) ** 2
        + (u_refl_rcn_band / refl_rcn_band) ** 2
    )
    * 100
)

def plot(path:str, sat:str, name:str, date:str, sat_wav:np.ndarray, sat_refl:np.ndarray, sat_unc:np.ndarray, hyp_wav:np.ndarray, hyp_refl:np.ndarray, hyp_unc:np.ndarray, rcn_wav:np.ndarray,
         rcn_refl:np.ndarray, rcn_unc:np.ndarray, bias:np.ndarray, bias_unc:np.ndarray, wavs_band:np.ndarray, reflectance_band:np.ndarray, reflectance_band_unc:np.ndarray, vza:float, bias_rcn:np.ndarray, bias_rcn_unc:np.ndarray):
    print("preparing %s plot" % name)

    plt.figure(figsize=(20, 12))
    plt.subplot(2, 1, 1)
    plt.errorbar(wavs_band, reflectance_band, yerr=reflectance_band_unc, fmt='o', ls='none', ms=10, color='m',
                 label='HYPERNETS for satellite bands')
    plt.errorbar(sat_wav, sat_refl, yerr=sat_unc, fmt='o', ls='none', ms=10, color='g',
                 label=sat)
    plt.fill_between(hyp_wav, hyp_refl - hyp_unc, hyp_refl + hyp_unc,
                     alpha=0.3, color="b")
    plt.errorbar(rcn_wav, rcn_refl, yerr=rcn_unc,
                 label='RadCalNet', color="orange")
    plt.plot(hyp_wav, hyp_refl, '-b', label='HYPERNETS full-resolution model')
    if sat == "Landsat-8" or sat == "Landsat-9":
        plt.title('Landsat-8/9 (vza=%.1f) vs HYPERNETS TOA Comparison at %s' % (
            vza, date),
                  fontsize=20)
    else:
        plt.title('%s (vza=%.1f) vs HYPERNETS Comparison at %s' % (
            sat, vza, date),
                  fontsize=20)
    plt.ylabel('Reflectance', fontsize=20)
    plt.xlim(380, 1700)
    plt.ylim(0.0, 0.6)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc=2, numpoints=1, scatterpoints=1, facecolor='white')

    plt.subplot(2, 1, 2)
    plt.errorbar(sat_wav, bias, yerr=bias_unc, fmt='o', mfc='blue', ls='none', ms=15, capsize=3,
                 label="HYPERNETS-%s bias" % sat)
    plt.errorbar(sat_wav, bias_rcn, yerr=bias_rcn_unc, fmt='o', mfc='orange', ls='none', ms=15, capsize=3, alpha=0.5,
                 label="RadCalNet-%s bias" % sat)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.ylabel('Relative Difference (%)', fontsize=20)
    plt.xlabel('Wavelength (nm)', fontsize=20)
    plt.xlim(380, 1700)
    plt.ylim(-10, 10)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend()
    # plt.legend(loc=2, numpoints=1, scatterpoints=1, facecolor='white')
    plt.savefig(os.path.join(path, name + "_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()


plot("./", "LANDSAT 9", "bias_L9_test", "2022-06-06", wav_L9, refl_L9, u_refl_L9, wav_HYP_full, refl_HYP_full, u_refl_HYP_full, wav_rcn,
     refl_rcn, u_refl_rcn, bias, u_bias, wav_L9, refl_HYP, u_refl_HYP, vza, bias_rcn, u_bias_rcn)



ds_refl_S2 = xr.load_dataset("example_S2_20220628.nc")
ds_refl_S2_rcn = xr.load_dataset("example_S2_20220628_rcn.nc")

bands_S2=["B01","B02","B03","B04","B05","B06","B07","B08","B09","B10","B11","B8A"]
wav_S2 = [442.69504835,  492.43657768,  559.84905824,  664.62175142,
        704.11493669,  740.49182383,  782.75291928,  832.79041366,
        945.05446558, 1373.46188735, 1613.65941706,  864.71079209]
vza_S2=5.6

refl_S2 = np.array([np.mean(ds_refl_S2[band].values) for band in bands_S2])
u_refl_S2 = np.array([np.std(ds_refl_S2[band].values) for band in bands_S2])
refl_S2_rcn = np.array([np.mean(ds_refl_S2_rcn[band].values) for band in bands_S2])
u_refl_S2_rcn = np.array([np.std(ds_refl_S2_rcn[band].values) for band in bands_S2])

wav_TOA_HYP_full, refl_TOA_HYP_full, u_refl_TOA_HYP_full = np.load("hypernets_TOA_example_20220608_full.npy")
wav_TOA_HYP_band, refl_TOA_HYP_band, u_refl_TOA_HYP_band = np.load("hypernets_TOA_example_20220608_band.npy")

rcn_data = np.genfromtxt("GONA01_2022_159_v04.09.output", dtype="str", delimiter="\t", skip_header=4)
if np.all(rcn_data[:,-1]==''):
    rcn_data=rcn_data[:,:-1]
wav_TOA_rcn = rcn_data[15:226, 0].astype(float)
times_rcn = rcn_data[2, 1:]

refl_TOA_rcn = rcn_data[15:226, 3].astype(float)
u_refl_TOA_rcn = np.abs(rcn_data[232:443, 3].astype(float))

refl_TOA_rcn_band, band_centres, u_refl_TOA_rcn_band = band_integration.spectral_band_int_sensor(
            d=refl_TOA_rcn,
            wl=wav_TOA_rcn,
            platform_name="Sentinel-2B",
            sensor_name="MSI",
            u_d=u_refl_TOA_rcn,
        )
refl_TOA_rcn_band, u_refl_TOA_rcn_band = np.delete(refl_TOA_rcn_band, 11),np.delete(u_refl_TOA_rcn_band,11)

bias_TOA = ((refl_S2/refl_TOA_HYP_band) - 1) * 100
u_bias_TOA = (
    np.sqrt(
        (u_refl_S2 / refl_S2) ** 2
        + (u_refl_TOA_HYP_band / refl_TOA_HYP_band) ** 2
    )
    * 100
)

bias_TOA_rcn = ((refl_S2_rcn/refl_TOA_rcn_band) - 1) * 100
u_bias_TOA_rcn = (
    np.sqrt(
        (u_refl_S2_rcn / refl_S2_rcn) ** 2
        + (u_refl_TOA_rcn_band / refl_TOA_rcn_band) ** 2
    )
    * 100
)

plot("./", "Sentinel-2", "bias_S2_test", "2022-06-08", wav_S2, refl_S2, u_refl_S2, wav_TOA_HYP_full, refl_TOA_HYP_full, u_refl_TOA_HYP_full, wav_TOA_rcn,
     refl_TOA_rcn, u_refl_TOA_rcn, bias_TOA, u_bias_TOA, wav_TOA_HYP_band, refl_TOA_HYP_band, u_refl_TOA_HYP_band, vza_S2, bias_TOA_rcn, u_bias_TOA_rcn)