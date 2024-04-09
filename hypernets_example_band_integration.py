import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from matheo.band_integration import band_integration
from matheo.utils import function_def as fd
import obsarray

ds_HYP = xr.open_dataset("HYPERNETS_L_GHNA_L2A_REF_20231103T0901_20240124T2246_v2.0.nc")  # read digital effects table
reflectance_HYP=ds_HYP["reflectance"].values
u_reflectance_HYP=ds_HYP.unc["reflectance"].total_unc()
wavelength_HYP=ds_HYP["wavelength"].values

refl_S2, band_centres_S2 = band_integration.spectral_band_int_sensor(
    d=reflectance_HYP,
    wl=wavelength_HYP,
    platform_name="Sentinel-2A",
    sensor_name="MSI",
)

wav_sat = np.arange(400,1600,10)
width_sat = 10*np.ones_like(wav_sat)

refl_band = band_integration.pixel_int(
   d=reflectance_HYP,
   x=wavelength_HYP,
   x_pixel=wav_sat,
   width_pixel=width_sat,
   band_shape="triangle",
)

wav_SRF = np.arange(390,1610,0.1)
r_SRF = np.array([fd.f_triangle(wav_SRF, sat_wav_i, 10) for sat_wav_i in wav_sat])
refl_band2 = band_integration.band_int(reflectance_HYP, wavelength_HYP, r_SRF, wav_SRF)

plt.plot(wavelength_HYP,reflectance_HYP[:,30],label="HYPERNETS reflectance for index 30")
plt.fill_between(wavelength_HYP,reflectance_HYP[:,30]-u_reflectance_HYP[:,30],reflectance_HYP[:,30]+u_reflectance_HYP[:,30], alpha=0.3, label="HYPERNETS reflectance uncertainty")
plt.plot(wav_sat,refl_band[:,30],label="pixel_int with 10nm sampling and 10 nm width")
plt.plot(wav_sat,refl_band2[:,30],label="band_int with 10nm sampling and 10 nm width")
plt.plot(band_centres_S2,refl_S2[:,30],"s",label="spectral_band_int_sensor with S2A SRF")
plt.legend()
plt.ylim([0,0.6])
plt.show()