import xarray as xr
import numpy as np
from punpy import MeasurementFunction, MCPropagation
from matheo.band_integration import band_integration

class BandIntegrateS2A(MeasurementFunction):
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
            platform_name="Sentinel-2A",
            sensor_name="MSI",
            u_d=None,
        )
        return refl_band

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

ds_HYP_full = xr.open_dataset("HYPERNETS_L_GHNA_L2A_REF_20231103T0901_20240124T2246_v2.0.nc")  # read digital effects table
prop = MCPropagation(100,parallel_cores=1)
band_int_S2 = BandIntegrateS2A(prop, use_err_corr_dict=True)
ds_HYP_full_S2 = band_int_S2.propagate_ds(ds_HYP_full)
print(ds_HYP_full_S2)