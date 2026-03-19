# Import packages
import obsarray
import punpy
import xarray as xr
import numpy as np
import warnings
warnings.filterwarnings('ignore') # hide any confusing warnings!


# open xarray.Dataset from netCDF file
dataset_path = "avhrr_ds.nc"
avhrr_ds = xr.open_dataset(dataset_path)

# inspect dataset
avhrr_ds["bt"][0].plot()

# define u_noise values - set as 1%
u_noise_values = avhrr_ds["bt"].values * 0.01

# add an uncertainty component associated with noise error to the brightness temperature
avhrr_ds.unc["bt"]["u_bt_noise"] = (["band", "y", "x"], u_noise_values)

# create cross-channel error-correlation matrix
chl_err_corr_matrix = np.array([[1.0, 0.7],[0.7, 1.0]])
avhrr_ds["chl_err_corr_matrix"] = (("band", "band"), chl_err_corr_matrix)

# use this to define error-correlation parameterisation attribute
err_corr_def = [
    # fully systematic in the x and y dimension
    {
        "dim": ["y", "x"],
        "form": "systematic",
        "params": [],
        "units": []
    },
    # defined by err-corr matrix var in band dimension
    {
        "dim": ["band"],
        "form": "err_corr_matrix",
        "params": ["chl_err_corr_matrix"],  # defines link to err-corr matrix var
        "units": []
    }
]

# define u_cal values - set as 5%
u_cal_values = avhrr_ds["bt"].values * 0.05

# add an uncertainty component associated with calibration error to the brightness temperature
avhrr_ds.unc["bt"]["u_bt_cal"] = (["band", "y", "x"], u_cal_values, {"err_corr": err_corr_def})

class SplitWindowSST(punpy.MeasurementFunction):
    #
    # define primary method of class - the measurement function
    #
    def meas_function(self, bt: np.ndarray) -> np.ndarray:
        """
        Returns SST from input L1 BTs using split window method

        :param bt: brightness temperature datacube
        :returns: evaluated SST
        """

        # set parameter values
        a = 2
        b = 1

        # evaluate SST
        sst = a * bt[0,:,:] - b * bt[1,:,:]

        return sst

    #
    # define helper methods to configure class
    #
    def get_measurand_name_and_unit(self) -> tuple[str, str]:
        """
        For dataset evaluate by measurement function, returns a tuple of
        measurand variable name and units

        :returns: measurand name, measurand unit name
        """
        return "sst", "K"

    def get_argument_names(self) -> list[str]:
        """
        Returns orders list input dataset variables names associated with
        meas_function arguments

        :returns: input dataset variable names
        """
        return ["bt"]

# create punpy propagation object
prop = punpy.MCPropagation(100, parallel_cores=1,verbose=True)

# Instatiate measurement function object with prop
sst_ret = SplitWindowSST(
    prop=prop,
    ydims=["y", "x"],
    sizes_dict={"y": 100, "x": 100},
    use_err_corr_dict=True
)

# run uncertainty propagatoin
sst_ds = sst_ret.propagate_ds(avhrr_ds,include_corr=False)
print(sst_ds)