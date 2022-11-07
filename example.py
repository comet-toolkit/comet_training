import xarray as xr
import obsarray
from punpy import MeasurementFunction, MCPropagation

# read digital effects table
ds = xr.open_dataset("comet_training/digital_effects_table_gaslaw_example.nc")

# Define your measurement function inside a subclass of MeasurementFunction
class IdealGasLaw(MeasurementFunction):
    def meas_function(self, pres, temp, n):
        return (n *temp * 8.134)/pres

# Create Monte Carlo Propagation object, and create MeasurementFunction class
# object with required parameters such as names of input quantites in ds
prop = MCPropagation(10000)
gl = IdealGasLaw(prop, xvariables=["pressure", "temperature", "n_moles"],
                 yvariable="volume", yunit="m^3")

# propagate the uncertainties on the input quantites in ds to the measurand
# uncertainties in ds_y (propagate_ds returns random, systematic and structured)
ds_y = gl.propagate_ds(ds, store_unc_percent=True)
