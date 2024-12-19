from typing import TypeVar
import xarray as xr

XRData = TypeVar("XRData", xr.Dataset, xr.DataArray)


def coords_to_station(dataset: XRData) -> XRData:
    return dataset.stack(station=('x', 'y'))

def station_to_coords(dataset: XRData) -> XRData:
    return dataset.unstack('station')
