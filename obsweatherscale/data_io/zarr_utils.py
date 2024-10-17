from pathlib import Path

import xarray as xr

def open_zarr_file(
    filename: Path,
    data_key: list[str] | None = None,
    time_slice: slice = slice(None)
) -> xr.Dataset:
    data = xr.open_zarr(filename)
    data = data.isel(time=time_slice)
    return data if data_key is None else data[data_key]
