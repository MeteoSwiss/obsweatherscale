from pathlib import Path
from typing import Optional

import xarray as xr
import zarr


def open_zarr_file(
    filename: Path,
    data_key: Optional[list[str]] = None,
    time_slice: slice = slice(None),
) -> xr.Dataset:
    data = xr.open_zarr(filename)
    data = data.isel(time=time_slice)
    return data if data_key is None else data[data_key]


def to_zarr_with_compressor(
    data: xr.Dataset,
    filename: Path,
    compressor: zarr.Blosc,
    chunk_size: Optional[dict] = None,
) -> None:
    if chunk_size is None:
        chunk_size = dict(data.sizes)
        chunk_size["time"] = "auto"

    # Chunk the data
    chunked_data = data.chunk(chunk_size)
    chunked_data = chunked_data.unify_chunks()

    # Save the chunked data to zarr
    chunked_data.to_zarr(
        filename,
        mode="w",
        encoding={var: {"compressor": compressor} for var in data.variables},
    )
