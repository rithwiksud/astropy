"""
This file contains the code for Quantizing / Dequantizing floats.
"""

import numpy as np

from astropy.io.fits.hdu.base import BITPIX2DTYPE
from astropy.io.fits.hdu.compressed._compression import (
    quantize_double_c,
    quantize_float_c,
    unquantize_double_c,
    unquantize_float_c,
)

__all__ = ["Quantize"]


DITHER_METHODS = {
    "NONE": 0,
    "NO_DITHER": -1,
    "SUBTRACTIVE_DITHER_1": 1,
    "SUBTRACTIVE_DITHER_2": 2,
}


class QuantizationFailedException(Exception):
    pass


class Quantize:
    """
    Quantization of floating-point data following the FITS standard.
    """

    def __init__(
        self, *, row: int, dither_method: int, quantize_level: int, bitpix: int
    ):
        super().__init__()
        self.row = row
        # TODO: pass dither method as a string instead of int?
        self.quantize_level = quantize_level
        self.dither_method = dither_method
        self.bitpix = bitpix

    # NOTE: below we use decode_quantized and encode_quantized instead of
    # decode and encode as we need to break with the numcodec API and take/return
    # scale and zero in addition to quantized value. We should figure out how
    # to properly use the numcodec API for this use case.

    def decode_quantized(self, buf, scale, zero):
        """
        Unquantize data.

        Parameters
        ----------
        buf : bytes or array_like
            The buffer to unquantize.

        Returns
        -------
        np.ndarray
            The unquantized buffer.
        """
        qbytes = np.asarray(buf)
        qbytes = qbytes.astype(qbytes.dtype.newbyteorder("="))
        # TODO: figure out if we need to support null checking
        if self.dither_method == -1:
            # For NO_DITHER we should just use the scale and zero directly
            return qbytes * scale + zero
        if self.bitpix == -32:
            ubytes = unquantize_float_c(
                qbytes.tobytes(),
                self.row,
                qbytes.size,
                scale,
                zero,
                self.dither_method,
                0,
                0,
                0.0,
                qbytes.dtype.itemsize,
            )
        elif self.bitpix == -64:
            ubytes = unquantize_double_c(
                qbytes.tobytes(),
                self.row,
                qbytes.size,
                scale,
                zero,
                self.dither_method,
                0,
                0,
                0.0,
                qbytes.dtype.itemsize,
            )
        else:
            raise TypeError("bitpix should be one of -32 or -64")
        return np.frombuffer(ubytes, dtype=BITPIX2DTYPE[self.bitpix]).data

    def encode_quantized(self, buf):
        """
        Quantize data.

        Parameters
        ----------
        buf : bytes or array_like
            The buffer to quantize.

        Returns
        -------
        np.ndarray
            A buffer with quantized data.
        """
        uarray = np.asarray(buf)
        uarray = uarray.astype(uarray.dtype.newbyteorder("="))
        # TODO: figure out if we need to support null checking
        if uarray.dtype.itemsize == 4:
            qbytes, status, scale, zero = quantize_float_c(
                uarray.tobytes(),
                self.row,
                uarray.size,
                1,
                0,
                0,
                self.quantize_level,
                self.dither_method,
            )[:4]
        elif uarray.dtype.itemsize == 8:
            qbytes, status, scale, zero = quantize_double_c(
                uarray.tobytes(),
                self.row,
                uarray.size,
                1,
                0,
                0,
                self.quantize_level,
                self.dither_method,
            )[:4]
        if status == 0:
            raise QuantizationFailedException()
        else:
            return np.frombuffer(qbytes, dtype=np.int32), scale, zero


def quantize_integer_arr(arr: np.ndarray, max_err: int, mask: np.ndarray, nbits: int = 16) -> np.ndarray:
    """
    Only applied to quantizing integer data for near-lossless compression.
    Quantize array values within a maximum error bound where specified by a mask.
    
    Quantization reduces precision of values to improve compressibility while ensuring
    the difference between original and quantized values does not exceed max_err.
    Only values below (2^nbits - max_err) are quantized to prevent overflow.

    Parameters
    ----------
    arr : np.ndarray
        Array to quantize. Can be 2D, 3D, or 4D. The two largest dimensions
        are assumed to be spatial dimensions that the mask applies to.
    max_err : int
        Maximum allowed difference between original and quantized values.
        Must be positive.
    mask : np.ndarray, optional
        2D boolean array indicating which spatial pixels to quantize.
        True values indicate pixels that should be quantized. If None, assumes
        all pixels are to be quantized. Default is None.
    nbits : int, optional
        Bit depth of the data. Used to determine maximum safe value for
        quantization. Default is 16.

    Returns
    -------
    np.ndarray
        Quantized array with the same shape as input arr. Values where
        mask is False or where original values are too large remain unchanged.

    Raises
    ------
    AssertionError
        If mask is provided but is not 2D.
    ValueError
        If arr has unsupported dimensionality or mismatched spatial dimensions with the mask.
    """
    # Input validation
    assert max_err > 0, f"max_err must be positive, got {max_err}"
    if arr.ndim not in [2, 3, 4]:
        raise ValueError(f"Array must be 2D, 3D or 4D, got shape {arr.shape}")
    
    # Identify spatial dimensions (two largest dimensions)
    spatial_dims = np.argsort(arr.shape)[-2:]  # Indices of the two largest dimensions
    spatial_shape = (arr.shape[spatial_dims[0]], arr.shape[spatial_dims[1]])
    
    # Handle the case where mask is None
    if mask is None:
        mask = np.ones(spatial_shape, dtype=bool)  # Default mask with all True values
    else:
        # Verify that the mask matches spatial dimensions
        assert mask.ndim == 2, f"Mask must be 2D, got shape {mask.shape}"
        if mask.shape != spatial_shape:
            raise ValueError(f"Mask shape {mask.shape} must match spatial dimensions {spatial_shape} of the array.")
    
    # Expand the mask to match the shape of the array using slicing
    expanded_mask = mask[(slice(None),) * spatial_dims[0] + np.index_exp[:] + (slice(None),) * (arr.ndim - spatial_dims[1] - 1)]
    
    # Prevent modifying values too large for safe quantization
    max_safe_value = 2**nbits - 1 - max_err
    not_too_big = arr <= max_safe_value  # Mask of safely quantizable values
    
    # Combine mask conditions
    combined_mask = expanded_mask & not_too_big
    
    # Perform quantization
    quantized_data = np.copy(arr)
    quant_size = 2 * max_err + 1  # Distance between quantization levels
    quantized_data[combined_mask] = (arr[combined_mask] // quant_size) * quant_size + max_err

    return quantized_data
