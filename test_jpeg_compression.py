"""
Comprehensive parametrized test suite for JPEG-LS compression in Astropy FITS.

Covers all combinations of shape x dtype, with xfail markers for
unsupported combinations.
"""

import io

import numpy as np
import pytest

from astropy.io import fits

# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

SHAPES_2D = [(1, 1), (1, 5), (5, 1), (3, 3), (1024, 1024)]
SHAPES_3D = [(1, 1, 1), (1, 1, 5), (1, 5, 1), (1, 3, 3), (2, 1024, 1024)]
SHAPES_4D = [(1, 1, 1, 1), (1, 1, 1, 5), (1, 1, 5, 1), (1, 1, 3, 3), (2, 2, 1024, 1024)]
ALL_SHAPES = SHAPES_2D + SHAPES_3D + SHAPES_4D

INT_DTYPES = [np.int8, np.int16, np.int32, np.int64,
              np.uint8, np.uint16, np.uint32, np.uint64]
FLOAT_DTYPES = [np.float16, np.float32, np.float64]
ALL_DTYPES = INT_DTYPES + FLOAT_DTYPES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_data(shape, dtype):
    """Generate reproducible test data with Gaussian-like distribution."""
    rng = np.random.default_rng(42)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
    else:
        info = np.finfo(dtype)
    mid = (float(info.max) + float(info.min)) / 2.0
    std = (float(info.max) - float(info.min)) / 10.0
    data = rng.normal(mid, std, shape)
    return np.clip(data, info.min, info.max).astype(dtype)


def compress_decompress(data, compression_type, **kwargs):
    """Round-trip data through CompImageHDU; return (result_array, compressed_file_bytes)."""
    comp_hdu = fits.CompImageHDU(data, compression_type=compression_type, **kwargs)
    buf = io.BytesIO()
    fits.HDUList([fits.PrimaryHDU(), comp_hdu]).writeto(buf)
    nbytes = buf.tell()
    buf.seek(0)
    result = fits.open(buf)[1].data
    return result, nbytes


# ---------------------------------------------------------------------------
# xfail predicates
# ---------------------------------------------------------------------------

def _jpegls_xfail_reason(shape, dtype):
    """Return an xfail reason string if this (shape, dtype) combo is not
    exactly round-trippable by JPEG-LS, else None."""
    # FITS has no BITPIX for float16
    if dtype == np.float16:
        return "float16 has no FITS BITPIX representation"
    # Floats round-trip through lossy quantization (default quantize_level),
    # so they are not bit-exact unless quantize_level=0 is passed explicitly.
    if np.issubdtype(dtype, np.floating):
        return "floats round-trip lossily via quantization, not bit-exact by default"
    # JPEG-LS supports 8/16-bit directly and 32-bit via the split-plane
    # container; 64-bit integers have no path.
    if dtype in (np.int64, np.uint64):
        return "JPEG-LS has no 64-bit integer path"
    return None


# ---------------------------------------------------------------------------
# Core round-trip test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", ALL_DTYPES, ids=[np.dtype(d).name for d in ALL_DTYPES])
@pytest.mark.parametrize("shape", ALL_SHAPES, ids=[str(s) for s in ALL_SHAPES])
def test_roundtrip(shape, dtype):
    """Data compressed and decompressed round-trips to exact equality."""
    reason = _jpegls_xfail_reason(shape, dtype)
    if reason is not None:
        pytest.xfail(reason)

    data = make_data(shape, dtype)

    try:
        result, _ = compress_decompress(data, "JPEGLS")
    except Exception as exc:
        # Surface unexpected errors as test failures with context
        pytest.fail(f"JPEGLS {dtype} {shape} raised: {exc}")

    np.testing.assert_array_equal(
        data,
        result,
        err_msg=f"JPEGLS round-trip failed for dtype={dtype} shape={shape}",
    )


# ---------------------------------------------------------------------------
# Compression-ratio test (only large tiles where compression is meaningful)
# ---------------------------------------------------------------------------

_RATIO_DTYPES = [np.int16, np.uint16, np.int32, np.uint32]
_RATIO_SHAPE = (1024, 1024)

@pytest.mark.parametrize("dtype", _RATIO_DTYPES, ids=[np.dtype(d).name for d in _RATIO_DTYPES])
def test_better_ratio_than_rice(dtype):
    """JPEG-LS must compress a 1024x1024 array better than RICE_1."""
    data = make_data(_RATIO_SHAPE, dtype)
    _, jpeg_size = compress_decompress(data, "JPEGLS")
    _, rice_size = compress_decompress(data, "RICE_1")
    assert jpeg_size < rice_size, (
        f"JPEGLS ({dtype.__name__}) compressed size {jpeg_size} "
        f"is NOT smaller than RICE_1 size {rice_size}"
    )
