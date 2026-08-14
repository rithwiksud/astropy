"""
This module contains the FITS compression algorithms in numcodecs style Codecs.
"""

from gzip import compress as gzip_compress
from gzip import decompress as gzip_decompress
try:
    from imagecodecs import jpegls_decode, jpegls_encode, jpegxl_decode, jpegxl_encode

    HAS_IMAGECODECS = True
except ImportError:
    HAS_IMAGECODECS = False

import numpy as np

from astropy.io.fits.hdu.compressed._compression import (
    compress_hcompress_1_c,
    compress_plio_1_c,
    compress_rice_1_c,
    decompress_hcompress_1_c,
    decompress_plio_1_c,
    decompress_rice_1_c,
)

from .settings import DEFAULT_NEAR_LOSSLESS_MAXERR, DEFAULT_JPEGXL_EFFORT

from ._quantization import quantize_integer_arr

# If numcodecs is installed, we use Codec as a base class for the codecs below
# so that they can optionally be used as codecs in any package relying on
# numcodecs - however this is optional and if numcodecs is not installed we use
# our own base class. This does not affect any compressed data functionality
# in astropy.io.fits.
try:
    from numcodecs.abc import Codec
except ImportError:

    class Codec:
        codec_id = None


__all__ = [
    "PLIO1",
    "Gzip1",
    "Gzip2",
    "HCompress1",
    "NoCompress",
    "Rice1",
    "JPEGLS",
    "JPEGXL",
]


def _as_big_endian_array(data):
    return data.astype(np.asarray(data).dtype.newbyteorder(">"), copy=False)


def _as_native_endian_array(data):
    if data.dtype.isnative:
        return data
    else:
        return data.astype(np.asarray(data).dtype.newbyteorder("="), copy=False)


class NoCompress(Codec):
    """
    A dummy compression/decompression algorithm that stores the data as-is.

    While the data is not compressed/decompressed, it is converted to big
    endian during encoding as this is what is expected in FITS files.
    """

    codec_id = "FITS_NOCOMPRESS"

    def decode(self, buf):
        """
        Decompress buffer using the NOCOMPRESS algorithm.

        Parameters
        ----------
        buf : bytes or array_like
            The buffer to decompress.

        Returns
        -------
        buf : np.ndarray
            The decompressed buffer.
        """
        return np.frombuffer(buf, dtype=np.uint8)

    def encode(self, buf):
        """
        Compress the data in the buffer using the NOCOMPRESS algorithm.

        Parameters
        ----------
        buf : bytes or array_like
            The buffer to compress.

        Returns
        -------
        bytes
            The compressed bytes.
        """
        return _as_big_endian_array(buf).tobytes()


class Gzip1(Codec):
    """
    The FITS GZIP 1 compression and decompression algorithm.

    The Gzip algorithm is used in the free GNU software compression utility of
    the same name. It was created by J. L. Gailly and M. Adler, based on the
    DEFLATE algorithm (Deutsch 1996), which is a combination of LZ77 (Ziv &
    Lempel 1977) and Huffman coding.
    """

    codec_id = "FITS_GZIP1"

    def decode(self, buf):
        """
        Decompress buffer using the GZIP_1 algorithm.

        Parameters
        ----------
        buf : bytes or array_like
            The buffer to decompress.

        Returns
        -------
        buf : np.ndarray
            The decompressed buffer.
        """
        # In principle we should be able to not have .tobytes() here and avoid
        # the copy but this does not work correctly in Python 3.11.
        cbytes = np.frombuffer(buf, dtype=np.uint8).tobytes()
        dbytes = gzip_decompress(cbytes)
        return np.frombuffer(dbytes, dtype=np.uint8)

    def encode(self, buf):
        """
        Compress the data in the buffer using the GZIP_1 algorithm.

        Parameters
        ----------
        buf _like
            The buffer to compress.

        Returns
        -------
        bytes
            The compressed bytes.
        """
        # Data bytes should be stored as big endian in files
        # In principle we should be able to not have .tobytes() here and avoid
        # the copy but this does not work correctly in Python 3.11.
        dbytes = _as_big_endian_array(buf).tobytes()
        return gzip_compress(dbytes)


class Gzip2(Codec):
    """
    The FITS GZIP2 compression and decompression algorithm.

    The gzip2 algorithm is a variation on 'GZIP 1'. In this case the buffer in
    the array of data values are shuffled so that they are arranged in order of
    decreasing significance before being compressed.

    For example, a five-element contiguous array of two-byte (16-bit) integer
    values, with an original big-endian byte order of:

    .. math::
        A1 A2 B1 B2 C1 C2 D1 D2 E1 E2

    will have the following byte order after shuffling:

    .. math::
        A1 B1 C1 D1 E1 A2 B2 C2 D2 E2,

    where A1, B1, C1, D1, and E1 are the most-significant buffer from
    each of the integer values.

    Byte shuffling shall only be performed for integer or floating-point
    numeric data types; logical, bit, and character types must not be shuffled.

    Parameters
    ----------
    itemsize
        The number of buffer per value (e.g. 2 for a 16-bit integer)

    """

    codec_id = "FITS_GZIP2"

    def __init__(self, *, itemsize: int):
        super().__init__()
        self.itemsize = itemsize

    def decode(self, buf):
        """
        Decompress buffer using the GZIP_2 algorithm.

        Parameters
        ----------
        buf : bytes or array_like
            The buffer to decompress.

        Returns
        -------
        buf : np.ndarray
            The decompressed buffer.
        """
        cbytes = np.frombuffer(buf, dtype=np.uint8).tobytes()
        # Start off by unshuffling buffer
        unshuffled_buffer = gzip_decompress(cbytes)
        array = np.frombuffer(unshuffled_buffer, dtype=np.uint8)
        return array.reshape((self.itemsize, -1)).T.ravel()

    def encode(self, buf):
        """
        Compress the data in the buffer using the GZIP_2 algorithm.

        Parameters
        ----------
        buf : bytes or array_like
            The buffer to compress.

        Returns
        -------
        bytes
            The compressed bytes.
        """
        # Data bytes should be stored as big endian in files
        array = _as_big_endian_array(buf).ravel()
        # Shuffle the buffer
        itemsize = array.dtype.itemsize
        array = array.view(np.uint8)
        shuffled_buffer = array.reshape((-1, itemsize)).T.ravel().tobytes()
        return gzip_compress(shuffled_buffer)


class Rice1(Codec):
    """
    The FITS RICE1 compression and decompression algorithm.

    The Rice algorithm [1]_ is simple and very fast It requires only enough
    memory to hold a single block of 16 or 32 pixels at a time. It codes the
    pixels in small blocks and so is able to adapt very quickly to changes in
    the input image statistics (e.g., Rice has no problem handling cosmic rays,
    bright stars, saturated pixels, etc.).

    Parameters
    ----------
    blocksize
        The blocksize to use, each tile is coded into blocks a number of pixels
        wide. The default value in FITS headers is 32 pixels per block.

    bytepix
        The number of 8-bit buffer in each original integer pixel value.

    References
    ----------
    .. [1] Rice, R. F., Yeh, P.-S., and Miller, W. H. 1993, in Proc. of the 9th
           AIAA Computing in Aerospace Conf., AIAA-93-4541-CP, American Institute of
           Aeronautics and Astronautics [https://doi.org/10.2514/6.1993-4541]
    """

    codec_id = "FITS_RICE1"

    def __init__(self, *, blocksize: int, bytepix: int, tilesize: int):
        self.blocksize = blocksize
        self.bytepix = bytepix
        self.tilesize = tilesize

    def decode(self, buf):
        """
        Decompress buffer using the RICE_1 algorithm.

        Parameters
        ----------
        buf : bytes or array_like
            The buffer to decompress.

        Returns
        -------
        buf : np.ndarray
            The decompressed buffer.
        """
        cbytes = np.frombuffer(_as_native_endian_array(buf), dtype=np.uint8).tobytes()
        dbytes = decompress_rice_1_c(
            cbytes, self.blocksize, self.bytepix, self.tilesize
        )
        return np.frombuffer(dbytes, dtype=f"i{self.bytepix}")

    def encode(self, buf):
        """
        Compress the data in the buffer using the RICE_1 algorithm.

        Parameters
        ----------
        buf : bytes or array_like
            The buffer to compress.

        Returns
        -------
        bytes
            The compressed bytes.
        """
        # We convert the data to native endian because it is passed to the
        # C compression code which will interpret it as being native endian.
        dbytes = (
            _as_native_endian_array(buf)
            .astype(f"i{self.bytepix}", copy=False)
            .tobytes()
        )
        return compress_rice_1_c(dbytes, self.blocksize, self.bytepix)


class PLIO1(Codec):
    """
    The FITS PLIO1 compression and decompression algorithm.

    The IRAF PLIO (pixel list) algorithm was developed to store integer-valued
    image masks in a compressed form. Such masks often have large regions of
    constant value hence are highly compressible. The compression algorithm
    used is based on run-length encoding, with the ability to dynamically
    follow level changes in the image, allowing a 16-bit encoding to be used
    regardless of the image depth.
    """

    codec_id = "FITS_PLIO1"

    def __init__(self, *, tilesize: int):
        self.tilesize = tilesize

    def decode(self, buf):
        """
        Decompress buffer using the PLIO_1 algorithm.

        Parameters
        ----------
        buf : bytes or array_like
            The buffer to decompress.

        Returns
        -------
        buf : np.ndarray
            The decompressed buffer.
        """
        cbytes = np.frombuffer(_as_native_endian_array(buf), dtype=np.uint8).tobytes()
        dbytes = decompress_plio_1_c(cbytes, self.tilesize)
        return np.frombuffer(dbytes, dtype="i4")

    def encode(self, buf):
        """
        Compress the data in the buffer using the PLIO_1 algorithm.

        Parameters
        ----------
        buf : bytes or array_like
            The buffer to compress.

        Returns
        -------
        bytes
            The compressed bytes.
        """
        # We convert the data to native endian because it is passed to the
        # C compression code which will interpret it as being native endian.
        dbytes = _as_native_endian_array(buf).astype("i4", copy=False).tobytes()
        return compress_plio_1_c(dbytes, self.tilesize)


class HCompress1(Codec):
    """
    The FITS HCompress compression and decompression algorithm.

    Hcompress is an the image compression package written by Richard L. White
    for use at the Space Telescope Science Institute. Hcompress was used to
    compress the STScI Digitized Sky Survey and has also been used to compress
    the preview images in the Hubble Data Archive.

    The technique gives very good compression for astronomical images and is
    relatively fast. The calculations are carried out using integer arithmetic
    and are entirely reversible. Consequently, the program can be used for
    either lossy or lossless compression, with no special approach needed for
    the lossless case.

    Parameters
    ----------
    scale
        The integer scale parameter determines the amount of compression. Scale
        = 0 or 1 leads to lossless compression, i.e. the decompressed image has
        exactly the same pixel values as the original image. If the scale
        factor is greater than 1 then the compression is lossy: the
        decompressed image will not be exactly the same as the original

    smooth
        At high compressions factors the decompressed image begins to appear
        blocky because of the way information is discarded. This blockiness
        ness is greatly reduced, producing more pleasing images, if the image
        is smoothed slightly during decompression.

    References
    ----------
    .. [1] White, R. L. 1992, in Proceedings of the NASA Space and Earth Science
           Data Compression Workshop, ed. J. C. Tilton, Snowbird, UT;
           https://archive.org/details/nasa_techdoc_19930016742
    """

    codec_id = "FITS_HCOMPRESS1"

    def __init__(self, *, scale: int, smooth: bool, bytepix: int, nx: int, ny: int):
        self.scale = scale
        self.smooth = smooth
        self.bytepix = bytepix
        # NOTE: we should probably make this less confusing, but nx is shape[0] and ny is shape[1]
        self.nx = nx
        self.ny = ny

    def decode(self, buf):
        """
        Decompress buffer using the HCOMPRESS_1 algorithm.

        Parameters
        ----------
        buf : bytes or array_like
            The buffer to decompress.

        Returns
        -------
        buf : np.ndarray
            The decompressed buffer.
        """
        cbytes = np.frombuffer(_as_native_endian_array(buf), dtype=np.uint8).tobytes()
        dbytes = decompress_hcompress_1_c(
            cbytes, self.nx, self.ny, self.scale, self.smooth, self.bytepix
        )
        # fits_hdecompress* always returns 4 byte integers irrespective of bytepix
        return np.frombuffer(dbytes, dtype="i4")

    def encode(self, buf):
        """
        Compress the data in the buffer using the HCOMPRESS_1 algorithm.

        Parameters
        ----------
        buf : bytes or array_like
            The buffer to compress.

        Returns
        -------
        bytes
            The compressed bytes.
        """
        # We convert the data to native endian because it is passed to the
        # C compression code which will interpret it as being native endian.
        dbytes = (
            _as_native_endian_array(buf)
            .astype(f"i{self.bytepix}", copy=False)
            .tobytes()
        )
        return compress_hcompress_1_c(
            dbytes, self.nx, self.ny, self.scale, self.bytepix
        )

    
class JPEGLS(Codec):
    """
    The JPEG-LS (Lossless and Near-Lossless JPEG) compression and decompression algorithm.
    JPEG-LS is an ISO/IEC standard (14495-1) for lossless and near-lossless compression 
    of continuous-tone images. It was developed to provide a low-complexity lossless and
    near-lossless image compression standard that could outperform other standards like 
    JPEG for specific types of images.

    The algorithm uses a very efficient adaptive prediction, context modeling, and 
    Golomb coding. It is particularly effective for medical, scientific, and natural 
    images with minimal noise. JPEG-LS achieves excellent compression ratios while 
    maintaining very low computational complexity.
    
    Can only compress int data.

    Parameters
    ----------
    max_err
        The near-lossless maximum error parameter. When set to 0, the compression
        is fully lossless, meaning the decompressed image will be bit-for-bit
        identical to the original. Values greater than 0 enable near-lossless
        compression, where each reconstructed sample differs from the original by
        no more than the specified error value. Default value is 0.

    References
    ----------
        [1] Weinberger, M. J., Seroussi, G., & Sapiro, G. (2000). The LOCO-I lossless 
           image compression algorithm: Principles and standardization into JPEG-LS. 
           IEEE Transactions on Image Processing, 9(8), 1309-1324.
    """

    codec_id = "JPEGLS"

    def __init__(
        self,
        *,
        max_err: int = DEFAULT_NEAR_LOSSLESS_MAXERR,
        bitpix: int = None,
        zblank: int = None,
    ):
        if not HAS_IMAGECODECS:
            raise ImportError(
                "The 'imagecodecs' package is required for JPEG-LS compression. "
                "Install it with: pip install imagecodecs"
            )
        self.max_err = max_err
        self.bitpix = bitpix
        self.zblank = zblank

    @staticmethod
    def _stream_near(stream):
        """
        Return the NEAR parameter recorded in a JPEG-LS codestream.

        NEAR is stored in the SOS marker segment, so a decoder needs no
        external metadata to tell lossless from near-lossless streams.
        """
        i = 2  # skip the SOI marker
        while i + 4 <= len(stream):
            if stream[i] != 0xFF:
                break
            if stream[i + 1] == 0xDA:  # SOS
                ncomponents = stream[i + 4]
                return stream[i + 5 + 2 * ncomponents]
            i += 2 + ((stream[i + 2] << 8) | stream[i + 3])
        raise ValueError("Invalid JPEG-LS codestream: no SOS marker found")

    @staticmethod
    def _encode_stream(buf, max_err):
        """Encode one uint8/uint16 plane as a JPEG-LS codestream."""
        # CFITSIO passes the tile to CharLS as a 2D frame when the tile is
        # 2D, and as a single row otherwise; mirror that so the codestreams
        # are interchangeable.
        buf = np.squeeze(buf)
        if buf.ndim != 2:
            buf = buf.reshape(1, -1)
        # JPEG-LS can *expand* incompressible data (by up to 6.25% plus
        # header overhead), and imagecodecs' default output buffer is too
        # small for that; preallocate the guaranteed worst case instead
        # (same bound as CFITSIO's imcomp_jpegls_max_encoded_size).
        out = bytearray(buf.nbytes + buf.nbytes // 16 + 1024)
        return bytes(jpegls_encode(buf, level=max_err, out=out))

    def decode(self, buf):
        """
        Decompress buffer using the JPEG-LS algorithm.

        Parameters
        ----------
        buf : bytes or array_like
            The buffer to decompress.

        Returns
        -------
        buf : np.ndarray
            The decompressed buffer.
        """
        cbytes = np.frombuffer(buf, dtype=np.uint8).tobytes()

        if self.bitpix not in (32, -32, -64):
            # uint8/uint16 samples decode directly; the +32768 offset for
            # 16-bit data is undone by the caller (_finalize_array).
            return jpegls_decode(cbytes)

        # 32-bit split container (matches CFITSIO): an 8-byte header -- the
        # 4-byte big-endian length of the high-plane stream, then the 4-byte
        # big-endian tile baseline -- followed by the high- and low-plane
        # JPEG-LS streams.
        if len(cbytes) < 8:
            raise ValueError("Invalid JPEG-LS stream for 32-bit tile")
        upper_len = int.from_bytes(cbytes[0:4], "big")
        baseline = int.from_bytes(cbytes[4:8], "big")
        if upper_len > len(cbytes) - 8:
            raise ValueError(
                "Corrupted JPEG-LS 32-bit tile: upper length exceeds data"
            )

        lower_stream = cbytes[8 + upper_len :]
        split = jpegls_decode(lower_stream).astype(np.uint32).ravel()
        if upper_len > 0:
            upper = jpegls_decode(cbytes[8 : 8 + upper_len]).astype(np.uint32)
            split |= upper.ravel() << 16
        # upper_len == 0 means the encoder found an identically-zero upper
        # plane (tile range fit in 16 bits after the rebase) and stored no
        # stream for it.

        # Near-lossless guard: NEAR error in the low plane can push
        # (split + baseline) past 2^32-1 for values near the top of the
        # range; saturate instead of wrapping, which stays within the NEAR
        # bound.  Lossless tiles skip the guard: their modulo-2^32
        # arithmetic is exact and wrapped null-marker pixels depend on it.
        if self._stream_near(lower_stream) > 0:
            split = np.minimum(split, np.uint32(0xFFFFFFFF) - np.uint32(baseline))

        uval = split + np.uint32(baseline)  # wraps modulo 2^32, as intended
        return (uval.astype(np.int64) - 0x80000000).astype(np.int32)

    def encode(self, buf):
        """
        Compress the data in the buffer using the JPEG-LS algorithm.

        Parameters
        ----------
        buf : bytes or array_like
            The buffer to compress.

        Returns
        -------
        bytes
            The compressed bytes.
        """
        buf = _as_native_endian_array(buf)

        # Null pixels must survive compression exactly: near-lossless coding
        # may perturb any value by up to NEAR, turning nulls into
        # valid-looking values and valid pixels within NEAR of the marker
        # into false nulls.  Any tile containing the marker is therefore
        # encoded losslessly (NEAR is per-codestream, so decoders handle a
        # mix of lossless and near-lossless tiles automatically).
        max_err = self.max_err
        if max_err > 0 and self.zblank is not None and np.any(buf == self.zblank):
            max_err = 0

        if buf.dtype == np.uint8:
            return self._encode_stream(buf, max_err)
        elif buf.dtype == np.int16:
            # Arithmetic conversion to unsigned: same method as CFITSIO (+32768)
            return self._encode_stream(
                (buf.astype(np.int32) + 32768).astype(np.uint16), max_err
            )
        elif buf.dtype == np.int32:
            return self._encode_int32(buf, max_err)
        else:
            raise ValueError(
                "JPEG-LS only supports 8, 16, or split 32-bit integer tiles, "
                f"got {buf.dtype}"
            )

    def _encode_int32(self, buf, max_err):
        """
        Encode an int32 tile as the CFITSIO JPEG-LS 32-bit split container.

        The 2^31-offset value is rebased to the tile's own minimum (so the
        16-bit plane split tracks the tile's local range rather than fixed
        65536 boundaries), then split into high and low 16-bit planes, each
        an independent JPEG-LS stream.  Layout: 4-byte big-endian high-plane
        length, 4-byte big-endian baseline, high stream, low stream.
        """
        uval = (buf.astype(np.int64) + 0x80000000).astype(np.uint32)

        # Null-marker pixels are excluded from the baseline: the marker is a
        # huge negative reserved value, and letting it set the baseline would
        # forfeit the rebase.  Excluded pixels wrap modulo 2^32 in the split,
        # which the decoder's modulo-2^32 reconstruction recovers exactly
        # (such tiles are always encoded losslessly, see encode()).
        if self.zblank is not None:
            valid = uval[buf != self.zblank]
            if valid.size > 0:
                baseline = int(valid.min())
            else:  # every pixel is null; any baseline is exact
                baseline = int(np.uint32(np.int64(self.zblank) + 0x80000000))
        else:
            baseline = int(uval.min())

        rebased = uval - np.uint32(baseline)  # wraps modulo 2^32, as intended
        upper = (rebased >> 16).astype(np.uint16)
        lower = (rebased & 0xFFFF).astype(np.uint16)

        # The high plane MUST be encoded losslessly: an error of 1 there
        # becomes 65536 in the reconstructed value.  NEAR on the low plane
        # alone keeps the total absolute error within max_err.  An
        # identically-zero high plane (tile range fit in 16 bits after the
        # rebase) is signalled with upper_len = 0 instead of a stream.
        upper_stream = b"" if not upper.any() else self._encode_stream(upper, 0)
        lower_stream = self._encode_stream(lower, max_err)

        return (
            len(upper_stream).to_bytes(4, "big")
            + baseline.to_bytes(4, "big")
            + upper_stream
            + lower_stream
        )


class JPEGXL(Codec):
    """
    The JPEG XL (ISO/IEC 18181) image compression format, designed as a universal 
    successor to JPEG. JPEG XL offers state-of-the-art compression with both lossless 
    and lossy modes, supporting high dynamic range, wide color gamut, and high bit depth 
    images. The format uses a modular transform coding architecture with advanced 
    features like progressive decoding, alpha channel support, and animation.

    The algorithm employs adaptive prediction, variable DCT transforms, context modeling,
    and entropy coding using ANS (Asymmetric Numeral Systems). It achieves superior 
    compression ratios compared to other formats while maintaining high visual quality
    and fast encoding/decoding speeds. JPEG XL is particularly well-suited for both 
    photographic and synthetic images across web, professional, and archival use cases.
    
    Can compress both int and float data.

    Parameters
    ----------
    max_err
        The near-lossless maximum error parameter. When set to 0, the compression
        is fully lossless, meaning the decompressed image will be bit-for-bit
        identical to the original. Values greater than 0 enable near-lossless
        compression, where each reconstructed sample differs from the original by
        no more than the specified error value. Default value is 0. Only works for
        integer data.

    effort
       Controls the encoder effort level (1-9). Higher values enable more thorough
       compression techniques at the cost of slower encoding. Default is 7.

    References
    ----------
       [1] Alakuijala, J., et al. (2019). JPEG XL Next-Generation Image Compression 
           Architecture and Coding Tools. Proc. SPIE 11137, Applications of Digital
           Image Processing XLII.
       [2] Rhatushnyak, A., & Wassenberg, J. (2020). The Design of JPEG XL. 
           Proceedings of the Picture Coding Symposium (PCS).
    """

    codec_id = "JPEGXL"

    def __init__(self, *, effort: int = DEFAULT_JPEGXL_EFFORT, max_err: int = DEFAULT_NEAR_LOSSLESS_MAXERR, quantization_mask: np.ndarray = None):
        if not HAS_IMAGECODECS:
            raise ImportError(
                "The 'imagecodecs' package is required for JPEG-XL compression. "
                "Install it with: pip install imagecodecs"
            )
        self.max_err = max_err
        self.effort = effort
        self.quantization_mask = quantization_mask

    def decode(self, buf):
        """
        Decompress buffer using the JPEG-LS algorithm.

        Parameters
        ----------
        buf : bytes or array_like
            The buffer to decompress.

        Returns
        -------
        buf : np.ndarray
            The decompressed buffer.
        """
        cbytes = np.frombuffer(_as_native_endian_array(buf), dtype=np.uint8).tobytes()
        return jpegxl_decode(cbytes)

    def encode(self, buf):
        """
        Compress the data in the buffer using the JPEG-LS algorithm.

        Parameters
        ----------
        buf : bytes or array_like
            The buffer to compress.

        Returns
        -------
        bytes
            The compressed bytes.
        """

        if buf.dtype == np.int16:
            # Arithmetic conversion to unsigned: same method as CFITSIO (+32768)
            buf = (buf.astype(np.int32) + 32768).astype(np.uint16)
        elif buf.dtype == np.int8:
            buf = (buf.astype(np.int16) + 128).astype(np.uint8)

        if self.max_err > 0:
            if buf.dtype == np.uint8:
                nbits = 8
            elif buf.dtype == np.uint16:
                nbits = 16
            else:
                raise RuntimeError("JPEG-XL near-lossless mode can only compress integer data. Set max_error=0 to compress floats.")

            buf = quantize_integer_arr(buf, self.max_err, self.quantization_mask, nbits)

        # Squeeze leading size-1 dimensions so imagecodecs sees a 2D image
        buf = np.squeeze(buf)
        if buf.ndim < 2:
            buf = buf.reshape(1, -1)
        return jpegxl_encode(buf, lossless=True, effort=self.effort)