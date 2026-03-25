//! FFT engine abstraction for pluggable FFT implementations.
//!
//! The `FftEngine` trait provides a uniform interface for real-to-complex
//! and complex-to-real FFT operations.  The default implementation wraps
//! the `realfft` crate (portable, works on all architectures).  Optional
//! platform engines can be swapped in behind cargo features.

#[cfg(feature = "fftw")]
use fftw::array::AlignedVec;
#[cfg(feature = "fftw")]
use fftw::plan::{C2RPlan, C2RPlan64, R2CPlan, R2CPlan64};
#[cfg(feature = "fftw")]
use fftw::types::{c64, Flag};
#[cfg(any(feature = "fftw", all(feature = "vdsp", target_os = "macos")))]
use std::cell::UnsafeCell;
#[cfg(all(feature = "vdsp", target_os = "macos"))]
use std::ffi::c_void;

use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;
use std::sync::Arc;

/// Trait for real FFT engines used by DwtSquarer.
///
/// Implementations must provide forward (real → complex) and inverse
/// (complex → real) transforms.  The forward transform takes N real values
/// and produces N/2+1 complex values.  The inverse does the opposite.
///
/// Neither transform applies normalization — the caller must divide by N
/// after the inverse transform.
pub trait FftEngine: Send {
    /// Forward real-to-complex FFT.
    ///
    /// `input` is length N (contents are modified by the transform).
    /// `output` is length N/2+1 (overwritten with the complex spectrum).
    fn forward(&self, input: &mut [f64], output: &mut [Complex<f64>], scratch: &mut [Complex<f64>]);

    /// Inverse complex-to-real FFT.
    ///
    /// `input` is length N/2+1 (contents are modified by the transform).
    /// `output` is length N (overwritten with real-valued result).
    fn inverse(&self, input: &mut [Complex<f64>], output: &mut [f64], scratch: &mut [Complex<f64>]);

    /// FFT length N.
    fn len(&self) -> usize;

    /// Required scratch buffer length for forward transform.
    fn forward_scratch_len(&self) -> usize;

    /// Required scratch buffer length for inverse transform.
    fn inverse_scratch_len(&self) -> usize;
}

/// Default FFT engine wrapping the `realfft` crate.
///
/// This is portable and works on all architectures.  It uses RustFFT
/// internally with automatic SIMD detection where available.
pub struct RealFftEngine {
    len: usize,
    r2c: Arc<dyn RealToComplex<f64>>,
    c2r: Arc<dyn ComplexToReal<f64>>,
}

impl RealFftEngine {
    /// Create a new RealFftEngine for transforms of length `len`.
    /// `len` must be a power of 2 and >= 4.
    pub fn new(len: usize) -> Self {
        let mut planner = RealFftPlanner::<f64>::new();
        let r2c = planner.plan_fft_forward(len);
        let c2r = planner.plan_fft_inverse(len);
        RealFftEngine { len, r2c, c2r }
    }
}

impl FftEngine for RealFftEngine {
    fn forward(&self, input: &mut [f64], output: &mut [Complex<f64>], scratch: &mut [Complex<f64>]) {
        self.r2c
            .process_with_scratch(input, output, scratch)
            .expect("real-to-complex FFT failed");
    }

    fn inverse(&self, input: &mut [Complex<f64>], output: &mut [f64], scratch: &mut [Complex<f64>]) {
        self.c2r
            .process_with_scratch(input, output, scratch)
            .expect("complex-to-real IFFT failed");
    }

    fn len(&self) -> usize {
        self.len
    }

    fn forward_scratch_len(&self) -> usize {
        self.r2c.get_scratch_len()
    }

    fn inverse_scratch_len(&self) -> usize {
        self.c2r.get_scratch_len()
    }
}

#[cfg(all(feature = "vdsp", target_os = "macos"))]
#[repr(C)]
struct DSPDoubleComplex {
    real: f64,
    imag: f64,
}

#[cfg(all(feature = "vdsp", target_os = "macos"))]
#[repr(C)]
struct DSPDoubleSplitComplex {
    realp: *mut f64,
    imagp: *mut f64,
}

#[cfg(all(feature = "vdsp", target_os = "macos"))]
#[allow(non_upper_case_globals)]
const kFFTDirection_Forward: i32 = 1;
#[cfg(all(feature = "vdsp", target_os = "macos"))]
#[allow(non_upper_case_globals)]
const kFFTDirection_Inverse: i32 = -1;
#[cfg(all(feature = "vdsp", target_os = "macos"))]
#[allow(non_upper_case_globals)]
const kFFTRadix2: i32 = 0;

#[cfg(all(feature = "vdsp", target_os = "macos"))]
unsafe extern "C" {
    fn vDSP_create_fftsetupD(log2n: u64, radix: i32) -> *mut c_void;
    fn vDSP_destroy_fftsetupD(setup: *mut c_void);
    fn vDSP_fft_zriptD(
        setup: *mut c_void,
        c: *const DSPDoubleSplitComplex,
        ic: i64,
        buffer: *const DSPDoubleSplitComplex,
        log2n: u64,
        direction: i32,
    );
    fn vDSP_ctozD(
        z: *const DSPDoubleComplex,
        iz: i64,
        c: *const DSPDoubleSplitComplex,
        ic: i64,
        n: u64,
    );
    fn vDSP_ztocD(
        c: *const DSPDoubleSplitComplex,
        ic: i64,
        z: *mut DSPDoubleComplex,
        iz: i64,
        n: u64,
    );
}

#[cfg(all(feature = "vdsp", target_os = "macos"))]
#[inline]
fn make_split_complex(realp: &mut [f64], imagp: &mut [f64]) -> DSPDoubleSplitComplex {
    debug_assert_eq!(realp.len(), imagp.len());
    DSPDoubleSplitComplex {
        realp: realp.as_mut_ptr(),
        imagp: imagp.as_mut_ptr(),
    }
}

#[cfg(all(feature = "vdsp", target_os = "macos"))]
struct VdspInner {
    setup: *mut c_void,
    log2n: u64,
    len: usize,
    split_re: Vec<f64>,
    split_im: Vec<f64>,
    temp_re: Vec<f64>,
    temp_im: Vec<f64>,
}

/// FFT engine backed by Apple Accelerate vDSP real transforms.
///
/// Uses `vDSP_fft_zriptD` and adapts vDSP scaling to this crate's
/// unnormalized forward/inverse convention:
/// - forward output is divided by 2
/// - inverse output is multiplied by 2
#[cfg(all(feature = "vdsp", target_os = "macos"))]
pub struct VdspEngine {
    inner: UnsafeCell<VdspInner>,
}

#[cfg(all(feature = "vdsp", target_os = "macos"))]
impl VdspEngine {
    /// Create a new vDSP engine for transforms of length `len`.
    pub fn new(len: usize) -> Self {
        assert!(len.is_power_of_two(), "vDSP requires power-of-two FFT sizes");
        assert!(len >= 2, "FFT length must be >= 2");

        let log2n = len.trailing_zeros() as u64;
        let setup = unsafe {
            // SAFETY: Plain FFI call with POD arguments.
            vDSP_create_fftsetupD(log2n, kFFTRadix2)
        };
        assert!(!setup.is_null(), "failed to create vDSP FFT setup");

        let n_over_2 = len / 2;
        Self {
            inner: UnsafeCell::new(VdspInner {
                setup,
                log2n,
                len,
                split_re: vec![0.0; n_over_2],
                split_im: vec![0.0; n_over_2],
                temp_re: vec![0.0; n_over_2],
                temp_im: vec![0.0; n_over_2],
            }),
        }
    }
}

#[cfg(all(feature = "vdsp", target_os = "macos"))]
impl Drop for VdspEngine {
    fn drop(&mut self) {
        let inner = unsafe {
            // SAFETY: &mut self guarantees exclusive access.
            &mut *self.inner.get()
        };

        if !inner.setup.is_null() {
            unsafe {
                // SAFETY: setup was allocated by vDSP_create_fftsetupD.
                vDSP_destroy_fftsetupD(inner.setup);
            }
            inner.setup = std::ptr::null_mut();
        }
    }
}

// SAFETY: VdspEngine can be moved across threads (Send) but is intentionally
// !Sync. Mutable access is internal and mediated by UnsafeCell.
#[cfg(all(feature = "vdsp", target_os = "macos"))]
unsafe impl Send for VdspEngine {}

#[cfg(all(feature = "vdsp", target_os = "macos"))]
impl FftEngine for VdspEngine {
    fn forward(&self, input: &mut [f64], output: &mut [Complex<f64>], _scratch: &mut [Complex<f64>]) {
        let inner = unsafe {
            // SAFETY: VdspEngine is !Sync; caller cannot race shared mutable access.
            &mut *self.inner.get()
        };

        assert_eq!(input.len(), inner.len, "forward input length mismatch");
        assert_eq!(output.len(), inner.len / 2 + 1, "forward output length mismatch");

        let n_over_2 = inner.len / 2;
        let split = make_split_complex(&mut inner.split_re, &mut inner.split_im);
        let temp = make_split_complex(&mut inner.temp_re, &mut inner.temp_im);

        unsafe {
            // SAFETY: Interpret N real samples as N/2 interleaved complex pairs.
            vDSP_ctozD(
                input.as_ptr().cast::<DSPDoubleComplex>(),
                2,
                &split as *const DSPDoubleSplitComplex,
                1,
                n_over_2 as u64,
            );

            // SAFETY: split/temp buffers match setup length and remain valid.
            vDSP_fft_zriptD(
                inner.setup,
                &split as *const DSPDoubleSplitComplex,
                1,
                &temp as *const DSPDoubleSplitComplex,
                inner.log2n,
                kFFTDirection_Forward,
            );
        }

        // vDSP forward scales by 2 for real FFT; divide by 2 to provide
        // unnormalized-forward output consistent with other engines.
        output[0] = Complex::new(inner.split_re[0] * 0.5, 0.0);
        for k in 1..n_over_2 {
            output[k] = Complex::new(inner.split_re[k] * 0.5, inner.split_im[k] * 0.5);
        }
        output[n_over_2] = Complex::new(inner.split_im[0] * 0.5, 0.0);
    }

    fn inverse(&self, input: &mut [Complex<f64>], output: &mut [f64], _scratch: &mut [Complex<f64>]) {
        let inner = unsafe {
            // SAFETY: Same serialization rationale as in `forward`.
            &mut *self.inner.get()
        };

        assert_eq!(input.len(), inner.len / 2 + 1, "inverse input length mismatch");
        assert_eq!(output.len(), inner.len, "inverse output length mismatch");

        let n_over_2 = inner.len / 2;

        // Pack our N/2+1 Hermitian spectrum into vDSP's packed split format.
        inner.split_re[0] = input[0].re;
        inner.split_im[0] = input[n_over_2].re;
        for k in 1..n_over_2 {
            inner.split_re[k] = input[k].re;
            inner.split_im[k] = input[k].im;
        }

        let split = make_split_complex(&mut inner.split_re, &mut inner.split_im);
        let temp = make_split_complex(&mut inner.temp_re, &mut inner.temp_im);

        unsafe {
            // SAFETY: split/temp buffers are valid and sized for this transform.
            vDSP_fft_zriptD(
                inner.setup,
                &split as *const DSPDoubleSplitComplex,
                1,
                &temp as *const DSPDoubleSplitComplex,
                inner.log2n,
                kFFTDirection_Inverse,
            );

            // SAFETY: Write N reals as N/2 interleaved complex pairs.
            vDSP_ztocD(
                &split as *const DSPDoubleSplitComplex,
                1,
                output.as_mut_ptr().cast::<DSPDoubleComplex>(),
                2,
                n_over_2 as u64,
            );
        }

        // vDSP inverse scales by 1/2 for real FFT; multiply by 2 so this engine
        // returns unnormalized inverse output like other backends.
        for v in output.iter_mut() {
            *v *= 2.0;
        }
    }

    fn len(&self) -> usize {
        let inner = unsafe {
            // SAFETY: Immutable interior access.
            &*self.inner.get()
        };
        inner.len
    }

    fn forward_scratch_len(&self) -> usize {
        0
    }

    fn inverse_scratch_len(&self) -> usize {
        0
    }
}

#[cfg(feature = "fftw")]
struct FftwInner {
    r2c: R2CPlan64,
    c2r: C2RPlan64,
    real_buf: AlignedVec<f64>,
    complex_buf: AlignedVec<c64>,
}

#[cfg(feature = "fftw")]
#[inline]
fn copy_complex_to_c64(src: &[Complex<f64>], dst: &mut [c64]) {
    assert_eq!(src.len(), dst.len(), "complex input/output length mismatch");
    debug_assert_eq!(std::mem::size_of::<Complex<f64>>(), std::mem::size_of::<c64>());
    debug_assert_eq!(std::mem::align_of::<Complex<f64>>(), std::mem::align_of::<c64>());
    unsafe {
        // SAFETY: Complex<f64> and c64 are layout-compatible complex64 values,
        // lengths are equal, and src/dst do not overlap.
        std::ptr::copy_nonoverlapping(src.as_ptr().cast::<c64>(), dst.as_mut_ptr(), src.len());
    }
}

#[cfg(feature = "fftw")]
#[inline]
fn copy_c64_to_complex(src: &[c64], dst: &mut [Complex<f64>]) {
    assert_eq!(src.len(), dst.len(), "complex input/output length mismatch");
    debug_assert_eq!(std::mem::size_of::<Complex<f64>>(), std::mem::size_of::<c64>());
    debug_assert_eq!(std::mem::align_of::<Complex<f64>>(), std::mem::align_of::<c64>());
    unsafe {
        // SAFETY: c64 and Complex<f64> are layout-compatible complex64 values,
        // lengths are equal, and src/dst do not overlap.
        std::ptr::copy_nonoverlapping(src.as_ptr().cast::<Complex<f64>>(), dst.as_mut_ptr(), src.len());
    }
}

/// FFT engine backed by FFTW real transforms.
///
/// Plans are created with `Flag::MEASURE`. Internal FFTW-aligned buffers are
/// reused for all calls; caller data is copied in/out.
#[cfg(feature = "fftw")]
pub struct FftwEngine {
    len: usize,
    inner: UnsafeCell<FftwInner>,
}

#[cfg(feature = "fftw")]
impl FftwEngine {
    /// Create a new FFTW engine for transforms of length `len`.
    pub fn new(len: usize) -> Self {
        let r2c = R2CPlan64::aligned(&[len], Flag::MEASURE)
            .expect("failed to create FFTW R2C plan");
        let c2r = C2RPlan64::aligned(&[len], Flag::MEASURE)
            .expect("failed to create FFTW C2R plan");

        let real_buf = AlignedVec::new(len);
        let complex_buf = AlignedVec::new(len / 2 + 1);

        Self {
            len,
            inner: UnsafeCell::new(FftwInner {
                r2c,
                c2r,
                real_buf,
                complex_buf,
            }),
        }
    }
}

// SAFETY: FftwEngine is movable across threads (Send) but not shareable (it is
// not Sync). All mutable access is through UnsafeCell and requires logical
// exclusive use. With FFTW built from source with thread support, executing
// plans on the owning thread after move is safe.
#[cfg(feature = "fftw")]
unsafe impl Send for FftwEngine {}

#[cfg(feature = "fftw")]
impl FftEngine for FftwEngine {
    fn forward(&self, input: &mut [f64], output: &mut [Complex<f64>], _scratch: &mut [Complex<f64>]) {
        assert_eq!(input.len(), self.len, "forward input length mismatch");
        assert_eq!(output.len(), self.len / 2 + 1, "forward output length mismatch");

        let inner = unsafe {
            // SAFETY: FftwEngine is !Sync, so concurrent shared access is not
            // possible through safe Rust; this mutable access is serialized.
            &mut *self.inner.get()
        };

        inner.real_buf[..].copy_from_slice(input);

        {
            let (r2c, real_buf, complex_buf) =
                (&mut inner.r2c, &mut inner.real_buf, &mut inner.complex_buf);
            r2c.r2c(real_buf, complex_buf)
                .expect("FFTW real-to-complex FFT failed");
        }

        copy_c64_to_complex(&inner.complex_buf[..], output);
        input.copy_from_slice(&inner.real_buf[..]);
    }

    fn inverse(&self, input: &mut [Complex<f64>], output: &mut [f64], _scratch: &mut [Complex<f64>]) {
        assert_eq!(input.len(), self.len / 2 + 1, "inverse input length mismatch");
        assert_eq!(output.len(), self.len, "inverse output length mismatch");

        let inner = unsafe {
            // SAFETY: Same reasoning as in `forward`.
            &mut *self.inner.get()
        };

        copy_complex_to_c64(input, &mut inner.complex_buf[..]);

        {
            let (c2r, complex_buf, real_buf) =
                (&mut inner.c2r, &mut inner.complex_buf, &mut inner.real_buf);
            c2r.c2r(complex_buf, real_buf)
                .expect("FFTW complex-to-real IFFT failed");
        }

        output.copy_from_slice(&inner.real_buf[..]);
        copy_c64_to_complex(&inner.complex_buf[..], input);
    }

    fn len(&self) -> usize {
        self.len
    }

    fn forward_scratch_len(&self) -> usize {
        0
    }

    fn inverse_scratch_len(&self) -> usize {
        0
    }
}