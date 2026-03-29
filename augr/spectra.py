"""
spectra.py — CMB BB power spectrum templates.

Loads precomputed C_ell^BB from CAMB data files and provides
JAX-traceable interpolation for use in the Fisher forecast.

Convention:
    C_ell^{CMB,BB}(r, A_L) = r * C_ell^{tensor,r=1} + A_L * C_ell^{lensing}

All spectra in μK² (C_ell, not D_ell).
"""

from __future__ import annotations

import os
import numpy as np
import jax.numpy as jnp

# Default data file locations (relative to this file)
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DEFAULT_LENS_FILE = os.path.join(_DATA_DIR, "camb_lens_nobb.dat")
DEFAULT_TENSOR_FILE = os.path.join(_DATA_DIR, "camb_r1_nobb.dat")


class CMBSpectra:
    """CMB BB power spectrum templates for Fisher forecasting.

    Holds precomputed lensing and tensor (r=1) C_ell^BB arrays and
    interpolates to arbitrary ell grids. Both r and A_lens enter
    linearly, so cl_bb() is JAX-differentiable in those parameters.

    Args:
        lens_file:   Path to lensed-ΛCDM BB file (ell, C_ell columns).
        tensor_file: Path to tensor r=1 BB file (ell, C_ell columns).
    """

    def __init__(self,
                 lens_file: str = DEFAULT_LENS_FILE,
                 tensor_file: str = DEFAULT_TENSOR_FILE):
        lens_data = np.loadtxt(lens_file, comments="#")
        tensor_data = np.loadtxt(tensor_file, comments="#")

        self._ells_lens = lens_data[:, 0].astype(int)
        self._cl_lens = jnp.array(lens_data[:, 1])      # C_ell lensing BB [μK²]

        self._ells_tensor = tensor_data[:, 0].astype(int)
        self._cl_tensor = jnp.array(tensor_data[:, 1])  # C_ell tensor BB r=1 [μK²]

        self.ell_min = max(int(self._ells_lens[0]), int(self._ells_tensor[0]))
        self.ell_max = min(int(self._ells_lens[-1]), int(self._ells_tensor[-1]))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def cl_bb(self, ells: jnp.ndarray, r: float, A_lens: float = 1.0) -> jnp.ndarray:
        """Total CMB BB power: r * C_ell^tensor + A_lens * C_ell^lensing.

        JAX-traceable in r and A_lens (both enter linearly).
        Uses linear interpolation on the precomputed CAMB grids.

        Args:
            ells:   1-D array of multipole values to evaluate at.
            r:      Tensor-to-scalar ratio.
            A_lens: Residual lensing amplitude (1 = full lensing, 0 = perfect delensing).

        Returns:
            C_ell^BB array of same length as ells, in μK².
        """
        cl_lens_interp = self._interpolate(ells, self._ells_lens, self._cl_lens)
        cl_tensor_interp = self._interpolate(ells, self._ells_tensor, self._cl_tensor)
        return r * cl_tensor_interp + A_lens * cl_lens_interp

    def cl_lensing(self, ells: jnp.ndarray) -> jnp.ndarray:
        """Lensed ΛCDM BB spectrum (A_lens=1, r=0) at requested ells."""
        return self._interpolate(ells, self._ells_lens, self._cl_lens)

    def cl_tensor_r1(self, ells: jnp.ndarray) -> jnp.ndarray:
        """Tensor BB spectrum for r=1 at requested ells."""
        return self._interpolate(ells, self._ells_tensor, self._cl_tensor)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _interpolate(ells_out: jnp.ndarray,
                     ells_in: np.ndarray,
                     cl_in: jnp.ndarray) -> jnp.ndarray:
        """Linear interpolation of cl_in from ells_in grid to ells_out.

        Uses jnp.interp which is JAX-traceable (but not differentiable
        w.r.t. ells_out — that's fine since ells are fixed integers).
        Values outside [ells_in[0], ells_in[-1]] are set to zero.
        """
        return jnp.interp(ells_out, jnp.array(ells_in, dtype=float),
                          cl_in, left=0.0, right=0.0)

    @classmethod
    def generate_with_camb(cls,
                           output_dir: str,
                           ell_max: int = 500,
                           H0: float = 67.4,
                           ombh2: float = 0.0224,
                           omch2: float = 0.120,
                           tau: float = 0.054,
                           As: float = 2.1e-9,
                           ns: float = 0.965) -> "CMBSpectra":
        """Generate templates using CAMB and write to output_dir.

        Requires camb to be installed. For standard use, the precomputed
        files in data/ are sufficient and CAMB is not needed at runtime.
        """
        import camb

        os.makedirs(output_dir, exist_ok=True)
        lens_file = os.path.join(output_dir, "camb_lens_nobb.dat")
        tensor_file = os.path.join(output_dir, "camb_r1_nobb.dat")

        # Lensed ΛCDM BB
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau)
        pars.InitPower.set_params(As=As, ns=ns, r=0)
        pars.WantTensors = False
        pars.set_for_lmax(ell_max, lens_potential_accuracy=1)
        pars.Want_CMB_lensing = True
        results = camb.get_results(pars)
        powers = results.get_lensed_scalar_cls(lmax=ell_max, CMB_unit='muK', raw_cl=True)
        ells = np.arange(powers.shape[0])
        np.savetxt(lens_file,
                   np.column_stack([ells, powers[:, 2]]),
                   header="ell  C_ell_BB [muK^2]  lensed LCDM, no tensors (r=0)",
                   fmt=["%.0f", "%.6e"])

        # Tensor BB for r=1
        pars2 = camb.CAMBparams()
        pars2.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau)
        pars2.InitPower.set_params(As=As, ns=ns, r=1.0, nt=0)
        pars2.WantTensors = True
        pars2.set_for_lmax(ell_max, lens_potential_accuracy=0)
        results2 = camb.get_results(pars2)
        powers2 = results2.get_tensor_cls(lmax=ell_max, CMB_unit='muK', raw_cl=True)
        ells2 = np.arange(powers2.shape[0])
        np.savetxt(tensor_file,
                   np.column_stack([ells2, powers2[:, 2]]),
                   header="ell  C_ell_BB [muK^2]  tensor r=1, unlensed",
                   fmt=["%.0f", "%.6e"])

        return cls(lens_file=lens_file, tensor_file=tensor_file)
