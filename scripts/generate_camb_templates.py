#!/usr/bin/env python3
"""Generate lensed ΛCDM BB and tensor (r=1) BB spectra using CAMB.

Saves two files in data/:
  camb_lens_nobb.dat  — lensed ΛCDM BB, no tensors
  camb_r1_nobb.dat    — unlensed tensor BB for r=1

Columns: ell, C_ell [μK²]  (not D_ell = ell*(ell+1)*C_ell/2pi)

Cosmology: Planck 2018 best-fit (TT+TE+EE+lowE+lensing)
  H0=67.4, ombh2=0.0224, omch2=0.120, tau=0.054
  As=2.1e-9, ns=0.965

Usage:
  python -m scripts.generate_camb_templates
  # or directly:
  python scripts/generate_camb_templates.py
"""

import os
import numpy as np
import camb

# Output to data/ relative to the repo root (parent of scripts/)
OUTDIR = os.path.join(os.path.dirname(__file__), os.pardir, "data")
ELL_MAX = 500


def generate_templates(outdir: str = OUTDIR, ell_max: int = ELL_MAX) -> None:
    """Generate lensing and tensor BB templates and write to disk."""
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)

    # -------------------------------------------------------------------
    # Lensed ΛCDM BB (no tensors)
    # -------------------------------------------------------------------
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.4, ombh2=0.0224, omch2=0.120, tau=0.054)
    pars.InitPower.set_params(As=2.1e-9, ns=0.965, r=0)
    pars.WantTensors = False
    pars.set_for_lmax(ell_max, lens_potential_accuracy=1)
    pars.Want_CMB_lensing = True

    results = camb.get_results(pars)
    # raw_cl=True returns C_ell (not D_ell); shape (lmax+1, 4): TT, EE, BB, TE
    powers = results.get_lensed_scalar_cls(lmax=ell_max, CMB_unit='muK', raw_cl=True)
    ells = np.arange(powers.shape[0])
    cl_lens_bb = powers[:, 2]

    outfile = os.path.join(outdir, "camb_lens_nobb.dat")
    np.savetxt(outfile,
               np.column_stack([ells, cl_lens_bb]),
               header="ell  C_ell_BB [muK^2]  lensed LCDM, no tensors (r=0), Planck2018 cosmology",
               fmt=["%.0f", "%.6e"])
    print(f"Wrote {outfile}")
    print(f"  lensing BB peak: {cl_lens_bb[50:150].max():.3e} muK^2 at ell~{ells[50 + cl_lens_bb[50:150].argmax()]}")

    # -------------------------------------------------------------------
    # Tensor BB for r=1 (unlensed)
    # -------------------------------------------------------------------
    pars2 = camb.CAMBparams()
    pars2.set_cosmology(H0=67.4, ombh2=0.0224, omch2=0.120, tau=0.054)
    pars2.InitPower.set_params(As=2.1e-9, ns=0.965, r=1.0, nt=0)
    pars2.WantTensors = True
    pars2.set_for_lmax(ell_max, lens_potential_accuracy=0)

    results2 = camb.get_results(pars2)
    # shape (lmax+1, 4): TT, EE, BB, TE
    powers2 = results2.get_tensor_cls(lmax=ell_max, CMB_unit='muK', raw_cl=True)
    ells2 = np.arange(powers2.shape[0])
    cl_tensor_bb = powers2[:, 2]

    outfile2 = os.path.join(outdir, "camb_r1_nobb.dat")
    np.savetxt(outfile2,
               np.column_stack([ells2, cl_tensor_bb]),
               header="ell  C_ell_BB [muK^2]  tensor r=1 unlensed, Planck2018 cosmology",
               fmt=["%.0f", "%.6e"])
    print(f"Wrote {outfile2}")
    print(f"  reionization bump: {cl_tensor_bb[2:15].max():.3e} muK^2 at ell~{ells2[2 + cl_tensor_bb[2:15].argmax()]}")
    print(f"  recombination peak: {cl_tensor_bb[60:120].max():.3e} muK^2 at ell~{ells2[60 + cl_tensor_bb[60:120].argmax()]}")


if __name__ == "__main__":
    generate_templates()
