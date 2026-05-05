#!/usr/bin/env julia
# Run Falcons.jl at LiteBIRD-standard scan params and dump h_k maps for k=1,2,4
# to data/falcons_validation/. Used to validate the augr closed-form crossing-
# angle envelope (tier-2 spin formalism) before scaffolding augr/crosslinks.py.
#
# Conventions follow Takase 2024 / Falcons.jl:
#   alpha = precession opening angle (anti-sun to spin axis)   [augr names this beta]
#   beta  = spin opening angle (spin axis to boresight)         [augr names this alpha]
#
# h_{n,m}(p) = <exp(-i(n*psi + m*phi))> averaged over hits in pixel p.
# psi = crossing angle, phi = HWP angle (0 here, no HWP).

using Falcons
using FITSIO

const NSIDE          = 128
const DURATION_SEC   = 60 * 60 * 24 * 365            # 1 year
const SAMPLING_RATE  = 1.0                            # Hz
const ALPHA_DEG      = 45.0                           # Falcons / Takase alpha
const BETA_DEG       = 50.0                           # Falcons / Takase beta
const PREC_PERIOD_MIN = 192.348
const SPIN_PERIOD_MIN = 20.0
const SPIN_NS        = [1, 2, 4]
const SPIN_MS        = [0]
const DIVISION       = 12                             # year split into 12 monthly chunks

println("Configuring scan strategy...")
ss = Falcons.gen_ScanningStrategy(
    nside         = NSIDE,
    duration      = DURATION_SEC,
    sampling_rate = SAMPLING_RATE,
    alpha         = ALPHA_DEG,
    beta          = BETA_DEG,
    gamma         = 0.0,
    prec_rpm      = Falcons.period2rpm(PREC_PERIOD_MIN),
    spin_rpm      = Falcons.period2rpm(SPIN_PERIOD_MIN),
    hwp_rpm       = 0.0,
    start_point   = "equator",
    start_angle   = 0.0,
    coord         = "E",
)
Falcons.show_ss(ss)

println("\nRunning scan + spin field accumulation (this will take a few minutes)...")
@time field = Falcons.get_scanfield(ss;
                                    division = DIVISION,
                                    spin_n   = SPIN_NS,
                                    spin_m   = SPIN_MS)

# Where the data lives. Script is at scripts/falcons_validation/validate_hk.jl,
# data goes to forecasting/data/falcons_validation/.
const OUTDIR = normpath(joinpath(@__DIR__, "..", "..", "data", "falcons_validation"))
mkpath(OUTDIR)
println("\nWriting outputs to: $OUTDIR")

function save_complex_map(path, h_complex)
    # Two image HDUs: HDU[2] = real part, HDU[3] = imag part. (HDU[1] is empty primary.)
    f = FITS(path, "w")
    write(f, Float64.(real.(h_complex)))
    write(f, Float64.(imag.(h_complex)))
    close(f)
end

for (idx, n) in enumerate(SPIN_NS)
    h_n = field.h[idx, 1, :]
    outpath = joinpath(OUTDIR, "h$(n)_litebird_nside$(NSIDE).fits")
    save_complex_map(outpath, h_n)
    println("  wrote $outpath  (npix=$(length(h_n)))")
end

# Hitmap: simple integer array, single image HDU.
hitpath = joinpath(OUTDIR, "hitmap_litebird_nside$(NSIDE).fits")
fh = FITS(hitpath, "w")
write(fh, Float64.(field.hitmap))
close(fh)
println("  wrote $hitpath")

# Tiny metadata sidecar so Python knows the conventions used.
meta_path = joinpath(OUTDIR, "metadata.txt")
open(meta_path, "w") do io
    println(io, "# Falcons.jl h_k validation run")
    println(io, "nside           = $NSIDE")
    println(io, "duration_sec    = $DURATION_SEC")
    println(io, "sampling_rate   = $SAMPLING_RATE")
    println(io, "alpha_deg       = $ALPHA_DEG  (Falcons/Takase: precession opening, anti-sun to spin axis)")
    println(io, "beta_deg        = $BETA_DEG  (Falcons/Takase: spin opening, spin axis to boresight)")
    println(io, "prec_period_min = $PREC_PERIOD_MIN")
    println(io, "spin_period_min = $SPIN_PERIOD_MIN")
    println(io, "spin_ns         = $SPIN_NS")
    println(io, "coord           = E (ecliptic)")
    println(io, "convention      = h(n,m,psi,phi) = exp(-i (n*psi + m*phi))")
    println(io, "hwp_off         = true (m=0 only)")
    println(io, "ordering        = RING (Healpix.jl default for ang2pixRing)")
end

println("\nDone.")
