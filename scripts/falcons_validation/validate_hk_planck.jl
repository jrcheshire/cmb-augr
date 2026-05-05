#!/usr/bin/env julia
# Run Falcons.jl with Planck-style scan params (alpha=7.5 deg, beta=85 deg,
# T_prec = 6 months, T_spin = 1 min, 1 yr) and dump h_k for k=1,2,4.
# Uses Falcons' built-in `get_satellite("Planck")` preset for the scan params,
# overrides nside/duration/sampling for our target.

using Falcons
using FITSIO

const SATELLITE = "Planck"
const NSIDE = 128
const DURATION_SEC = 3 * 60 * 60 * 24 * 365   # 3 yr; 1 yr leaves the sky sparsely sampled at 1 Hz
const SAMPLING_RATE = 1.0
const SPIN_NS = [1, 2, 4]
const SPIN_MS = [0]
const DIVISION = 36

ss = Falcons.get_satellite(SATELLITE)
ss.nside = NSIDE
ss.duration = DURATION_SEC
ss.sampling_rate = SAMPLING_RATE
ss.coord = "E"
ss.start_point = "equator"

Falcons.show_ss(ss)

println("\nRunning scan + spin field accumulation...")
@time field = Falcons.get_scanfield(ss; division=DIVISION, spin_n=SPIN_NS, spin_m=SPIN_MS)

const OUTDIR = normpath(joinpath(@__DIR__, "..", "..", "data", "falcons_validation"))
mkpath(OUTDIR)

function save_complex_map(path, h_complex)
    f = FITS(path, "w")
    write(f, Float64.(real.(h_complex)))
    write(f, Float64.(imag.(h_complex)))
    close(f)
end

for (idx, n) in enumerate(SPIN_NS)
    h_n = field.h[idx, 1, :]
    outpath = joinpath(OUTDIR, "h$(n)_planck_nside$(NSIDE).fits")
    save_complex_map(outpath, h_n)
    println("  wrote $outpath")
end

hitpath = joinpath(OUTDIR, "hitmap_planck_nside$(NSIDE).fits")
fh = FITS(hitpath, "w")
write(fh, Float64.(field.hitmap))
close(fh)
println("  wrote $hitpath")

# Metadata sidecar
meta_path = joinpath(OUTDIR, "metadata_planck.txt")
open(meta_path, "w") do io
    println(io, "# Falcons.jl h_k validation, Planck-like params")
    println(io, "alpha_deg       = $(ss.alpha)  (Falcons/Takase: precession opening)")
    println(io, "beta_deg        = $(ss.beta)  (Falcons/Takase: spin opening)")
    println(io, "prec_period_min = ", 1.0/ss.prec_rpm)
    println(io, "spin_period_min = ", 1.0/ss.spin_rpm)
    println(io, "nside           = $NSIDE")
    println(io, "duration_sec    = $DURATION_SEC")
    println(io, "sampling_rate   = $SAMPLING_RATE")
    println(io, "spin_ns         = $SPIN_NS")
    println(io, "coord           = E (ecliptic)")
end
println("\nDone.")
