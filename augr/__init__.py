"""CMB B-mode Fisher forecast framework."""

# Enable 64-bit precision in JAX. The foreground SEDs involve steep power laws
# and exponentials (modified blackbody at high frequencies, synchrotron at low)
# whose autodiff tangents lose precision in float32, producing NaN.
import jax

jax.config.update("jax_enable_x64", True)
