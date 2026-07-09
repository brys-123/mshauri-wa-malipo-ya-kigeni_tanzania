import os
import cmdstanpy
import prophet

# Find exactly where Prophet expects its bundled CmdStan to live
prophet_dir = os.path.dirname(prophet.__file__)
target_dir = os.path.join(prophet_dir, "stan_model")

print("Prophet package is installed at:", prophet_dir)
print("Installing CmdStan 2.33.1 into:", target_dir)
print("This will take several minutes. Do not close this window.")

cmdstanpy.install_cmdstan(
    version="2.33.1",
    dir=target_dir,
    compiler=True,
    overwrite=True,
)

print("\nDone. Now run: python debug_prophet2.py")