import traceback

print("Step 1: checking cmdstanpy path...")
try:
    import cmdstanpy
    print("cmdstan_path:", cmdstanpy.cmdstan_path())
except Exception:
    traceback.print_exc()

print("\nStep 2: trying to directly load the CmdStanPy backend Prophet uses...")
try:
    from prophet.models import StanBackendEnum
    backend_cls = StanBackendEnum['CMDSTANPY'].value
    backend = backend_cls()
    print("Backend object created:", backend)
except Exception:
    traceback.print_exc()

print("\nStep 3: trying to load/compile the actual Stan model Prophet ships with...")
try:
    from prophet.models import StanBackendEnum
    backend_cls = StanBackendEnum['CMDSTANPY'].value
    backend = backend_cls()
    backend.load_model()
    print("Model loaded successfully!")
except Exception:
    traceback.print_exc()