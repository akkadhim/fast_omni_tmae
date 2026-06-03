__version__ = "0.1.0"

try:
    import fast_tmae.host.build.fast_tmaelib
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import cffi compiled libraries: {e}. "
                  "You may need to rebuild the package with 'pip install -e .'", 
                  ImportWarning)
