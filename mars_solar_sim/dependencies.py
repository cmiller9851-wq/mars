import sys

REQUIRED_MODULES = ["numpy", "mmap", "struct", "ctypes"]

def verify_environment():
    print("=== INITIALIZING PYTHONISTA 3 RUNTIME ENVIRONMENT ===")
    print(f"Python Version: {sys.version}")
    
    missing = []
    for mod in REQUIRED_MODULES:
        try:
            __import__(mod)
            print(f"[OK] Module loaded: {mod}")
        except ImportError:
            missing.append(mod)
            print(f"[FAIL] Missing module: {mod}")
            
    if missing:
        raise EnvironmentError(f"Critical dependencies missing: {missing}")
    print("=== ALL SYSTEMS GO ===\n")

if __name__ == "__main__":
    verify_environment()