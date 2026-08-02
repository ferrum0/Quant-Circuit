import os
import sys

# Allows running the test suite against the repo checkout without installing
# the package first (`pip install -e .` also works and makes this a no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
