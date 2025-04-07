"""
Test script to examine the attributes of the ScopeData object from lecroyparser
"""
import sys
from pathlib import Path
from lecroyparser import ScopeData

# Get the first trc file
trc_file = Path(__file__).resolve().parent.parent.parent/"tes01"/"generated_data"/"raw_trc"/"p01"/"r004"/"C1--Trace--00000.trc"

if not trc_file.exists():
    print(f"File not found: {trc_file}")
    sys.exit(1)

# Load the file
data = ScopeData(str(trc_file))

# Print all attributes
print("Available attributes in ScopeData:")
for attr in dir(data):
    if not attr.startswith('__'):
        try:
            value = getattr(data, attr)
            print(f"{attr}: {value}")
        except Exception as e:
            print(f"{attr}: Error accessing - {e}")