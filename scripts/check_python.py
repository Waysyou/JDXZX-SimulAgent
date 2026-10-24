"""Installer probe kept in a file to avoid PowerShell native-argument quoting."""
import struct
import sys

if sys.version_info[:2] != (3, 13) or struct.calcsize("P") != 8:
    raise SystemExit("This runtime baseline requires Python 3.13 x64.")
print("3.13-64")
