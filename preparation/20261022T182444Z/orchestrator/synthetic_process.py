"""Provide a synthetic child process for timeout checks."""
import sys
import time
time.sleep(float(sys.argv[1]))
print('Synthetic process finished.')
