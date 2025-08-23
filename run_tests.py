#!/usr/bin/env python3
"""
Script to run tests with the correct Python path setup.
"""

import sys
import os
import subprocess

def run_test(test_file):
    """Run a specific test file with the correct Python path."""
    # Add the current directory to Python path
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd()
    
    # Run the test
    result = subprocess.run([sys.executable, test_file], env=env, capture_output=True, text=True)
    
    print(f"Running {test_file}...")
    if result.returncode == 0:
        print(f"✅ {test_file} passed")
        print(result.stdout)
    else:
        print(f"❌ {test_file} failed")
        print(result.stderr)
    
    return result.returncode == 0

def main():
    """Run all tests or a specific test."""
    if len(sys.argv) > 1:
        # Run specific test
        test_file = sys.argv[1]
        if not os.path.exists(test_file):
            print(f"Test file {test_file} not found")
            return 1
        
        success = run_test(test_file)
        return 0 if success else 1
    else:
        # Run the test decoding parameter test
        success = run_test("test_decoding_parameter.py")
        return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
