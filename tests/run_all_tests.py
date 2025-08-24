#!/usr/bin/env python3
"""
Comprehensive test runner for LatticeQuant.

This script runs all tests in the tests folder and provides a summary of results.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Add the parent directory to the path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_test_file(test_file):
    """Run a single test file and return the result."""
    print(f"\n{'='*60}")
    print(f"Running: {test_file}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, timeout=300)
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ PASSED ({duration:.2f}s)")
            return True, duration, result.stdout, result.stderr
        else:
            print(f"❌ FAILED ({duration:.2f}s)")
            print(f"Error: {result.stderr}")
            return False, duration, result.stdout, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT (>300s)")
        return False, 300, "", "Test timed out"
    except Exception as e:
        print(f"💥 ERROR: {e}")
        return False, 0, "", str(e)

def categorize_tests():
    """Categorize tests by type."""
    test_files = []
    
    # Core functionality tests
    core_tests = [
        'test_closest_point.py',
        'test_nested_lattice_quantizer.py',
        'test_gemv.py',
        'test_adaptive_matvec.py',
        'test_layer_wise_histogram.py',
        'test_standalone_layer_wise_histogram.py'
    ]
    
    # Coarse-to-fine decoding tests
    coarse_to_fine_tests = []
    
    # Analysis and debugging tests
    analysis_tests = [
        'test_error_types.py',
        'test_error_trends.py'
    ]
    
    # Other tests
    other_tests = []
    
    return {
        'Core Functionality': core_tests,
        'Analysis & Debugging': analysis_tests,
        'Other': other_tests
    }

def run_test_category(category_name, test_files):
    """Run all tests in a category."""
    print(f"\n{'#'*80}")
    print(f"CATEGORY: {category_name}")
    print(f"{'#'*80}")
    
    results = []
    for test_file in test_files:
        if os.path.exists(test_file):
            success, duration, stdout, stderr = run_test_file(test_file)
            results.append({
                'file': test_file,
                'success': success,
                'duration': duration,
                'stdout': stdout,
                'stderr': stderr
            })
        else:
            print(f"⚠️  File not found: {test_file}")
            results.append({
                'file': test_file,
                'success': False,
                'duration': 0,
                'stdout': '',
                'stderr': f'File not found: {test_file}'
            })
    
    return results

def print_summary(all_results):
    """Print a summary of all test results."""
    print(f"\n{'#'*80}")
    print("TEST SUMMARY")
    print(f"{'#'*80}")
    
    total_tests = 0
    total_passed = 0
    total_failed = 0
    total_duration = 0
    
    for category, results in all_results.items():
        print(f"\n{category}:")
        category_passed = sum(1 for r in results if r['success'])
        category_failed = len(results) - category_passed
        category_duration = sum(r['duration'] for r in results)
        
        print(f"  Passed: {category_passed}/{len(results)}")
        print(f"  Failed: {category_failed}/{len(results)}")
        print(f"  Duration: {category_duration:.2f}s")
        
        total_tests += len(results)
        total_passed += category_passed
        total_failed += category_failed
        total_duration += category_duration
        
        # List failed tests
        failed_tests = [r['file'] for r in results if not r['success']]
        if failed_tests:
            print(f"  Failed tests: {', '.join(failed_tests)}")
    
    print(f"\nOVERALL SUMMARY:")
    print(f"  Total tests: {total_tests}")
    print(f"  Passed: {total_passed}")
    print(f"  Failed: {total_failed}")
    print(f"  Success rate: {total_passed/total_tests*100:.1f}%")
    print(f"  Total duration: {total_duration:.2f}s")
    
    if total_failed == 0:
        print(f"\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {total_failed} tests failed. Check the output above for details.")

def run_specific_test(test_name):
    """Run a specific test by name."""
    test_categories = categorize_tests()
    
    for category, test_files in test_categories.items():
        for test_file in test_files:
            if test_name in test_file:
                print(f"Running specific test: {test_file}")
                success, duration, stdout, stderr = run_test_file(test_file)
                return success
    
    print(f"Test not found: {test_name}")
    return False

def main():
    """Main function to run all tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run LatticeQuant tests')
    parser.add_argument('--test', type=str, help='Run a specific test (partial name match)')
    parser.add_argument('--category', type=str, help='Run tests from a specific category')
    parser.add_argument('--list', action='store_true', help='List all available tests')
    
    args = parser.parse_args()
    
    if args.list:
        print("Available test categories and files:")
        test_categories = categorize_tests()
        for category, test_files in test_categories.items():
            print(f"\n{category}:")
            for test_file in test_files:
                print(f"  {test_file}")
        return
    
    if args.test:
        success = run_specific_test(args.test)
        sys.exit(0 if success else 1)
    
    if args.category:
        test_categories = categorize_tests()
        if args.category in test_categories:
            results = run_test_category(args.category, test_categories[args.category])
            all_results = {args.category: results}
            print_summary(all_results)
        else:
            print(f"Category not found: {args.category}")
            print(f"Available categories: {list(test_categories.keys())}")
        return
    
    # Run all tests
    print("Starting comprehensive test suite for LatticeQuant...")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    test_categories = categorize_tests()
    all_results = {}
    
    for category, test_files in test_categories.items():
        results = run_test_category(category, test_files)
        all_results[category] = results
    
    print_summary(all_results)
    
    # Exit with appropriate code
    total_failed = sum(sum(1 for r in results if not r['success']) 
                      for results in all_results.values())
    sys.exit(0 if total_failed == 0 else 1)

if __name__ == "__main__":
    main()
