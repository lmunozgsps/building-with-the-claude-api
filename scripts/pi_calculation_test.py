"""
Unit tests for pi_calculation module.
"""

import unittest
import math
from pi_calculation import (
    calculate_pi,
    calculate_pi_leibniz,
    calculate_pi_nilakantha,
    calculate_pi_chudnovsky,
    calculate_pi_monte_carlo
)


class TestPiCalculation(unittest.TestCase):
    """Test cases for pi calculation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.actual_pi = math.pi
        self.tolerance_5_digits = 0.00001  # Tolerance for 5 decimal places
        self.tolerance_4_digits = 0.0001   # Tolerance for 4 decimal places
    
    def test_leibniz_method(self):
        """Test the Leibniz method for calculating pi."""
        result = calculate_pi_leibniz(iterations=1000000)
        self.assertAlmostEqual(result, self.actual_pi, places=5,
                               msg=f"Leibniz method result {result} not close enough to pi")
    
    def test_leibniz_with_fewer_iterations(self):
        """Test the Leibniz method with fewer iterations."""
        result = calculate_pi_leibniz(iterations=10000)
        # With fewer iterations, we expect less precision
        self.assertAlmostEqual(result, self.actual_pi, places=3)
    
    def test_nilakantha_method(self):
        """Test the Nilakantha method for calculating pi."""
        result = calculate_pi_nilakantha(iterations=100000)
        self.assertAlmostEqual(result, self.actual_pi, places=5,
                               msg=f"Nilakantha method result {result} not close enough to pi")
    
    def test_nilakantha_converges_faster(self):
        """Test that Nilakantha converges faster than Leibniz."""
        # With same number of iterations, Nilakantha should be more accurate
        iterations = 10000
        nilakantha_result = calculate_pi_nilakantha(iterations=iterations)
        leibniz_result = calculate_pi_leibniz(iterations=iterations)
        
        nilakantha_error = abs(nilakantha_result - self.actual_pi)
        leibniz_error = abs(leibniz_result - self.actual_pi)
        
        self.assertLess(nilakantha_error, leibniz_error,
                        msg="Nilakantha should converge faster than Leibniz")
    
    def test_chudnovsky_method(self):
        """Test the Chudnovsky method for calculating pi."""
        result = calculate_pi_chudnovsky(precision=2)
        self.assertAlmostEqual(result, self.actual_pi, places=5,
                               msg=f"Chudnovsky method result {result} not close enough to pi")
    
    def test_chudnovsky_high_precision(self):
        """Test the Chudnovsky method with higher precision."""
        result = calculate_pi_chudnovsky(precision=3)
        # Chudnovsky should give very high precision even with few terms
        self.assertAlmostEqual(result, self.actual_pi, places=10)
    
    def test_monte_carlo_method(self):
        """Test the Monte Carlo method for calculating pi."""
        # Monte Carlo is probabilistic, so we use a larger tolerance
        result = calculate_pi_monte_carlo(samples=1000000)
        self.assertAlmostEqual(result, self.actual_pi, places=2,
                               msg=f"Monte Carlo method result {result} not close enough to pi")
    
    def test_calculate_pi_default(self):
        """Test the default calculate_pi function."""
        result = calculate_pi()
        self.assertAlmostEqual(result, self.actual_pi, places=5)
    
    def test_calculate_pi_leibniz(self):
        """Test calculate_pi with Leibniz method."""
        result = calculate_pi(method='leibniz', iterations=1000000)
        self.assertAlmostEqual(result, self.actual_pi, places=5)
    
    def test_calculate_pi_nilakantha(self):
        """Test calculate_pi with Nilakantha method."""
        result = calculate_pi(method='nilakantha', iterations=100000)
        self.assertAlmostEqual(result, self.actual_pi, places=5)
    
    def test_calculate_pi_chudnovsky(self):
        """Test calculate_pi with Chudnovsky method."""
        result = calculate_pi(method='chudnovsky', precision=2)
        self.assertAlmostEqual(result, self.actual_pi, places=5)
    
    def test_calculate_pi_monte_carlo(self):
        """Test calculate_pi with Monte Carlo method."""
        result = calculate_pi(method='monte_carlo', samples=500000)
        self.assertAlmostEqual(result, self.actual_pi, places=1)
    
    def test_calculate_pi_invalid_method(self):
        """Test that calculate_pi raises ValueError for invalid method."""
        with self.assertRaises(ValueError):
            calculate_pi(method='invalid_method')
    
    def test_pi_to_5_digits(self):
        """Test that we can calculate pi accurate to 5 decimal places (3.14159)."""
        # Test with the fastest/most accurate method
        result = calculate_pi_chudnovsky(precision=2)
        expected = 3.14159
        # Check that first 5 decimal places match
        self.assertAlmostEqual(result, expected, places=5,
                               msg=f"Result {result:.10f} should match {expected} to 5 decimal places")
    
    def test_all_methods_achieve_5_digit_accuracy(self):
        """Test that all deterministic methods can achieve 5-digit accuracy."""
        methods_and_params = [
            ('leibniz', {'iterations': 1000000}),
            ('nilakantha', {'iterations': 100000}),
            ('chudnovsky', {'precision': 2}),
        ]
        
        for method, params in methods_and_params:
            with self.subTest(method=method):
                result = calculate_pi(method=method, **params)
                self.assertAlmostEqual(result, self.actual_pi, places=5,
                                       msg=f"{method} method should achieve 5-digit accuracy")
    
    def test_result_is_float(self):
        """Test that all methods return float values."""
        self.assertIsInstance(calculate_pi(), float)
        self.assertIsInstance(calculate_pi_leibniz(), float)
        self.assertIsInstance(calculate_pi_nilakantha(), float)
        self.assertIsInstance(calculate_pi_chudnovsky(), float)
        self.assertIsInstance(calculate_pi_monte_carlo(), float)
    
    def test_positive_result(self):
        """Test that all methods return positive values."""
        self.assertGreater(calculate_pi(), 0)
        self.assertGreater(calculate_pi_leibniz(), 0)
        self.assertGreater(calculate_pi_nilakantha(), 0)
        self.assertGreater(calculate_pi_chudnovsky(), 0)
        self.assertGreater(calculate_pi_monte_carlo(), 0)
    
    def test_result_in_reasonable_range(self):
        """Test that results are in a reasonable range around pi."""
        methods = [
            calculate_pi_leibniz(),
            calculate_pi_nilakantha(),
            calculate_pi_chudnovsky(),
        ]
        
        for result in methods:
            self.assertGreater(result, 3.0)
            self.assertLess(result, 3.3)


class TestPiCalculationPerformance(unittest.TestCase):
    """Performance and edge case tests."""
    
    def test_leibniz_minimal_iterations(self):
        """Test Leibniz with minimal iterations."""
        result = calculate_pi_leibniz(iterations=1)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)
    
    def test_nilakantha_minimal_iterations(self):
        """Test Nilakantha with minimal iterations."""
        result = calculate_pi_nilakantha(iterations=1)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)
    
    def test_chudnovsky_minimal_precision(self):
        """Test Chudnovsky with minimal precision."""
        result = calculate_pi_chudnovsky(precision=1)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)


def run_tests():
    """Run all tests and display results."""
    # Create a test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestPiCalculation))
    suite.addTests(loader.loadTestsFromTestCase(TestPiCalculationPerformance))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_tests()
