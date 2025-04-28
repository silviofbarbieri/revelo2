import unittest
import numpy as np
from scipy.sparse import csr_matrix

class TestCSRMatrixRound(unittest.TestCase):
    
    def test_round_method_exists(self):
        """Test that the round method exists for csr_matrix"""
        a = csr_matrix((3, 4))
        try:
            round(a)
        except TypeError as e:
            self.fail(f"round() raised TypeError: {e}")
    
    def test_round_with_integer_data(self):
        """Test round with integer data (should remain unchanged)"""
        data = np.array([1, 2, 3])
        row = np.array([0, 1, 2])
        col = np.array([0, 1, 2])
        a = csr_matrix((data, (row, col)), shape=(3, 4))
        
        rounded = round(a)
        self.assertTrue(isinstance(rounded, csr_matrix), "Should return a csr_matrix")
        np.testing.assert_array_equal(rounded.toarray(), a.toarray())
    
    def test_round_with_float_data(self):
        """Test round with float data"""
        data = np.array([1.4, 2.6, 3.5])
        row = np.array([0, 1, 2])
        col = np.array([0, 1, 2])
        a = csr_matrix((data, (row, col)), shape=(3, 4))
        
        expected = np.zeros((3, 4))
        expected[0, 0] = 1
        expected[1, 1] = 3
        expected[2, 2] = 4
        
        rounded = round(a)
        self.assertTrue(isinstance(rounded, csr_matrix), "Should return a csr_matrix")
        np.testing.assert_array_equal(rounded.toarray(), expected)
    
    def test_round_with_ndigits(self):
        """Test round with specified ndigits parameter"""
        data = np.array([1.44, 2.66, 3.55])
        row = np.array([0, 1, 2])
        col = np.array([0, 1, 2]) 
        a = csr_matrix((data, (row, col)), shape=(3, 4))
        
        expected1 = np.zeros((3, 4))
        expected1[0, 0] = 1.4
        expected1[1, 1] = 2.7
        expected1[2, 2] = 3.6
        
        rounded1 = round(a, 1)
        self.assertTrue(isinstance(rounded1, csr_matrix), "Should return a csr_matrix")
        np.testing.assert_array_almost_equal(rounded1.toarray(), expected1)
        
        expected2 = np.zeros((3, 4))
        expected2[0, 0] = 1.44
        expected2[1, 1] = 2.66
        expected2[2, 2] = 3.55
        
        rounded2 = round(a, 2)
        self.assertTrue(isinstance(rounded2, csr_matrix), "Should return a csr_matrix")
        np.testing.assert_array_almost_equal(rounded2.toarray(), expected2)
    
    def test_round_preserves_sparsity(self):
        """Test that round preserves the sparsity pattern"""
        data = np.array([1.4, 2.6, 3.5])
        row = np.array([0, 1, 2])
        col = np.array([0, 1, 2])
        a = csr_matrix((data, (row, col)), shape=(3, 4))
        
        rounded = round(a)
        
        # Check that zeros remained zeros (sparsity preserved)
        a_array = a.toarray()
        rounded_array = rounded.toarray()
        
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if a_array[i, j] == 0:
                    self.assertEqual(rounded_array[i, j], 0, 
                                    f"Zero at position ({i},{j}) should remain zero after rounding")

if __name__ == "__main__":
    unittest.main()
