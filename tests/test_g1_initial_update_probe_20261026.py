"""Gold timeout-bootstrap boundary for reconstructing frozen PPO batches."""
import unittest
import numpy as np
from scripts.diagnose_g1_initial_update_20261026 import bootstrap_rewards


class BootstrapBoundary(unittest.TestCase):
    def test_known_small_example(self):
        rewards=np.array([[1.,2.],[3.,4.]])
        values=np.array([[[10.],[20.]],[[30.],[40.]]])
        timeouts=np.array([[False,True],[True,False]])
        before=rewards.copy()
        result=bootstrap_rewards(rewards,values,timeouts,.9)
        np.testing.assert_allclose(result,[[[1.],[20.]],[[30.],[4.]]],rtol=0,atol=1e-12)
        np.testing.assert_array_equal(rewards,before)

    def test_no_timeout_is_identity(self):
        rewards=np.arange(6,dtype=float).reshape(2,3)
        result=bootstrap_rewards(rewards,np.ones((2,3,1))*100,np.zeros((2,3),bool),.99)
        np.testing.assert_array_equal(result,rewards[...,None])

    def test_reject_broadcasting_wrong_shape(self):
        with self.assertRaises(ValueError):
            bootstrap_rewards(np.zeros((2,3)),np.zeros((2,3)),np.zeros((2,3),bool),.99)


if __name__=='__main__':
    unittest.main()
