import unittest
import base64
import numpy as np
from zeta_harmonic import harmonic_slab, reference_payload

class HarmonicTests(unittest.TestCase):
    def test_boundaries_and_shape(self):
        raw=np.random.default_rng(22).standard_normal((11,9,13,3,2)).astype("f")
        H=harmonic_slab(raw)
        self.assertEqual(H.shape,raw.shape)
        np.testing.assert_array_equal(H[0],raw[0])
        np.testing.assert_array_equal(H[-1],raw[-1])
        self.assertGreater(float(np.mean(np.abs(H[5]-raw[5]))),.01)

    def test_constant_spatial_mode(self):
        raw=np.zeros((7,8,8,3,2),"f")
        raw[0,:,:,0,0]=.3
        raw[-1,:,:,0,0]=.7
        H=harmonic_slab(raw)
        for j in range(7):
            np.testing.assert_allclose(H[j,:,:,0,0],.3+.4*j/6,atol=1e-6)

    def test_neumann_cosine_mode(self):
        n=17
        mode=np.cos(np.pi*np.linspace(0,2,n)/2)
        raw=np.zeros((9,n,n,3,2),"f")
        raw[0,:,:,0,0]=mode[None,:]
        H=harmonic_slab(raw)
        k=np.pi/2
        expect=np.broadcast_to(mode[None,:]*(np.sinh(k)/np.sinh(2*k)),(n,n))
        np.testing.assert_allclose(H[4,:,:,0,0],expect,atol=1e-5)

    def test_packing_and_discrepancy_curve(self):
        raw=np.ones((5,7,8,3,2),"f")*.12
        raw[2]*=2
        d=reference_payload(raw,1.)
        self.assertEqual(len(d["harmonic_rms_curve"]),5)
        self.assertEqual(d["harmonic_rms_curve"][0],0)
        self.assertEqual(d["harmonic_rms_curve"][-1],0)
        self.assertGreater(d["harmonic_rms_curve"][2],0)
        self.assertEqual(len(base64.b64decode(d["harm0"])),5*7*8*4)
        self.assertEqual(len(base64.b64decode(d["harm1"])),5*7*8*4)

if __name__=="__main__":
    unittest.main()
