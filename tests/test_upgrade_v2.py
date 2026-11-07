"""Integration test: old self-contained volume -> V2 exported observatory."""
import base64
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from upgrade_existing_world import upgrade, read_payload

class UpgradeV2Test(unittest.TestCase):
    def test_upgraded_world_contains_complex_reference(self):
        n=8
        rng=np.random.default_rng(44)
        raw=rng.uniform(-.7,.7,(n,n,n,3,2)).astype(np.float32)
        signed=np.rint((raw*.5+.5)*255).astype(np.uint8)
        c0=signed[...,:2,:].reshape(n,n,n,4)
        c1=np.zeros((n,n,n,4),np.uint8)
        c1[...,:2]=signed[...,2,:]
        rgb=np.full((n,n,n,4),128,np.uint8)
        enc=lambda v:base64.b64encode(v.tobytes(order="C")).decode("ascii")
        old=dict(size=n,t0=5.,t1=9.,anchors=[5.,9.],zeros=[],
                 rgb=enc(rgb),c0=enc(c0),c1=enc(c1))
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/"old.html"
            src.write_text("<script>const EMBEDDED = "+json.dumps(old)+";</script>",encoding="utf8")
            report=upgrade(src)
            output=Path(report["output"])
            self.assertTrue(output.exists())
            html=output.read_text("utf8")
            payload=read_payload(html)
            self.assertIn("harm0",payload)
            self.assertIn("harm1",payload)
            self.assertIn("segments",payload)
            self.assertIn("harmonic_rms_curve",payload)
            self.assertEqual(len(payload["harmonic_rms_curve"]),n)
            self.assertEqual(payload["harmonic_rms_curve"][0],0)
            self.assertEqual(payload["harmonic_rms_curve"][-1],0)
            self.assertEqual(len(base64.b64decode(payload["harm0"])),n*n*n*4)
            self.assertIn("uHarm0",html)
            self.assertIn("uPartner",html)

if __name__=="__main__":
    unittest.main()
