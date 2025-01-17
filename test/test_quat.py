import pytest
import numpy as np
import geometry_msgs

import ros2_numpy as rnp
import tf_transformations as trans

class TestQuat():
    def test_representation(self):
        q = trans.quaternion_from_euler(0., 0., 0.)
        assert np.allclose(q, np.array([0., 0., 0., 1.]))

    def test_identity_transform(self):
        H = rnp.numpify(geometry_msgs.msg.Transform())
        assert np.allclose(H, np.eye(4))
