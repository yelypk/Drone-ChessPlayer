import numpy as np
from src.drone.core.types import CameraIntrinsics
from src.drone.core.geom import pixel_to_normalized_ray

def test_pixel_to_normalized_ray_shapes():
    K = np.array([[500.,0.,320.],[0.,500.,240.],[0.,0.,1.]])
    intr = CameraIntrinsics(K=K, dist=np.zeros(5))
    ray = pixel_to_normalized_ray(320, 240, intr)
    assert ray.shape == (3,)
    assert np.isclose(np.linalg.norm(ray), 1.0, atol=1e-6)
