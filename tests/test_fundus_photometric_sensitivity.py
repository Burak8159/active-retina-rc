from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from active_retina.audit_fundus_photometric_sensitivity import photometric_image


class FundusPhotometricSensitivityTests(unittest.TestCase):
    def setUp(self) -> None:
        y, x = np.mgrid[0:32, 0:40]
        self.image = Image.fromarray(np.stack((x * 4, y * 6, (x + y) * 3), axis=2).astype(np.uint8))

    def test_identity_preserves_pixels(self) -> None:
        actual = np.asarray(photometric_image(self.image, "identity"))
        self.assertTrue(np.array_equal(actual, np.asarray(self.image)))

    def test_green_channel_replicates_green_plane(self) -> None:
        actual = np.asarray(photometric_image(self.image, "green_channel"))
        self.assertTrue(np.array_equal(actual[:, :, 0], actual[:, :, 1]))
        self.assertTrue(np.array_equal(actual[:, :, 1], actual[:, :, 2]))

    def test_all_normalizations_return_finite_rgb_images(self) -> None:
        for mode in ("gray_world", "channel_percentile", "green_autocontrast"):
            actual = np.asarray(photometric_image(self.image, mode))
            self.assertEqual(actual.shape, (32, 40, 3))
            self.assertTrue(np.isfinite(actual).all())
            self.assertGreaterEqual(int(actual.min()), 0)
            self.assertLessEqual(int(actual.max()), 255)

    def test_unknown_transform_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            photometric_image(self.image, "invented")


if __name__ == "__main__":
    unittest.main()
