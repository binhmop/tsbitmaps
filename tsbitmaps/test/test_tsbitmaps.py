import numpy as np
import pandas as pd
import unittest

from tsbitmaps.tsbitmapper import TSBitMapper


class TestBitmapAlgorithm(unittest.TestCase):
    def test_bitmap(self):
        bmp = TSBitMapper(feature_window_size=5, num_bins=8, level_size=2, lag_window_size=10, lead_window_size=10,
                          q=95)
        x = np.random.rand(500)
        binned_x = bmp.discretize(x)

        self.assertEqual(len(binned_x), len(x))
        self.assertTrue(set(binned_x) == set('01234567'))

        sample_bitmap = bmp.get_bitmap('01234567890123')
        self.assertEqual(len(sample_bitmap), 10)
        self.assertTrue('45' in sample_bitmap.keys())
        self.assertTrue('90' in sample_bitmap.keys())
        self.assertEqual(sample_bitmap['01'], 1)

        sample_bitmap_w = bmp.get_bitmap_with_feat_window('01234567890123')
        self.assertEqual(len(sample_bitmap_w), 8)
        self.assertTrue('45' not in sample_bitmap_w.keys())
        self.assertTrue('90' not in sample_bitmap_w.keys())
        self.assertEqual(sample_bitmap_w['01'], 1)

        ypred = bmp.fit_predict(x)
        scores = bmp.get_ref_bitmap_scores()
        self.assertTrue((scores[0:bmp.lag_window_size] == 0.0).all())
        self.assertTrue((scores[bmp.lag_window_size:-bmp.lead_window_size] >= 0).all())
        self.assertTrue(0 < (ypred == -1).sum() <= 25)

    def test_anomaly_detection_ecg(self):
        ecg_norm = np.loadtxt('data/ecg_normal.txt')
        ecg_anom = np.loadtxt('data/ecg_anom.txt')

        bmp = TSBitMapper(feature_window_size=20, num_bins=5, level_size=3, lag_window_size=200, lead_window_size=40)
        ypred_unsupervised = bmp.fit_predict(ecg_anom)
        self.assertTrue(0 < (ypred_unsupervised == -1).sum() <= 3)

        bmp.fit(ecg_norm)
        ypred_supervised = bmp.predict(ecg_anom)
        self.assertTrue(0 < (ypred_supervised == -1).sum() <= 3)

    def test_anomaly_detection_pattern(self):
        pattern_norm = np.loadtxt('data/pattern_normal.txt')
        pattern_anom = pd.read_csv('data/pattern_anom.txt').iloc[:, 0]

        bmp = TSBitMapper(feature_window_size=50, num_bins=5, level_size=2, lag_window_size=200, lead_window_size=100)
        ypred_unsupervised = bmp.fit_predict(pattern_anom)
        self.assertTrue(0 < (ypred_unsupervised == -1).sum() <= 3)

        bmp.fit(pattern_norm)
        ypred_supervised = bmp.predict(pattern_anom)
        self.assertTrue(0 < (ypred_supervised == -1).sum() <= 3)


if __name__ == '__main__':
    unittest.main()
