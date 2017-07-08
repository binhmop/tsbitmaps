import numpy as np
import pandas as pd
import unittest

from tsbitmaps.tsbitmapper import TSBitMapper


class TestBitmapAlgorithm(unittest.TestCase):

    # @unittest.skip("tmp")
    def test0_get_bitmap(self):
        bmp = TSBitMapper(feature_window_size = 5, num_bins = 8, level_size = 2, lag_window_size = 10, lead_window_size = 10)
        x = np.random.rand(500)
        binned_x = bmp.discretize(x)

        self.assertEqual(len(binned_x), len(x))
        self.assertTrue(set(binned_x) == set('01234567'))

        sample_bitmap = bmp.get_bitmap('01234567890123')
        self.assertEquals(len(sample_bitmap), 10)
        self.assertTrue('45' in sample_bitmap.keys())
        self.assertTrue('90' in sample_bitmap.keys())
        self.assertEquals(sample_bitmap['01'], 1)

        sample_bitmap_w = bmp.get_bitmap_with_feat_window('01234567890123')
        self.assertEquals(len(sample_bitmap_w), 8)
        self.assertTrue('45' not in sample_bitmap_w.keys())
        self.assertTrue('90' not in sample_bitmap_w.keys())
        self.assertEquals(sample_bitmap_w['01'], 1)

        bmp.fit(x)
        scores = bmp.get_bitmap_scores()
        anoms = bmp.get_anomalies(3)
        self.assertTrue(len(anoms) < len(scores))

    def test1_anomaly_detection(self):
        ecg_anom = np.loadtxt('data/ecg_anom.txt')
        bmp = TSBitMapper(feature_window_size = 20, num_bins = 5, level_size= 3, lag_window_size = 200, lead_window_size = 40)
        bmp.fit(ecg_anom)
        scores = bmp.get_bitmap_scores()
        anoms = bmp.get_anomalies(3)
        self.assertFalse(anoms.empty)

    def test2_anomaly_detection(self):
        pattern_anom = pd.read_csv('data/pattern_anom.txt').iloc[:,0]
        bmp = TSBitMapper(feature_window_size = 50, num_bins = 5, level_size= 2, lag_window_size = 200, lead_window_size = 100)
        bmp.fit(pattern_anom)
        scores = bmp.get_bitmap_scores()
        anoms = bmp.get_anomalies(3)
        self.assertFalse(anoms.empty)

if __name__ == '__main__':
    unittest.main()

