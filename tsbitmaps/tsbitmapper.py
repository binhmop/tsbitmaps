import numpy as np
import pandas as pd
from collections import defaultdict


class TSBitMapper:
    """
    
    Implements Time-series Bitmap model for unsupervised anomaly detection
    
    Based on the papers "Time-series Bitmaps: A Practical Visualization Tool for working with Large Time Series Databases"
    and "Assumption-Free Anomaly Detection in Time Series"
    
    Test data and parameter settings taken from http://alumni.cs.ucr.edu/~wli/SSDBM05/
    """

    def __init__(self, feature_window_size=None, num_bins=5, level_size=3, lag_window_size=None, lead_window_size=None):

        """
        
        :param int feature_window_size: should be about the size at which events happen
        :param int num_bins: number of equal-width bins, i.e. symbols, to discretize a time series
        :param int level_size: desired level of recursion of the bitmap
        :param int lag_window_size: how far to look back
        :param int lead_window_size: how far to look ahead
        """
        assert feature_window_size, 'feature_window_size must be given'
        assert min(lag_window_size,
                   lead_window_size) >= level_size, 'lag_window_size and lead_window_size must be >= feature_window_size'

        # bitmap parameters
        self.feature_window_size = feature_window_size
        self.level_size = level_size

        if lag_window_size is None:
            self.lag_window_size = 3 * self.feature_window_size
        else:
            self.lag_window_size = lag_window_size

        if lead_window_size is None:
            self.lead_window_size = 2 * self.feature_window_size
        else:
            self.lead_window_size = lead_window_size

        self.num_bins = num_bins

    def discretize(self, ts, num_bins=None, global_min=None, global_max=None):
        min_value = ts.min()
        max_value = ts.max()
        if min_value == max_value:
            min_value = global_min
            max_value = global_max
        if num_bins is None:
            num_bins = self.num_bins
        step = (max_value - min_value) / num_bins
        bins = np.arange(min_value, max_value, step)

        inds = np.digitize(ts, bins)
        binned_ts_str = ''.join([str(i - 1) for i in inds])
        return binned_ts_str

    def discretize_by_feat_window(self, ts, num_bins=None, feature_window_size=None):
        if num_bins is None:
            num_bins = self.num_bins

        if feature_window_size is None:
            feature_window_size = self.feature_window_size

        n = len(ts)
        windows = []
        global_min = ts.min()
        global_max = ts.max()
        for i in range(0, n - n % feature_window_size, feature_window_size):
            binned_fw = self.discretize(ts[i: i + feature_window_size], num_bins, global_min, global_max)
            windows.append(binned_fw)
        if n % feature_window_size > 0:
            last_binned_fw = self.discretize(ts[- (n % feature_window_size):], num_bins, global_min, global_max)
            windows.append(last_binned_fw)

        return ''.join(windows)

    def get_bitmap(self, chunk, level_size=None):
        """
        
        :param str chunk: symbol sequence representation of a univariate time series
        :param int level_size: desired level of recursion of the bitmap
        :return: bitmap representation of the time series
        """
        bitmap = defaultdict(int)
        n = len(chunk)
        if level_size is None:
            level_size = self.level_size
        for i in range(n):
            if i <= n - level_size:
                feat = chunk[i: i + level_size]
                bitmap[feat] += 1  # frequency count
        max_freq = max(bitmap.values())
        for feat in bitmap.keys():
            bitmap[feat] = bitmap[feat] / max_freq
        return bitmap

    def get_bitmap_with_feat_window(self, chunk, level_size=None, step=None):
        """
        
        :param str chunk: symbol sequence representation of a univariate time series
        :param int level_size: desired level of recursion of the bitmap
        :param int step: length of the feature window
        :return: bitmap representation of the time series
        """
        if step is None:
            step = self.feature_window_size
        if level_size is None:
            level_size = self.level_size

        bitmap = defaultdict(int)
        n = len(chunk)

        for i in range(0, n - n % step, step):

            for j in range(step - level_size + 1):
                feat = chunk[i + j: i + j + level_size]
                bitmap[feat] += 1  # frequency count

        if n % step > 0:
            for i in range(n - n % step, n - level_size + 1):
                feat = chunk[i: i + level_size]
                bitmap[feat] += 1

        max_freq = max(bitmap.values())
        for feat in bitmap.keys():
            bitmap[feat] = bitmap[feat] / max_freq
        return bitmap

    def _slide_chunks(self, ts):
        lag_bitmap = {}
        lead_bitmap = {}
        scores = np.zeros(len(ts))

        egress_lag_feat = ''
        egress_lead_feat = ''

        binned_ts = self.discretize_by_feat_window(ts)
        ts_len = len(binned_ts)

        lagws = self.lag_window_size
        leadws = self.lead_window_size
        featws = self.level_size

        for i in range(self.lag_window_size, ts_len - self.lead_window_size + 1):

            if i == self.lag_window_size:
                lag_chunk = binned_ts[i - lagws: i]
                lead_chunk = binned_ts[i: i + leadws]

                lag_bitmap = self.get_bitmap_with_feat_window(lag_chunk)
                lead_bitmap = self.get_bitmap_with_feat_window(lead_chunk)

                scores[i] = self.bitmap_distance(lag_bitmap, lead_bitmap)

                egress_lag_feat = lag_chunk[0: featws]
                egress_lead_feat = lead_chunk[0: featws]

            else:

                lag_chunk = binned_ts[i - lagws: i]
                lead_chunk = binned_ts[i: i + leadws]

                ingress_lag_feat = lag_chunk[-featws:]
                ingress_lead_feat = lead_chunk[-featws:]

                lag_bitmap[ingress_lag_feat] += 1
                lag_bitmap[egress_lag_feat] -= 1

                lead_bitmap[ingress_lead_feat] += 1
                lead_bitmap[egress_lead_feat] -= 1

                scores[i] = self.bitmap_distance(lag_bitmap, lead_bitmap)

                egress_lag_feat = lag_chunk[0: featws]
                egress_lead_feat = lead_chunk[0: featws]

        self.bitmap_scores = scores

    def bitmap_distance(self, lag_bitmap, lead_bitmap):
        """
        Computes the dissimilarity of two bitmaps
        """
        dist = 0
        lag_feats = set(lag_bitmap.keys())
        lead_feats = set(lead_bitmap.keys())
        shared_feats = lag_feats & lead_feats

        for feat in shared_feats:
            dist += (lead_bitmap[feat] - lag_bitmap[feat]) ** 2

        for feat in lag_feats - shared_feats:
            dist += lag_bitmap[feat] ** 2

        for feat in lead_feats - shared_feats:
            dist += lead_bitmap[feat] ** 2

        return dist

    def fit(self, ts):
        """
        Computes the bitmap distance at each timestamp in a univariate time series `ts`
        
        :param ts: 1-D numpy array or pandas.Series 
        
        """
        assert len(
            ts) >= self.lag_window_size + self.lead_window_size, 'sequence length must be larger than sum of lag_window_size and lead_window_size'
        self.ts = ts
        self._slide_chunks(ts)

    def get_bitmap_scores(self):
        return self.bitmap_scores

    def get_anomalies(self, num_std=3):
        """
        Get the abnormal scores which are `num_std` standard deviations from the mean score
        
        """
        scores = self.bitmap_scores
        mean_score = np.mean(scores[self.lag_window_size: -self.lead_window_size])
        std_score = np.std(scores[self.lag_window_size: -self.lead_window_size])

        upper_threshold = mean_score + num_std * std_score  # num_std standard deviations from the mean score

        inds = []
        anomalies = []
        ts_values = []

        for idx, score in enumerate(scores):
            if score >= upper_threshold:
                inds.append(idx)
                anomalies.append(score)
                ts_values.append(self.ts[idx])

        anomaly_df = pd.DataFrame({'idx': inds, 'abnormal_score': anomalies, 'ts_value': ts_values})
        return anomaly_df
