import glob
import os
import pandas as pd
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

# Beware of the leading space
# angle_type_prefix = ' pose_R'
ANGLE_TYPE_PREFIX = ' p_r'
# 7 or 9 is max and seems to be the best
FILTER_KERNEL = 9 
FRAME_RATE = 30

def get_peaks_params(input_dir, angle_type, include_nan, verbose=False):
    
    reach_ends_cnt = 0
    avg_w = []
    avg_h = []

    for csv_file in glob.glob(os.path.join(input_dir, '*.csv')):
        df = pd.read_csv(csv_file)

        # Resample to common frame rate
        csv_frame_rate = (df.shape[0] - 1) / np.sum(np.diff(df[' timestamp']))
        if verbose:
            print(f'csv frame rate: {csv_frame_rate}')
        n_resampled_points = int(1 + (len(df[ANGLE_TYPE_PREFIX + angle_type]) - 1) * FRAME_RATE / csv_frame_rate)
        angles = scipy.signal.resample(df[ANGLE_TYPE_PREFIX + angle_type], num=n_resampled_points)
        
        # Take the first derivative of the signal
        diff_angles = np.diff(angles)

        # Apply median filter
        filt_diff_angles = scipy.signal.medfilt(diff_angles, kernel_size=FILTER_KERNEL)

        # Low-pass filter
#         fc = 3
#         b, a = scipy.signal.butter(N=2, Wn=2*fc/FRAME_RATE, btype='low', analog=False, output='ba')
#         filt_diff_angles = scipy.signal.filtfilt(b, a, diff_angles)

        # Get peak locations
        pos_peaks, _ = scipy.signal.find_peaks(filt_diff_angles)
        neg_peaks, _ = scipy.signal.find_peaks(-filt_diff_angles)

        # Widths and heights for the current file/recording
        curr_w = []
        curr_h = []
        
        # Detect neighboring +- and -+
        i, j = 0, 0
        while i < len(pos_peaks) and j < len(neg_peaks):
            
            # Height difference of adjacent extremes + - or - +
            h = filt_diff_angles[pos_peaks[i]] - filt_diff_angles[neg_peaks[j]]

            if pos_peaks[i] == neg_peaks[j]:
                raise Exception('Positive and negative peaks coincide!')
            # + -
            elif pos_peaks[i] < neg_peaks[j]:
                # Search left for first negative peak neg_peaks[j-1]
                if j <= 0:
                    left_end = 0
                    reach_ends_cnt += 1
                    if verbose: print('L End reached + -')
                else:
                    left_end = neg_peaks[j-1]
                                    
                # Search right for first positive peak
                if i >= len(pos_peaks) - 1:
                    right_end = len(filt_diff_angles) - 1
                    reach_ends_cnt += 1
                    if verbose: print('R End reached + -')
                else:
                    right_end = pos_peaks[i+1]
                                        
                i += 1
                
            # - +
            else:
                # Search left for first positive peak
                if i <= 0:
                    left_end = 0
                    reach_ends_cnt += 1
                    if verbose: print('L End reached - +')
                else:
                    left_end = pos_peaks[i-1]
                                    
                # Search right for first negative peak
                if j >= len(neg_peaks) - 1:
                    right_end = len(filt_diff_angles) - 1
                    reach_ends_cnt += 1
                    if verbose: print('R End reached - +')
                else:
                    right_end = neg_peaks[j+1]
                                
                j += 1
                
            # Peak width
            w = abs(right_end - left_end) + 1
                
            curr_w.append(w)
            curr_h.append(h)       
            
        # If nans should not be included and no peaks found, skip
        if include_nan or len(curr_w) != 0:
            avg_w.append(np.mean(curr_w))
            avg_h.append(np.mean(curr_h))

        if verbose:
            # Plot
            plt.figure(figsize=(14, 4))
            plt.plot(diff_angles, 'k-o', label='1st derivative')
            plt.plot(filt_diff_angles, 'orange', label='filtered 1st derivative')
            for p in pos_peaks:
                plt.axvline(p, color='red')
            for p in neg_peaks:
                plt.axvline(p, color='blue')
            plt.legend()
            plt.xlim(0, 200)
            plt.show()
            print(f'Widths: {curr_w}')
            
    if verbose:
        print(f'Ends reached {reach_ends_cnt} times')
            
    return avg_w, avg_h


def annotate_dataframe(df, angle_type, clf, scaler, verbose=False):

    # Resample to common frame rate
    csv_frame_rate = (df.shape[0] - 1) / np.sum(np.diff(df[' timestamp']))
    if verbose:
        print(f'csv frame rate: {csv_frame_rate}')
    n_resampled_points = int(1 + (len(df[ANGLE_TYPE_PREFIX + angle_type]) - 1) * FRAME_RATE / csv_frame_rate)
    angles = scipy.signal.resample(df[ANGLE_TYPE_PREFIX + angle_type], num=n_resampled_points)

    # Take the first derivative of the signal
    diff_angles = np.diff(angles)

    # Apply median filter
    filt_diff_angles = scipy.signal.medfilt(diff_angles, kernel_size=FILTER_KERNEL)

    # Low-pass filter
#         fc = 3
#         b, a = scipy.signal.butter(N=2, Wn=2*fc/FRAME_RATE, btype='low', analog=False, output='ba')
#         filt_diff_angles = scipy.signal.filtfilt(b, a, diff_angles)

    # Get peak locations
    pos_peaks, _ = scipy.signal.find_peaks(filt_diff_angles)
    neg_peaks, _ = scipy.signal.find_peaks(-filt_diff_angles)

    # Probabilistic annotations (wrt positive classes: nod or shake)
    annotations = np.zeros(len(angles))

    # Number of detected proper peaks (by kNN)
    peak_cnt = 0

    # Detect neighboring +- and -+
    i, j = 0, 0
    while i < len(pos_peaks) and j < len(neg_peaks):

        # Height difference of adjacent extremes + - or - +
        h = filt_diff_angles[pos_peaks[i]] - filt_diff_angles[neg_peaks[j]]

        if pos_peaks[i] == neg_peaks[j]:
            raise Exception('Positive and negative peaks coincide!')
        # + -
        elif pos_peaks[i] < neg_peaks[j]:
            # Search left for first negative peak neg_peaks[j-1]
            if j <= 0:
                left_end = 0
                if verbose: print('L End reached + -')
            else:
                left_end = neg_peaks[j-1]

            # Search right for first positive peak
            if i >= len(pos_peaks) - 1:
                right_end = len(filt_diff_angles) - 1
                if verbose: print('R End reached + -')
            else:
                right_end = pos_peaks[i+1]

            i += 1

        # - +
        else:
            # Search left for first positive peak
            if i <= 0:
                left_end = 0
                if verbose: print('L End reached - +')
            else:
                left_end = pos_peaks[i-1]

            # Search right for first negative peak
            if j >= len(neg_peaks) - 1:
                right_end = len(filt_diff_angles) - 1
                if verbose: print('R End reached - +')
            else:
                right_end = neg_peaks[j+1]

            j += 1

        # Peak width
        w = abs(right_end - left_end) + 1

        # Z-normalize inputs
        znorm_input_X = scaler.transform([[w, h]])
        
        # Predict probability of the positive class
        # +1 and +2 are shifts due to the fact that np.diff decrements the number of samples and at the end we want to have len(angles) samples
        annotations[left_end + 1:right_end + 2] = clf.predict_proba(znorm_input_X)[0, 1]

        # If it is a proper detected peak, count it on
        if clf.predict(znorm_input_X)[0] == 1:
            peak_cnt += 1

    # Replicate the first element to match the length of angles
    if len(annotations) > 0:
        annotations[0] = annotations[1]
    assert len(annotations) == len(angles)
       
    if verbose:
        print(f'Found {peak_cnt} peaks.')
        print(annotations)
        # Plot
        plt.figure(figsize=(12,8))
        plt.subplot(211)
        plt.imshow([annotations], cmap='Blues')
        plt.axis('off')
        plt.subplot(212)
        plt.plot(angles, '-o', label=angle_type + ' raw angle')
        plt.legend()
        plt.tight_layout(h_pad=-4.9)
        plt.xlim(0, 50)
        plt.ylim(-0.3, 0.3)
        plt.show()
    
    return annotations, angles, peak_cnt
    

def get_perframe_features(df, angle_type, verbose=False):

    # Resample to common frame rate
    csv_frame_rate = (df.shape[0] - 1) / np.sum(np.diff(df[' timestamp']))
    if verbose:
        print(f'csv frame rate: {csv_frame_rate}')
    n_resampled_points = int(1 + (len(df[ANGLE_TYPE_PREFIX + angle_type]) - 1) * FRAME_RATE / csv_frame_rate)
    angles = scipy.signal.resample(df[ANGLE_TYPE_PREFIX + angle_type], num=n_resampled_points)

    # Take the first derivative of the signal
    diff_angles = np.diff(angles)

    # Apply median filter
    filt_diff_angles = scipy.signal.medfilt(diff_angles, kernel_size=FILTER_KERNEL)

    # Low-pass filter
#         fc = 3
#         b, a = scipy.signal.butter(N=2, Wn=2*fc/FRAME_RATE, btype='low', analog=False, output='ba')
#         filt_diff_angles = scipy.signal.filtfilt(b, a, diff_angles)

    # Get peak locations
    pos_peaks, _ = scipy.signal.find_peaks(filt_diff_angles)
    neg_peaks, _ = scipy.signal.find_peaks(-filt_diff_angles)

    # 2D array of features: # original samples x 2 (w and h)
    features = np.zeros((len(angles), 2))

    # Detect neighboring +- and -+
    i, j = 0, 0
    while i < len(pos_peaks) and j < len(neg_peaks):

        # Height difference of adjacent extremes + - or - +
        h = filt_diff_angles[pos_peaks[i]] - filt_diff_angles[neg_peaks[j]]

        if pos_peaks[i] == neg_peaks[j]:
            raise Exception('Positive and negative peaks coincide!')
        # + -
        elif pos_peaks[i] < neg_peaks[j]:
            # Search left for first negative peak neg_peaks[j-1]
            if j <= 0:
                left_end = 0
                if verbose: print('L End reached + -')
            else:
                left_end = neg_peaks[j-1]

            # Search right for first positive peak
            if i >= len(pos_peaks) - 1:
                right_end = len(filt_diff_angles) - 1
                if verbose: print('R End reached + -')
            else:
                right_end = pos_peaks[i+1]

            i += 1

        # - +
        else:
            # Search left for first positive peak
            if i <= 0:
                left_end = 0
                if verbose: print('L End reached - +')
            else:
                left_end = pos_peaks[i-1]

            # Search right for first negative peak
            if j >= len(neg_peaks) - 1:
                right_end = len(filt_diff_angles) - 1
                if verbose: print('R End reached - +')
            else:
                right_end = neg_peaks[j+1]

            j += 1

        # Peak width
        w = abs(right_end - left_end) + 1
        
        # Set features for this peak's range
        # +1 and +2 are shifts due to the fact that np.diff decrements the number of samples and at the end we want to have len(angles) samples
        features[left_end + 1:right_end + 2] = [w, h]

    # Replicate the first element to match the length of angles
    if len(features) > 0:
        features[0] = features[1]
    assert len(features) == len(angles)
        
    if verbose:
        print(features)
        # Plot
        plt.figure(figsize=(12,8))
        plt.plot(features[:, 0], '-', label='Peak width feature')
        plt.plot(features[:, 1], '-', label='Peak height feature')
        plt.legend()
        plt.show()
    
    return features, angles