import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder


def frame_labels_to_segments(y_window, le):
    """
    Find a list of all activity segments in the window of labels. A segment is defined as a continuous set of frames
    (labels) with the same label.
    For each segment, the start frame, duration, and class ID (from the label encoder) are stored.

    Note that this includes "Other" (or other background labels) as their own segment.

    :param y_window: the set of labels for all frames in the window
    :param le: label encoder
    :return: a list of tuples of (start frame index, frame duration, and class_id) of all segments in the window
    """
    segments = []
    current_label = y_window[0]
    start = 0
    for i in range(1, len(y_window)):
        if y_window[i] != current_label:
            end = i
            duration = end - start
            class_id = le.transform([current_label])[0]
            segments.append((start, duration, class_id))
            start = i
            current_label = y_window[i]
    end = len(y_window)
    duration = end - start
    class_id = le.transform([current_label])[0]
    segments.append((start, duration, class_id))
    return segments


def segments_to_detections(segments_for_window, le, window_size, num_detectors, anchors, background_label = 'Other'):
    """
    Convert list of activity segments into detection cell outputs for the window.

    Divides the window into specified number of detection cells. Each segment is placed within the detection cell which
    contains the segment's midpoint. For that detection cell, the target vector will consist of values:
    [p_c, t_x, t_w, c_1, c_2, ..., c_N] where
     - p_c is the confidence of there being a segment in that cell (in this case, 1.0)
     - t_x is the center point location of the segment (from the start of the detection cell), as a fraction of the
        cell's total width [0.0-1.0]
     - t_w is the log-space width of the segment, defined as ln(segment_width / cell_width)
        - this is used as it is more stable than necessarily using the raw width ratio - large and small widths will
          be closer in log-sapce
     - c_1, c_2, ..., c_N are one-hot-encoded class label (from the label encoder) - the position corresponding to the
        class label of the segment is set to 1, all others to 0
        N is the number of class labels stored in the label encoder (may contain the background label class)

    These output vectors are combined into a (num_detectors x (3+N)) array, with each row corresponding to the ith
    detection cell.

    If background_label is set, that label is considered background, and so its segments are not assigned to detectors.

    (Note: If multiple segments have midpoints in the same cell, only the last of those segments will be included. This
    is something that would be fixed with different bounding boxes in each detector.)

    :param segments_for_window: list of (confidence, start frame index, frame duration, and class_id) segments
    :param le: the label encoder
    :param window_size: number of frames in the window
    :param num_detectors: number of detection cells to use
    :param anchors: scaled durations of activities
    :param background_label: label that is considered "background" (i.e. segments ignored) - defaults to Other
    :return: a (num_detectors x (3+N)) array of detection cells labels for the window
    """

    num_classes = len(le.classes_)
    num_anchors = len(anchors)
    detections = np.zeros((num_detectors, num_anchors, 3 + num_classes), dtype=np.float32)

    cell_width = window_size // num_detectors  # (round down to nearest integer)
    # TODO: Handle window size not an exact multiple of the number of detectors

    # Get index of background label to ignore if set:
    background_label_index: int | None = None
    if background_label is not None and background_label in le.classes_:
        background_label_index = le.transform([background_label])[0]

    for start, duration, class_id in segments_for_window:
        if class_id == background_label_index:
            # Ignore this segment, as it's background label
            continue

        midpoint = start + duration // 2

        # Find the detection cell the midpoint is in:
        for i in range(num_detectors):
            cell_start = cell_width * i
            cell_end = cell_start + cell_width

            if cell_start <= midpoint < cell_end:
                segment_width_ratio = duration / cell_width
                intersection = torch.min(torch.tensor(segment_width_ratio), anchors.squeeze())
                union = torch.tensor(segment_width_ratio) + anchors.squeeze() - intersection
                ious = intersection / (union + 1e-6)
                best_anchor_idx = torch.argmax(ious).item()
                p_c = 1.0
                
                t_x = (midpoint - cell_start) / cell_width
                t_w = np.log(duration / (anchors[best_anchor_idx].item() * cell_width) + 1e-16)

                c = [0] * num_classes
                c[class_id] = 1
                # Check if an anchor in this cell is already assigned
                if detections[i, best_anchor_idx, 0] == 0:
                    detections[i, best_anchor_idx] = [p_c, t_x, t_w] + c
                
                
                break

    return detections

def load_and_window_single_segment(csv_files, target_activity, target_start_frame, target_duration, similar, nonA, le, exclude_features, label_column, window_size, num_detection, anchor):
    """
    Loads and windows data for a single segment of a target activity.
    Other segments of the same activity and similar activities are removed.
    All other dissimilar activities are re-labeled as nonA.
    """
    all_X, all_Y = [], []
    offset = 0  # global frame index
    
    for csv_path in csv_files:
        df = pd.read_csv(csv_path, skiprows=[1])
        df.drop(columns=[col for col in exclude_features if col in df.columns], inplace=True, errors="ignore")
        df[label_column].fillna("Other", inplace=True)
        df.dropna(inplace=True)
        
        if label_column in df.columns:
            y = df[label_column].values
            X = df.drop(columns=[label_column]).values.astype(np.float32)

            for i in range(len(y)):
                global_i = i + offset
                label = y[i]
                
                is_target_segment = (
                    label == target_activity
                    and target_start_frame <= global_i < target_start_frame + target_duration
                )
                is_similar_activity = (label in similar)
                is_other_same_activity = (label == target_activity and not is_target_segment)

                if is_target_segment:
                    all_Y.append(label)
                    all_X.append(X[i])
                elif is_similar_activity or is_other_same_activity:
                    continue
                else:
                    all_Y.append(nonA)
                    all_X.append(X[i])
        
        offset += len(y)

    if len(all_Y) == 0:
        return None, None, None

    X_all = np.vstack(all_X)
    Y_all = np.hstack(all_Y)
    X_mean = np.mean(X_all, axis=0)
    X_std = np.std(X_all, axis=0) + 1e-6
    X_all = (X_all - X_mean) / X_std

    X_windows, Y_windows = [], []
    for i in range(0, len(X_all) - window_size + 1, window_size):
        x_win = X_all[i:i+window_size]
        y_win = Y_all[i:i+window_size]

        segments_for_window = frame_labels_to_segments(y_win, le)
        detections_for_window = segments_to_detections(
            segments_for_window, le, window_size, num_detection, anchor
        )

        X_windows.append(x_win)
        Y_windows.append(detections_for_window)

    return np.array(X_windows), np.array(Y_windows), le


def load_and_window_segment(train_files, target_activity, window_size,stride,  nonA, exclude_features, label_column, num_detection, anchor):
    """
    Loads and windows data for the target activity.
    All other activities are re-labeled as nonA.
    """
    all_X, all_Y = [], []
    

    for csv_path in train_files:
        chunk_size = 100000  # adjust as needed
        chunks = []
        for chunk in pd.read_csv(csv_path, skiprows=[1], chunksize=chunk_size, low_memory=False):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        df.drop(columns=[col for col in exclude_features if col in df.columns], inplace=True, errors="ignore")
        df[label_column].fillna("Other", inplace=True)
        df.dropna(inplace=True)
        
        if label_column in df.columns:
            y = df[label_column].values
            X = df.drop(columns=[label_column]).values.astype(np.float32)

            current_y = []
            current_X = []
            for i in range(len(y)):
                label = y[i]
                
                is_target_segment = (label == target_activity )
                
                if is_target_segment:
                    current_y.append(label)
                    current_X.append(X[i])
                else:
                    # Re-label as nonA
                    current_y.append(nonA)
                    current_X.append(X[i])

            if len(current_y) > 0:
                y_filtered = np.array(current_y)
                X_filtered = np.array(current_X)
                

                if len(X_filtered) > 0:
                    all_X.extend(X_filtered)
                    all_Y.extend(y_filtered)
    
    X_all = np.vstack(all_X)
    Y_all = np.hstack(all_Y)
    X_mean = np.mean(X_all, axis=0)
    X_std = np.std(X_all, axis=0) + 1e-6
    X_all = (X_all - X_mean) / X_std
    
    print("Total frames:", len(X_all))
    print("Unique labels:", np.unique(Y_all))
    label_encoder = LabelEncoder()
    label_encoder.fit(Y_all)
    
    X_windows = []
    Y_windows = []

    for i in range(0, len(X_all) - window_size+1, stride):
        x_win = X_all[i:i+window_size]
        y_win = Y_all[i:i+window_size]
        segments_for_window = frame_labels_to_segments(y_win, label_encoder)
        
        detections_for_window = segments_to_detections(segments_for_window, label_encoder, window_size, num_detection, anchor)

        X_windows.append(x_win)
        Y_windows.append(detections_for_window)
    print(f"Generated {len(X_windows)} windows.")
    return np.array(X_windows), np.array(Y_windows), label_encoder

def find_all_segments(csv_files, exclude_feature, label_column):
    """
    Finds all segments (occurrences) of all activities in the dataset.
    all_segments: (activity_label, start_frame, duration).
    """
    
    all_segments = []
    all_Y = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path, skiprows=[1])
        df.drop(columns=[col for col in exclude_feature if col in df.columns], inplace=True, errors="ignore")
        df[label_column].fillna("Other", inplace=True)
        df.dropna(inplace=True)

        if label_column in df.columns:
            y = df[label_column].values
            all_Y.extend(y)
    if len(all_Y) > 0:
                current_label = all_Y[0]
                start = 0
                for i in range(1, len(all_Y)):
                    if all_Y[i] != current_label:
                        end = i
                        duration = end - start
                        all_segments.append((current_label, start, duration))
                        start = i
                        current_label = all_Y[i]

                end = len(all_Y)
                duration = end - start
                all_segments.append((current_label, start, duration))

            
    return all_segments
