import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,accuracy_score
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# features to exclude for pamap2
EXCLUDE_FEATURES = [
    "seconds_since_start", "stamp", "yaw", "pitch", "roll",
     "altitude", "course", "speed","latitude","longitude",
    "horizontal_accuracy", "vertical_accuracy", "battery_state"
]

LABEL_COLUMN = "user_activity_label"
BATCH_SIZE = 32
EPOCHS = 50
NUM_DETECTORS = 10
WINDOW_SIZE = 16384
STRIDE = 16384
ANCHOR_BOXES = torch.tensor([[0.5], [1.0], [2.0]]).to(device)
NUM_ANCHORS = len(ANCHOR_BOXES)
TARGET = 'Walking'

# Define similar activities for evaluation metric
SIMILAR_ACTIVITIES = {
    'Walking': ['NordicWalking', 'Running'],
    'Running': ['Walking', 'NordicWalking'],
    'NordicWalking': ['Walking', 'Running'],
    'Sitting': ['Lying', 'Standing'],
    'Standing': ['Sitting', 'Lying'],
    'Lying': ['Sitting', 'Standing'],
    'AscendingStairs': ['DescendingStairs'],
    'DescendingStairs': ['AscendingStairs']
}

def load_and_window_data(csv_files, window_size, stride, num_detectors, anchors):
    
    all_X, all_Y, all_labels = [], [], []

    for csv_path in csv_files:
        df = pd.read_csv(csv_path, skiprows=[1])
        df.drop(columns=[col for col in EXCLUDE_FEATURES if col in df.columns], inplace=True, errors="ignore")
        df.dropna(inplace=True)

        if LABEL_COLUMN in df.columns:
            y = df[LABEL_COLUMN].values
            X = df.drop(columns=[LABEL_COLUMN]).values.astype(np.float32)
            
            all_X.append(X)
            all_Y.append(y)
            all_labels.extend(y)

    X_all = np.vstack(all_X)
    Y_all = np.hstack(all_Y)
    X_mean = np.mean(X_all, axis=0)
    X_std = np.std(X_all, axis=0) + 1e-6
    X_all = (X_all - X_mean) / X_std
    
    print("Total frames:", len(X_all))
    print("Unique labels:", np.unique(Y_all))
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    
    X_windows = []
    Y_windows = []

    for i in range(0, len(X_all) - window_size+1, stride):
        x_win = X_all[i:i+window_size]
        y_win = Y_all[i:i+window_size]
        segments_for_window = frame_labels_to_segments(y_win, label_encoder)
        detections_for_window = segments_to_detections(segments_for_window, label_encoder, window_size, num_detectors, anchors)

        X_windows.append(x_win)
        Y_windows.append(detections_for_window)
    print(f"Generated {len(X_windows)} windows.")
    return np.array(X_windows), np.array(Y_windows), label_encoder


def frame_labels_to_segments(y_window, le):
    """
    Find a list of all activity segments in the window of labels. A segment is defined as a continuous set of frames
    (labels) with the same label.
    For each segment, the start frame, duration, and class ID (from the label encoder) are stored.

    Note that this includes "Other" (or other background labels) as their own segment.

    :param y_window: the set of labels for all frames in the window
    :param le: label encoder
    :return: a list of tuples of (confidence, start frame index, frame duration, and class_id) of all segments in the window
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
    detections = np.zeros((num_detectors, num_anchors, 4 + num_classes), dtype=np.float32)  # one row for each detection [num_detectors, num_anchors, 4 + num_classes]

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
                    detections[i, best_anchor_idx] = [p_c, t_x, t_w, 1.0] + c
                
                
                break  # move to next segment

        # TODO: Fix this
        # # If we get this far, no detection box was found (maybe due to window_size / detection_size not being integer)?
        # raise RuntimeError(f'No detection box found for segment index {seg_index} of class ID {class_id}')

    return detections


def load_and_window_single_segment(csv_files, target_activity, target_start_frame, target_duration, similar, nonA, le):
    all_X, all_Y = [], []
    offset = 0  # global frame index
    
    for csv_path in csv_files:
        df = pd.read_csv(csv_path, skiprows=[1])
        df.drop(columns=[col for col in EXCLUDE_FEATURES if col in df.columns],
                inplace=True, errors="ignore")
        df.dropna(inplace=True)
        
        if LABEL_COLUMN in df.columns:
            y = df[LABEL_COLUMN].values
            X = df.drop(columns=[LABEL_COLUMN]).values.astype(np.float32)

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
    for i in range(0, len(X_all) - WINDOW_SIZE + 1, WINDOW_SIZE):
        x_win = X_all[i:i+WINDOW_SIZE]
        y_win = Y_all[i:i+WINDOW_SIZE]

        segments_for_window = frame_labels_to_segments(y_win, le)
        detections_for_window = segments_to_detections(
            segments_for_window, le, WINDOW_SIZE, NUM_DETECTORS, ANCHOR_BOXES
        )

        X_windows.append(x_win)
        Y_windows.append(detections_for_window)

    return np.array(X_windows), np.array(Y_windows), le


def load_and_window_segment(train_files, target_activity, window_size,stride,  nonA):
    """
    Loads and windows data for a single segment of a target activity.
    Other segments of the same activity and similar activities are removed.
    All other dissimilar activities are re-labeled as nonA.
    """
    all_X, all_Y = [], []
    

    for csv_path in train_files:
        df = pd.read_csv(csv_path, skiprows=[1])
        df.drop(columns=[col for col in EXCLUDE_FEATURES if col in df.columns], inplace=True, errors="ignore")
        df.dropna(inplace=True)
        
        if LABEL_COLUMN in df.columns:
            y = df[LABEL_COLUMN].values
            X = df.drop(columns=[LABEL_COLUMN]).values.astype(np.float32)

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
        
        detections_for_window = segments_to_detections(segments_for_window, label_encoder, window_size, NUM_DETECTORS, ANCHOR_BOXES)

        X_windows.append(x_win)
        Y_windows.append(detections_for_window)
    print(f"Generated {len(X_windows)} windows.")
    return np.array(X_windows), np.array(Y_windows), label_encoder

def find_all_segments(csv_files):
    """
    Finds all segments (occurrences) of all activities in the dataset.
    all_segments: (activity_label, start_frame, duration).
    """
    
    all_segments = []
    all_Y = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path, skiprows=[1])
        df.drop(columns=[col for col in EXCLUDE_FEATURES if col in df.columns], inplace=True, errors="ignore")
        df.dropna(inplace=True)

        if LABEL_COLUMN in df.columns:
            y = df[LABEL_COLUMN].values
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



class YodaSensorDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_tensor = torch.tensor(self.X[idx]).transpose(0, 1)  # [features, time]
        y_tensor = torch.tensor(self.Y[idx])  # [num_detectors, num_anchors, detection_array]

        return x_tensor, y_tensor

class Yoda(nn.Module):
    def __init__(self, input_channels, num_classes, num_detectors, num_anchors,cpc_pretrained_path):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_detectors = num_detectors
        # The output size for each anchor: 1 (confidence) + 1 (center_x) + 1 (width) + 1 (objectness) + num_classes
        self.num_outputs_per_anchor = 4 + self.num_classes
        if cpc_pretrained_path and os.path.exists(cpc_pretrained_path):
            print(f"Loading CPC pretrained encoder...")
            from cpc import EnhancedEncoder1D
            cpc_encoder = EnhancedEncoder1D(in_channels=input_channels)
            cpc_state = torch.load(cpc_pretrained_path, map_location=device)
            cpc_encoder.load_state_dict(cpc_state, strict=False)
            self.feature_extractor = cpc_encoder.net
            print("CPC encoder loaded.")
        else:
            self.feature_extractor = nn.Sequential(
                nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
            )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_detectors * num_anchors * self.num_outputs_per_anchor)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        logits = x.view(-1, self.num_detectors, self.num_anchors, self.num_outputs_per_anchor)
        return logits


class YODALoss(nn.Module):
    """
    Implement the YODA loss as a composite of:
     - Localization loss (compare predicted to actual box sizes and locations)
     - Confidence loss (compare the predicted to actual confidence scores)
     - Classification loss (compare predicted to actual class predictions)
    """

    def __init__(self, lambda_localization=5.0, lambda_noseg=0.5):
        """
        Initialize the loss
        :param lambda_localization: scalar for the localization component of loss
        :param lambda_noseg: scalar for the confidence loss when no segment is present in that cell
        """
        super().__init__()
        self.lambda_localization = lambda_localization
        self.lambda_noseg = lambda_noseg
        # Set up losses (use reduction = none so parts of loss can be summed then averaged over entire batch size
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')  # use with logits as the output of the model is not scaled by sigmoid
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        """
        Compute the loss of the predicted against actual detections.

        Each should be a tensor of dimensions (batch_size, # boxes, 3 + num_classes)
        Where the last dimension consists of:
        [p_c, b_x, b_w, class_probs]
        [p_c, t_x, t_w, c_1, c_2, ..., c_N]
        and none of the values have been scaled (i.e. raw model output, not passed through sigmoid/etc)
        """

        # Determine which cells have a segment target assigned (ground-truth p_c > 0):
        target_mask = target[..., 0] > 0
        
        # Objectness (confidence) loss:
        # Compute the BCE loss separately for the cases where there is a segment in the target vs those where there is
        # not. This is comparing the p_c of the prediction (probability of a segment detected in the cell) against the
        # ground_truth p_c (which is either 1 or 0):
        conf_loss_with_segment = self.bce_loss(pred[..., 0], target[..., 0]) * target_mask  # mult by target mask to only include ones with segment targets
        conf_loss_without_segment = self.bce_loss(pred[..., 0], target[..., 0]) * ~target_mask
        confidence_loss = (conf_loss_with_segment.sum() * self.lambda_localization) + (self.lambda_noseg * conf_loss_without_segment.sum())
        #conf_loss_with_segment.sum() + self.lambda_noseg * conf_loss_without_segment.sum()
        # Localization loss:
        # Compute the loss separately for the segment center (t_x) values and the log-space segment width (t_w) values
        # Only include the cells where there is a target

        # For t_x, use BCE loss since the t_x value should be in [0, 1]
        localization_loss_x = self.bce_loss(pred[..., 1], target[..., 1]) * target_mask

        # For t_w, use MSE loss since the t_w value can be unbounded:
        localization_loss_w = self.mse_loss(pred[..., 2], target[..., 2]) * target_mask
        localization_loss_a = self.mse_loss(pred[..., 3], target[..., 2]) * target_mask
        # Classification loss:
        # Compute the loss as the BCE loss over the prediction values, only for cells with a target segment.
        classification_loss = self.bce_loss(pred[..., 4:], target[..., 4:]) * target_mask.unsqueeze(-1)  # unsqueeze to match dims

        # Combine the losses using scaling factors, and normalize by the batch size (to get mean across batch):
        num_boxes = pred.shape[0] * pred.shape[1]
        total_loss = (
            confidence_loss +
            self.lambda_localization * localization_loss_x.sum() +
            self.lambda_localization * localization_loss_w.sum() +
            self.lambda_localization * localization_loss_a.sum()+
            classification_loss.sum()
        ) / num_boxes

        return total_loss

def compute_iou(gt_start, gt_end, pred_start, pred_end):
    inter_start = max(gt_start, pred_start)
    inter_end = min(gt_end, pred_end)
    inter = max(0, inter_end - inter_start)
    union = (gt_end - gt_start) + (pred_end - pred_start) - inter
    iou_result = inter / (union + 1e-6)
    
    return iou_result


def reconstruct_gt_segments_from_tensor(target_tensor, le, window_size, anchors):
    """Reconstructs ground truth segments from the target tensor without applying transformations."""
    num_detectors, num_anchors, _ = target_tensor.shape
    cell_width = window_size / num_detectors
    gt_segments = []

    # Find where confidence (p_c) is 1
    responsible_anchors = torch.where(target_tensor[..., 0] == 1)
    
    for i, j in zip(*responsible_anchors):
        i, j = i.item(), j.item() # cell index, anchor index
        
        data = target_tensor[i, j]
        t_x = data[1].item()
        t_w = data[2].item()
        
        # Decode coordinates from ground truth values
        midpoint = (i + t_x) * cell_width
        # Reverse the log-space transformation for width
        duration = torch.exp(torch.tensor(t_w)) * anchors[j].item() * cell_width
        
        start = max(0, int((midpoint - duration )/ 2))
        end = min(window_size - 1, int((midpoint + duration) / 2))
        
        class_id = torch.argmax(data[4:]).item()
        class_label = le.inverse_transform([class_id])[0]
        
        gt_segments.append((start, end, 1.0, class_id, class_label))
        
    return gt_segments

def train(model, dataloader, optimizer, criterion, epochs=10):
    print("Starting training...")

    for epoch in range(epochs):
        train_loss = 0
        model.train()
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Training: Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")
def evaluate_segments_by_iou(pred_segments, target_segments, background_label_index, min_iou: float = 0.5, onset_tolerance: int = 100):
    """
    Compare predicted segments to ground truth using IoU, and also compute detection latency and onset alignment.

    :param pred_segments: list of (start_in_sequence, end_in_sequence, p_c, label_index, label)
    :param target_segments: list of (start_in_sequence, end_in_sequence, p_c, label_index, label)
    :param background_label_index: index used for background class
    :param min_iou: IoU threshold for match
    :param onset_tolerance: number of frames allowed for onset alignment
    """

    matched_pred_idxes: list[int] = list()
    max_iou_per_target: list[float] = list()
    segments_matched: list[bool] = list()
    latencies: list[float] = list()
    onset_correct: list[bool] = list()

    for target_segment in target_segments:
        target_start, target_end, target_pc, target_label_index, target_label = target_segment
        if target_label_index == background_label_index:
            continue

        max_iou: float = -math.inf
        max_iou_idx: int | None = None

        for pred_idx, pred_segment in enumerate(pred_segments):
            if pred_idx in matched_pred_idxes:
                continue

            pred_start, pred_end, pred_pc, pred_label_index, pred_label = pred_segment
            if pred_label_index != target_label_index:
                continue

            iou_with_target = iou(target_segment, pred_segment)
            if iou_with_target < min_iou:
                continue

            if iou_with_target > max_iou:
                max_iou = iou_with_target
                max_iou_idx = pred_idx

        if max_iou_idx is not None:
            max_iou_per_target.append(max_iou)
            segments_matched.append(True)
            matched_pred_idxes.append(max_iou_idx)

            # Compute latency and onset correctness
            pred_start, pred_end, _, _, _ = pred_segments[max_iou_idx]
            latency = max(0, pred_start - target_start)
            latencies.append(latency)
            onset_correct.append(abs(pred_start - target_start) <= onset_tolerance)

        else:
            max_iou_per_target.append(0.0)
            segments_matched.append(False)

    total_matches = sum(segments_matched)
    total_target_segments = len(target_segments)
    total_predicted_segments = len(pred_segments)

    precision = total_matches / total_predicted_segments if total_predicted_segments > 0 else 0.0
    recall = total_matches / total_target_segments if total_target_segments > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    best_iou = max(max_iou_per_target) if max_iou_per_target else 0.0
    mean_iou = sum(max_iou_per_target) / len(max_iou_per_target) if max_iou_per_target else 0.0

    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    onset_accuracy = sum(onset_correct) / len(onset_correct) if onset_correct else 0.0
    mean_latency*=0.02
    onset_accuracy*=0.02
    onset_tolerance*=0.02
    print("\n=== Segment-Level Evaluation ===")
    print(f"Total ground-truth segments:   {total_target_segments}")
    print(f"Total predicted segments:      {total_predicted_segments}")
    print(f"Matched segments:              {total_matches}\n")
    print(f"Precision:                     {precision:.2f}")
    print(f"Recall:                        {recall:.2f}")
    print(f"F1 Score:                      {f1:.2f}")
    print(f"Best IoU:                      {best_iou:.2f}")
    print(f"Mean IoU:                      {mean_iou:.2f}")
    print(f"Mean Detection Latency:        {mean_latency:.2f} seconds")
    print(f"Onset Accuracy (±{onset_tolerance}s):    {onset_accuracy:.2f}")
    
def evaluate_plot(model, dataset, dataset_name, window_size, stride, le,anchors, background_label = 'Other'):
    """
    Evaluate the model by running it on all of the input data in the data loader, then construct labels and plot
    them against the ground truth data.

    :param model: the model to evaluate
    :param dataset: the dataset to use as input and ground truth (note: not data loader, as we will create a non-
        shuffled loader for this purpose)
    :param window_size: the size of the window used in the dataset
    :param stride: the stride of the windows used in the dataset
    :param le: label encoder (to get labels from indices)
    """

    if stride != window_size:
        raise RuntimeError("Currently can only handle stride and window size being equal (no overlap)")

    model.eval()

    eval_dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    num_windows = len(dataset)
    total_num_events = num_windows * window_size

    background_label_index = le.transform([background_label])[0]
    all_preds = []
    all_targets = []

    for x_batch, y_batch in eval_dataloader:
        x_batch = x_batch.to(device)

        with torch.no_grad():
            preds = model(x_batch)

            all_preds.append(preds.to('cpu'))
            all_targets.append(y_batch)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    pred_segments = []

    for window_index, window_detections in enumerate(all_preds):
        segments_for_window = construct_segments_from_detections(window_detections, le, window_size, anchors.cpu(), 0.5, 0.5)

        # Convert the segment indices to overall indices using the window_index:
        for segment in segments_for_window:
            seg_start_in_window, seg_end_in_window, p_c, label_index, label = segment
            seg_start_in_sequence = window_index * window_size + seg_start_in_window
            seg_end_in_sequence = window_index * window_size + seg_end_in_window

            pred_segments.append((seg_start_in_sequence, seg_end_in_sequence, p_c, label_index, label))

    actual_segments = []

    for window_index, window_detections in enumerate(all_targets):
        segments_for_window = reconstruct_gt_segments_from_tensor(window_detections, le, window_size, anchors.cpu())
        
        # Convert the segment indices to overall indices using the window_index:
        for segment in segments_for_window:
            seg_start_in_window, seg_end_in_window, p_c, label_index, label = segment
            seg_start_in_sequence = window_index * window_size + seg_start_in_window
            seg_end_in_sequence = window_index * window_size + seg_end_in_window

            actual_segments.append((seg_start_in_sequence, seg_end_in_sequence, p_c, label_index, label))
            

    # Find individual event labels from the segments and calculate accuracy:
    pred_labels = segments_to_event_labels(pred_segments, total_num_events, background_label_index)
    actual_labels = segments_to_event_labels(actual_segments, total_num_events, background_label_index)

    print(classification_report(
        actual_labels,
        pred_labels,
        labels=range(len(le.classes_)),
        target_names=le.classes_
    ))
    print(accuracy_score(actual_labels,pred_labels))
    # Compare the segments by trying to find highest-IoU matching for each target segment:
    evaluate_segments_by_iou(pred_segments, actual_segments,background_label_index)

    plot_segments(actual_segments, pred_segments, num_windows, window_size, le, dataset_name)
def segments_to_event_labels(segments: list[tuple[int, int, float, int, str]], total_num_events: int, background_label_index: int) -> np.ndarray:
    """
    Converts a list of segments into per-event labels (i.e. one label per each input sensor instance).
    The input should be a list of segments, each a tuple of (start_index, end_index, p_c, label_integer, label_str).
    Returns a 1D array of integers, where the value at each index is the integer corresponding to the label for that
    event. (With the background_label_index applying where there is no label)
    """

    labels = np.full(total_num_events, background_label_index, dtype=int)

    for segment in segments:
        start_index, end_index, p_c, label_integer, label_str = segment

        labels[start_index:end_index+1] = label_integer

    return labels

def plot_segments(actual_segments, pred_segments, num_windows, window_size, le, dataset_mame):
    """Plot the actual vs predicted segments."""

    total_length = num_windows * window_size

    classes = list(le.classes_)

    cmap = plt.get_cmap('tab20', len(classes))

    fig, ax = plt.subplots(figsize=(20, 3), constrained_layout=True)

    # Plot the actual segments:
    for segment in actual_segments:
        seg_start, seg_end, p_c, label_index, label = segment

        ax.barh(0, left=seg_start, width=seg_end-seg_start, height=0.5, color=cmap(label_index))

    # Plot the predicted segments:
    for segment in pred_segments:
        seg_start, seg_end, p_c, label_index, label = segment

        ax.barh(-0.5, left=seg_start, width=seg_end-seg_start, height=0.5, color=cmap(label_index))

    # # Plot window dividers:
    # for i in range(total_length // window_size):
    #     ax.axvline(i * window_size, color='k')

    # Drop a separation:
    ax.axhline(-0.25, color='black')

    # Set axis limits:
    ax.set_xlim(0, total_length)
    ax.set_ylim(-0.75, 0.25)
    ax.set_yticks([])
    ax.set_xlabel('Frames Since Start')
    ax.set_title(f'{dataset_mame} Activity Segments')

    # Create legend:
    legend_patches = [mpatches.Patch(color=cmap(label_index), label=label) for label_index, label in enumerate(sorted(classes))]
    ax.legend(handles=legend_patches, loc='upper center', ncol=len(classes)/3 + 1, bbox_to_anchor=(0.5, -0.25))

    # plt.subplots_adjust(bottom=0.65)
    plt.tight_layout()
    plt.show()

def reconstruct_gt_segments_from_tensor(target_tensor, le, window_size, anchors):
    """Reconstructs ground truth segments from the target tensor without applying transformations."""
    num_detectors, num_anchors, _ = target_tensor.shape
    cell_width = window_size / num_detectors
    gt_segments = []

    responsible_anchors = torch.where(target_tensor[..., 0] >0)
    
    for i, j in zip(*responsible_anchors):
        i, j = i.item(), j.item()
        
        data = target_tensor[i, j]
        t_x = data[1].item()
        t_w = data[2].item()
        
        # Decode coordinates from ground truth values
        midpoint = (i + t_x) * cell_width
        # Reverse the log-space transformation for width
        duration = torch.exp(torch.tensor(t_w)) * anchors[j].item() * cell_width
        
        start = max(0, int(midpoint - duration / 2))
        end = min(window_size - 1, int(midpoint + duration / 2))
        
        class_id = torch.argmax(data[4:]).item()
        class_label = le.inverse_transform([class_id])[0]
        
        gt_segments.append((start, end, 1.0, class_id, class_label))
        
    return gt_segments

def train(model, dataloader, optimizer, criterion, epochs=10):
    print("Starting training...")

    for epoch in range(epochs):
        train_loss = 0
        model.train()
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Training: Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")
def construct_segments_from_detections(detections, le, window_size,anchors, conf_threshold=0.5, nms_iou_threshold=0.5):
    """
    Reconstruct label segments in a window based on the detection cell predictions (or ground truths), along with the
    window size.

    Will apply functions to the raw detection values to convert them into the event space (i.e. convert t_w from
    log-space into event space, etc)

    Returns a list of segment tuples of (start_index, end_index, p_c, label_index, label_str).

    :param detections: list of detections for the window as a #detectors x (3 + #classes) array (each row is a detector
        output of [p_c, t_x, t_w, c_1, ..., c_N]. These should not be scaled at all yet - that is done in this function
    :param le: label encoder (to get the label names)
    :param window_size: the number of frames in the window
    :param anchors: scaled durations of activities
    :param min_p_c: minimum p_c value to keep a detection (ignores detections with lower p_c)
    :param allowed_iou: for non-max-suppression, keep detections where the iou between the detections is this value or
        less. To disallow any overlap, use 0.0. If two detections overlap, the higher-p_c of the two is used for the
        overlap range, and the lower one(s) only include their parts that don't overlap.
    """

    num_detectors, num_anchors, _ = detections.shape
    cell_width = window_size / num_detectors
    
    proposed_segments = []

    for i in range(num_detectors):
        for j in range(num_anchors):
            detection = detections[i, j, :]
            
            p_c = torch.sigmoid(detection[0]).item()
            if p_c < conf_threshold:
                continue
            b_x = torch.sigmoid(detection[1]).item()
            b_w = torch.exp(detection[2]).item() * anchors[j].item()
            
            midpoint = (i + b_x) * cell_width
            duration = b_w * cell_width
            
            start = max(0, int(midpoint - duration / 2))
            end = min(window_size - 1, int(midpoint + duration / 2))
            
            class_probs = torch.sigmoid(detection[4:])
            class_id = torch.argmax(class_probs).item()
            class_label = le.inverse_transform([class_id])[0]
            
            proposed_segments.append((start, end, p_c, class_id, class_label))

    # Perform Non-Maximum Suppression (NMS)
    proposed_segments.sort(key=lambda x: x[2], reverse=True)
    final_segments = []

    while proposed_segments:
        current_segment = proposed_segments.pop(0)
        final_segments.append(current_segment)
        remaining_segments = []
        for seg in proposed_segments:
            if seg[3] != current_segment[3]:
                remaining_segments.append(seg)
                continue
            
            iou_val = iou(current_segment[:2], seg[:2])
            if iou_val < nms_iou_threshold:
                remaining_segments.append(seg)
        proposed_segments = remaining_segments
        
    return final_segments
def iou(segment1, segment2):
    """Compute the iou between the two segments, where each segment is a (start, end) tuple."""

    intersection_start = max(segment1[0], segment2[0])
    intersection_end = min(segment1[1], segment2[1])
    intersection = max(0, intersection_end - intersection_start)

    segment1_length = segment1[1] - segment1[0]
    segment2_length = segment2[1] - segment2[0]

    union = segment1_length + segment2_length - intersection
    epsilon = 1e-6  # avoid divide by zero

    return intersection / (union + epsilon)


if __name__ == "__main__":
    train_files = [
        'subject101.hand.csv',
        'subject102.hand.csv',
        'subject103.hand.csv',
        'subject104.hand.csv',
        'subject105.hand.csv',
        'subject106.hand.csv'
    ]
    test_files = [
        'subject107.hand.csv',
        'subject108.hand.csv',
        'subject109.hand.csv'
    ]
    cpc_pretrained_path = "cpc_pretrained.pt"
    #train_X, train_Y, label_encoder = load_and_window_data(train_files, window_size=WINDOW_SIZE, stride=STRIDE, num_detectors=NUM_DETECTORS,anchors=ANCHOR_BOXES)
    train_X, train_Y, label_encoder = load_and_window_segment(train_files,TARGET,WINDOW_SIZE,STRIDE,'Other')
    train_dataset = YodaSensorDataset(train_X, train_Y)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    #test_X, test_Y, _ = load_and_window_data(test_files, window_size=WINDOW_SIZE, stride=STRIDE, num_detectors=NUM_DETECTORS, existing_label_encoder=label_encoder)
    #test_dataset = YodaSensorDataset(test_X, test_Y)
    num_classes = len(label_encoder.classes_)
    model = Yoda(input_channels=train_X.shape[2], num_classes=num_classes, num_detectors=NUM_DETECTORS, num_anchors=NUM_ANCHORS,cpc_pretrained_path=cpc_pretrained_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.to(device)

    train(model, train_loader, optimizer, YODALoss(), epochs=EPOCHS)
    evaluate_plot(model, train_dataset, 'Training', window_size=WINDOW_SIZE, stride=STRIDE, le=label_encoder,anchors=ANCHOR_BOXES)
    all_test_segments = find_all_segments(test_files)

    print("\n--- Evaluating each segment in the test set ---")
    for activity, start, duration in all_test_segments:
        # Ignore "Other" activities
        if activity != TARGET:
            continue
            
        similar_activities = SIMILAR_ACTIVITIES.get(activity, [])
        non_A = 'Other'
        temp_X, temp_Y, temp_le = load_and_window_single_segment(
            test_files, 
            target_activity=activity, 
            target_start_frame=start, 
            target_duration=duration,
            similar =similar_activities,
            nonA = non_A,
            le=label_encoder
        )
        if temp_X is not None:
            temp_dataset = YodaSensorDataset(temp_X, temp_Y)
            print(f"\nEvaluating segment: Activity={activity}, Start={start}, Duration={duration}")
            evaluate_plot(
                model, 
                temp_dataset, 
                f'Testing - {activity} Segment ({start})', 
                window_size=duration, 
                stride=duration, 
                le=label_encoder, 
                anchors=ANCHOR_BOXES
            )

    