import math
from collections import deque

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Set the accelerator if we have one available:
# NOTE: This is done so that different accelerators can be run (e.g. MPS on Apple Silicon)
device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device('cpu')

# features to exclude for pamap2:subject101.ankle.csv
EXCLUDE_FEATURES = [
    "seconds_since_start", "stamp", "yaw", "pitch", "roll",
     "altitude", "course", "speed","latitude","longitude",
    "horizontal_accuracy", "vertical_accuracy", "battery_state"
]

LABEL_COLUMN = "user_activity_label"

NUM_DETECTORS = 8

def load_and_window_data(csv_files, window_size, stride, num_detectors, existing_label_encoder = None):
    # TODO: Include time information?
    all_X, all_Y = [], []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file, skiprows=[1])
        df.drop(columns=[col for col in EXCLUDE_FEATURES if col in df.columns], inplace=True, errors='ignore')
        df.dropna(inplace=True)

        if LABEL_COLUMN in df.columns:
            y = df[LABEL_COLUMN].values
            X = df.drop(columns=[LABEL_COLUMN]).values.astype(np.float32)
            all_X.append(X)
            all_Y.append(y)

    X_all = np.vstack(all_X)
    Y_all = np.hstack(all_Y)

    print("Total frames:", len(X_all))
    print("Unique labels:", np.unique(Y_all))

    # Use existing label encoder if set, otherwise generate a new one with the labels from this dataset
    # NOTE: Should use this on training dataset, then pass the trained encoder to the other datasets
    label_encoder = existing_label_encoder
    if label_encoder is None:
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(Y_all)

    X_windows = []
    Y_windows = []

    for i in range(0, X_all.shape[0] - window_size, stride):
        x_win = X_all[i:i+window_size]
        y_win = Y_all[i:i+window_size]

        # Convert labels into activity segments:
        segments_for_window = frame_labels_to_segments(y_win, label_encoder)

        # Put segments in detection boxes and get the detections array:
        detections_for_window = segments_to_detections(segments_for_window, label_encoder, window_size, num_detectors)

        X_windows.append(x_win)
        Y_windows.append(detections_for_window)
    print(f"Generated {len(X_windows)} windows.")
    return np.array(X_windows), np.array(Y_windows), label_encoder

# TODO: Replace doing segments and then doing detections with just detections (combine this and next func)?
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
            segments.append((1.0, start, duration, class_id))  # Confidence = 1.0
            start = i
            current_label = y_window[i]
    end = len(y_window)
    duration = end - start
    class_id = le.transform([current_label])[0]
    segments.append((1.0, start, duration, class_id))
    return segments

def segments_to_detections(segments_for_window, le, window_size, num_detectors, background_label = 'Other'):
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
    # TODO: Fix the number of classes N to not include "Other" label as one of the slots

    (Note: If multiple segments have midpoints in the same cell, only the last of those segments will be included. This
    is something that would be fixed with different bounding boxes in each detector.)

    :param segments_for_window: list of (confidence, start frame index, frame duration, and class_id) segments
    :param le: the label encoder
    :param window_size: number of frames in the window
    :param num_detectors: number of detection cells to use
    :param background_label: label that is considered "background" (i.e. segments ignored) - defaults to Other
    :return: a (num_detectors x (3+N)) array of detection cells labels for the window
    """

    num_classes = len(le.classes_)

    detections = np.zeros((num_detectors, 3 + num_classes), dtype = np.float32)  # one row for each detection

    cell_width = window_size // num_detectors  # (round down to nearest integer)
    # TODO: Handle window size not an exact multiple of the number of detectors

    # Get index of background label to ignore if set:
    background_label_index: int | None = None
    if background_label is not None:
        background_label_index = le.transform([background_label])[0]

    # Now assign each segment to a detection cell:
    for seg_index, (confidence, start, duration, class_id) in enumerate(segments_for_window):
        if class_id == background_label_index:
            # Ignore this segment, as it's background label
            continue

        # Determine the midpoint of the segment:
        midpoint = start + duration // 2

        # Find the detection cell the midpoint is in:
        for i in range(num_detectors):
            cell_start = cell_width * i
            cell_end = cell_start + cell_width

            if cell_start <= midpoint < cell_end:
                # Segment midpoint belongs to the ith cell:
                p_c = 1.0  # confidence
                t_x = (midpoint - cell_start) / cell_width
                t_w = np.log(duration / cell_width)

                # Create the class section, with the class id marked as 1:
                c = [0] * num_classes
                c[class_id] = 1

                detections[i] = [p_c, t_x, t_w] + c

                break  # move to next segment

        # TODO: Fix this
        # # If we get this far, no detection box was found (maybe due to window_size / detection_size not being integer)?
        # raise RuntimeError(f'No detection box found for segment index {seg_index} of class ID {class_id}')

    return detections

# TODO: Function to go detection (p_c, t_x, t_w, and class value) back to segments (i.e. convert the relative pos and
#   width within the frame into actual segments). And probably do IoU, etc.


class YodaSensorDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_tensor = torch.tensor(self.X[idx]).transpose(0, 1)  # [features, time]
        y_tensor = torch.tensor(self.Y[idx])  # [num_detectors, detection_array]

        return x_tensor, y_tensor

class Yoda(nn.Module):
    def __init__(self, input_channels, num_classes, num_detectors):
        super().__init__()

        self.num_classes = num_classes
        self.num_detectors = num_detectors

        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_detectors * (3+num_classes))

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        logits = x.view(-1, self.num_detectors, 3 + self.num_classes)  # [batch, num_detectors, 3 + num_classes] - each item in 2nd dimension is a detection output

        # The logits are of the same form as our targets: batch_size x num_detectors x [p_c, t_x, t_w, c_1, c_2, ..., c_N]
        # Don't do any scaling or changing of the logits here - that is done later

        return logits

class YODALoss(nn.Module):
    """
    Implement the YODA loss as a composite of:
     - Confidence loss (compare the predicted to actual confidence scores)
     - Localization loss (compare predicted to actual segment sizes and locations)
     - Classification loss (compare predicted to actual class predictions)
    """

    def __init__(self, lambda_localization=5.0, lambda_noseg=0.5):
        """
        Initialize the loss
        TODO: should the default lambda values be 1 in this formulation (they were set based on the YOLOv1 version)
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

        # Localization loss:
        # Compute the loss separately for the segment center (t_x) values and the log-space segment width (t_w) values
        # Only include the cells where there is a target

        # For t_x, use BCE loss since the t_x value should be in [0, 1]
        localization_loss_x = self.bce_loss(pred[..., 1], target[..., 1]) * target_mask

        # For t_w, use MSE loss since the t_w value can be unbounded:
        localization_loss_w = self.mse_loss(pred[..., 2], target[..., 2]) * target_mask

        # Classification loss:
        # Compute the loss as the BCE loss over the prediction values, only for cells with a target segment.
        classification_loss = self.bce_loss(pred[..., 3:], target[..., 3:]) * target_mask.unsqueeze(-1)  # unsqueeze to match dims

        # Combine the losses using scaling factors, and normalize by the batch size (to get mean across batch):
        total_loss = (
            conf_loss_with_segment.sum() +
            self.lambda_noseg * conf_loss_without_segment.sum() +
            self.lambda_localization * localization_loss_x.sum() +
            self.lambda_localization * localization_loss_w.sum() +
            classification_loss.sum()
        ) / pred.shape[0]  # normalize by batch size

        return total_loss

def train(model, train_dataloader, val_dataloader, optimizer, criterion, epochs=10):
    print("Starting training...")

    for epoch in range(epochs):
        train_loss = 0
        model.train()
        for x_batch, y_batch in train_dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Training: Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")

        # Do validation loss:
        val_loss = 0
        model.eval()
        for x_batch, y_batch in val_dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            with torch.no_grad():
                preds = model(x_batch)

                loss = criterion(preds, y_batch)
                val_loss += loss.item()
        print(f"Validation: Epoch {epoch+1}/{epochs}, Loss: {val_loss:.4f}")


def evaluate_plot(model, dataset, dataset_name, window_size, stride, le, background_label = 'Other'):
    """
    Evaluate the model by running it on all of the input data in the dataset. Calculates the following metrics:
     - Accuracy of labels for all individual events
     - Segment-based (TBD)

    Also plots the ground-truth segments against the predicted segments for visual comparison.

    :param model: the model to evaluate
    :param dataset: the dataset to use as input and ground truth (note: not data loader, as we will create a non-
        shuffled loader for this purpose)
    :param dataset_name: the name of the dataset (i.e. train, validation, test)
    :param window_size: the size of the window used in the dataset
    :param stride: the stride of the windows used in the dataset
    :param le: label encoder (to get labels from indices)
    :param background_label: the background label (to treat as the no-activity label)
    """

    print(f"\nRunning evaluation for {dataset_name} data...")

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

    for window_index in range(all_preds.shape[0]):
        window_detections = all_preds[window_index]
        segments_for_window = construct_segments_from_detections(window_detections, le, window_size, 0.5, 0.0)

        # Convert the segment indices to overall indices using the window_index:
        for segment in segments_for_window:
            seg_start_in_window, seg_end_in_window, p_c, label_index, label = segment
            seg_start_in_sequence = window_index * window_size + seg_start_in_window
            seg_end_in_sequence = window_index * window_size + seg_end_in_window

            pred_segments.append((seg_start_in_sequence, seg_end_in_sequence, p_c, label_index, label))

    actual_segments = []

    for window_index in range(all_targets.shape[0]):
        window_detections = all_targets[window_index]
        segments_for_window = construct_segments_from_detections(window_detections, le, window_size, 0.5, 0.0)

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

    # Compare the segments by trying to find highest-IoU matching for each target segment:
    evaluate_segments_by_iou(pred_segments, actual_segments)

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

def construct_segments_from_detections(detections, le, window_size, min_p_c, allowed_iou):
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
    :param min_p_c: minimum p_c value to keep a detection (ignores detections with lower p_c)
    :param allowed_iou: for non-max-suppression, keep detections where the iou between the detections is this value or
        less. To disallow any overlap, use 0.0. If two detections overlap, the higher-p_c of the two is used for the
        overlap range, and the lower one(s) only include their parts that don't overlap.
    """

    # Convert detections into proposed segments based on window and detector sizes:
    num_detectors = detections.shape[0]
    detection_width = window_size // num_detectors

    proposed_segments = []

    for detector_index, detection in enumerate(detections):
        p_c = torch.sigmoid(detection[0]).item() # confidence, run logit through sigmoid to convert to [0, 1] range

        if p_c < min_p_c:
            # Not high enough confidence, skip
            continue

        # Calculate the start/end indices (relative to the window) of the detected segment:

        # Midpoint using size relative to detector (run through sigmoid to keep in [0,1]:
        detector_start = detector_index * detection_width
        segment_midpoint = detector_start + detection_width * torch.sigmoid(detection[1])

        # Find the segment length using an exponential of the t_w value, multiplied by the cell width
        segment_length = detection_width * torch.exp(detection[2])

        segment_start = max(int(segment_midpoint - segment_length / 2), 0)
        segment_end = min(int(segment_midpoint + segment_length / 2), window_size - 1)

        # Do sigmoid scaling on class probability logits to fit them in [0,1]:
        p_classes = torch.sigmoid(detection[3:])

        proposed_segments.append((segment_start, segment_end, p_c, p_classes))

    # Perform non-max-suppression on the proposed segments:
    # Sort the segments from highest to lowest p_c:
    remaining_proposed = deque(sorted(proposed_segments, key=lambda seg: seg[2], reverse=True))

    final_segments = []

    while len(remaining_proposed) > 0:
        # Get the remaining segment with highest confidence, and add it to final segments:
        highest_confidence_segment = remaining_proposed.popleft()
        seg_start, seg_end, p_c, p_classes = highest_confidence_segment

        # Find the class label with the highest probability:
        segment_class_index = np.argmax(p_classes).item()
        segment_class_name = le.inverse_transform([segment_class_index])[0]

        final_segments.append((seg_start, seg_end, p_c, segment_class_index, segment_class_name))

        # Go through the rest of the segments and discard any whose IOU is too much overlap with this one:
        segs_to_drop = []
        for i, segment in enumerate(remaining_proposed):
            iou_with_main_segment = iou(highest_confidence_segment[0:2], segment[0:2])

            if iou_with_main_segment > allowed_iou:
                segs_to_drop.append(i)

        # Rebuild the remaining proposed deque, excluding any segment to be dropped:
        remaining_proposed = deque(segment for i, segment in enumerate(remaining_proposed) if i not in segs_to_drop)

    return final_segments

def evaluate_segments_by_iou(pred_segments, target_segments, min_iou: float = 0.5):
    """
    Compare the predicted segments to the target (ground truth) segments using IoU.

    Will iterate through the segments, and try to find a predicted segment that matches each ground truth segment by
    using IoU. This is done by finding the predicted segment with the highest IoU to the target segment and having the
    same class index.

    :param pred_segments: the predicted segments as a list of (start_in_sequence, end_in_sequence, p_c, label_index, label)
    :param target_segments: the target segments as a list of (start_in_sequence, end_in_sequence, p_c, label_index, label)
    :param min_iou: predicted segment must have at least this much IoU with the target to be considered
    """

    # Store indices of predictions that have been matched ("used"):
    matched_pred_idxes: list[int] = list()

    # IoUs found for all target segments:
    max_iou_per_target: list[float] = list()

    # List of whether each target segment was matched:
    segments_matched: list[bool] = list()

    for target_segment in target_segments:
        target_start, target_end, _, target_label_index, _ = target_segment

        max_iou: float = -math.inf
        max_iou_idx: int | None = None

        for pred_idx, pred_segment in enumerate(pred_segments):
            if pred_idx in matched_pred_idxes:
                # Don't match a prediction to more than one target
                continue

            pred_start, pred_end, _, pred_label_index, _ = pred_segment

            if pred_label_index != target_label_index:
                # Labels don't match, so don't use this segment:
                continue

            iou_with_target = iou(target_segment, pred_segment)

            if iou_with_target < min_iou:
                # IoU too low, so skip:
                continue

            # Check if the segment has the highest IoU so far:
            if iou_with_target > max_iou:
                max_iou = iou_with_target
                max_iou_idx = pred_idx

        if max_iou_idx is not None:
            # We found a segment with high enough IoU, so use that:
            max_iou_per_target.append(max_iou)
            segments_matched.append(True)

            # Mark this predicted segment as matched:
            matched_pred_idxes.append(max_iou_idx)
        else:
            # No segment was found, so the IoU was zero:
            max_iou_per_target.append(0.0)
            segments_matched.append(False)

    total_matches = sum(segments_matched)
    total_target_segments = len(target_segments)
    total_predicted_segments = len(pred_segments)

    unmatched_targets = total_target_segments - total_matches
    unmatched_predicted_segments = total_predicted_segments - total_matches

    precision = total_matches / total_predicted_segments if total_predicted_segments > 0 else 0.0
    recall = total_matches / total_target_segments if total_target_segments > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall)

    best_iou = max(max_iou_per_target)
    mean_iou = sum(max_iou_per_target) / len(max_iou_per_target)

    print("\n=== Segment-Level Evaluation ===")
    print(f"Total ground-truth segments:   {total_target_segments}")
    print(f"Total predicted segments:      {total_predicted_segments}")
    print(f"Matched segments:              {total_matches}\n")
    print(f"Precision:                     {precision:.3f}")
    print(f"Recall:                        {recall:.3f}")
    print(f"F1 Score:                      {f1:.3f}")
    print(f"Best IoU:                      {best_iou:.3f}")
    print(f"Mean IoU:                      {mean_iou:.3f}")

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
    print("Loading training data")
    train_files = [
        'data/protocol/subject101.hand.csv',
        'data/protocol/subject102.hand.csv',
        'data/protocol/subject103.hand.csv',
        'data/protocol/subject104.hand.csv',
        'data/protocol/subject105.hand.csv'
    ]
    train_X, train_Y, label_encoder = load_and_window_data(train_files, window_size=8192, stride=8192, num_detectors=NUM_DETECTORS)

    train_dataset = YodaSensorDataset(train_X, train_Y)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    print("Loading validation data")
    val_files = [
        'data/protocol/subject106.hand.csv',
        'data/protocol/subject107.hand.csv'
    ]
    val_X, val_Y, _ = load_and_window_data(val_files, window_size=8192, stride=8192, num_detectors=NUM_DETECTORS, existing_label_encoder=label_encoder)
    val_dataset = YodaSensorDataset(val_X, val_Y)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print("Loading test data")
    test_files = [
        'data/protocol/subject108.hand.csv',
        'data/protocol/subject109.hand.csv'
    ]
    test_X, test_Y, _ = load_and_window_data(test_files, window_size=8192, stride=8192, num_detectors=NUM_DETECTORS, existing_label_encoder=label_encoder)
    test_dataset = YodaSensorDataset(test_X, test_Y)

    print("Training model")
    model = Yoda(input_channels=train_X.shape[2], num_classes=len(label_encoder.classes_), num_detectors=NUM_DETECTORS)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Move model and optimizer to accelerator:
    model.to(device)

    train(model, train_loader, val_loader, optimizer, YODALoss(), epochs=100)

    evaluate_plot(model, train_dataset, 'Training', window_size=8192, stride=8192, le=label_encoder)
    evaluate_plot(model, val_dataset, 'Validation', window_size=8192, stride=8192, le=label_encoder)
    evaluate_plot(model, test_dataset, 'Test', window_size=8192, stride=8192, le=label_encoder)
