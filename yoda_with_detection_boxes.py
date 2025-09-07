from collections import deque
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# features to exclude for pamap2:subject101.ankle.csv
EXCLUDE_FEATURES = [
    "seconds_since_start", "stamp", "yaw", "pitch", "roll",
     "altitude", "course", "speed","latitude","longitude",
    "horizontal_accuracy", "vertical_accuracy", "battery_state"
]

LABEL_COLUMN = "user_activity_label"
BATCH_SIZE = 32
EPOCHS = 50
MODEL_PATH = "yoda_model.pt"
ENCODER_PATH = "label_encoder.pkl"
NUM_DETECTORS = 10
WINDOW_SIZE = 81920
STRIDE = 81920
def load_and_window_data(csv_files, window_size, stride, num_detectors):
    
    # TODO: Include time information?
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
    For each segment, the center frame, duration, and class ID (from the label encoder) are stored.

    Note that this includes "Other" (or other background labels) as their own segment.

    :param y_window: the set of labels for all frames in the window
    :param le: label encoder
    :return: a list of tuples of (start, end, class_id) of all segments in the window
    """
    segments = []
    if len(y_window) == 0:
        return segments
    current_label = y_window[0]
    start = 0
    for i in range(1, len(y_window)):
        if y_window[i] != current_label:
            class_id = le.transform([current_label])[0]
            segments.append((start, i-1, class_id))
            start = i
            current_label = y_window[i]

    end = len(y_window)
    class_id = le.transform([current_label])[0]
    segments.append((start, end, class_id))
    return segments


def segments_to_detections(segments_for_window, le, window_size, num_detectors, background_label="Other"):
    num_classes = len(le.classes_)
    detections = np.zeros((num_detectors, 3 + num_classes), dtype=np.float32)
    box_width = window_size / num_detectors
    bg_index = le.transform([background_label])[0]

    for i in range(num_detectors):
        box_start = int(i * box_width)
        box_end = int((i + 1) * box_width)
        
        best_iou = 0.0
        best_segment = None

        for (seg_start, seg_end, seg_class) in segments_for_window:
            inter_start = max(box_start, seg_start)
            inter_end = min(box_end, seg_end)
            inter = max(0, inter_end - inter_start)
            union = (box_end - box_start) + (seg_end - seg_start) - inter
            iou = inter / union if union > 0 else 0.0

            if iou > best_iou:
                best_iou = iou
                best_segment = (seg_start, seg_end, seg_class)

        if best_segment is not None:
            seg_start, seg_end, seg_class = best_segment
            seg_center = (seg_start + seg_end) / 2
            seg_width = seg_end - seg_start
            # p_c
            detections[i, 0] = best_iou
            # b_x
            detections[i, 1] = (seg_center - box_start) / box_width
            # b_w
            detections[i, 2] = seg_width / box_width
            # One-hot encoded class label
            detections[i, 3 + seg_class] = 1.0

        else:
            # No segment found, this is a background box
            detections[i, 0] = 0.0
            detections[i, 1] = 0.5  # Center
            detections[i, 2] = 1.0  # Full width
            detections[i, 3 + bg_index] = 1.0

    return detections

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
        kernal = 5
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=kernal, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=kernal, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_detectors * (3+num_classes))

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        logits = x.view(-1, self.num_detectors, 3 + self.num_classes)  # [batch, num_detectors, 3 + num_classes] - each item in 2nd dimension is a detection output

        # Use activation on the logits to get values:
        # TODO: Can use 0:1, 1:2, 2:3, to not need to unsqueeze the individual pieces?
        p_c = torch.clamp(logits[..., 0], 0, 1)  # limit confidence to [0, 1], encourage a higher congidence score, not using sigmoid
        b_x = torch.sigmoid(logits[..., 1])  # box location in [0, 1]
        b_w = torch.exp(torch.clamp(logits[..., 2], min=-5, max=5))  # box width allowed to be exponential
        class_probs = torch.softmax(logits[..., 3:], dim=-1)  # softmax the class probabilities to sum to 1 Assume that an object belongs to one class use sigmoid in v3

        detections = torch.cat((p_c.unsqueeze(-1), b_x.unsqueeze(-1), b_w.unsqueeze(-1), class_probs), dim=-1)

        return logits, detections


# TODO: What loss used in regular YOLO?
#   Per Copilot, seems to be mix of localization loss (CIoU), Binary Cross-Entropy of confidence score, and BC-E of
#   class prediction
# TODO: Need to use non-max suppression for the bounding boxes of segments (see Coursera video)
#   We should do this for the whole segment (all activities at once), not just per-activity? Or at least when we expect
#   (read: enforce) one activity at a time?


class YODALoss(nn.Module):
    """
    Implement the YODA loss as a composite of:
     - Localization loss (compare predicted to actual box sizes and locations)
     - Confidence loss (compare the predicted to actual confidence scores)
     - Classification loss (compare predicted to actual class predictions)
    """

    def __init__(self, lambda_localization=5.0, lambda_noobj=0.1):
        """
        Initialize the loss
        :param lambda_localization: scalar for the localization component of loss
        :param lambda_noobj: scalar for the confidence loss when no target is present
        """
        super().__init__()
        self.lambda_localization = lambda_localization
        self.lambda_noobj = lambda_noobj

    def forward(self, pred, target):
        """
        Compute the loss of the predicted against actual detections.

        Each should be a tensor of dimensions (batch_size, # boxes, 3 + num_classes)
        Where the last dimension consists of:
        [p_c, b_x, b_w, class_probs]
        """

        # Determine which cells have a segment target assigned (ground-truth p_c > 0):
        target_mask = target[..., 0] > 0

        # Localization loss:
        # Compute the MSE loss of the predicted locations and widths of segments (b_x and b_w),
        # only for those cells which have a detected target
        # TODO: Use sqrt for the width
        localization_loss = torch.tensor(0.)
        if target_mask.any():  # only compute if at least one box with target
            # Compute the part of the loss from target location relative to box (b_x):
            localization_loss = nn.MSELoss()(pred[..., 1:2][target_mask], target[..., 1:2][target_mask])

            # Compute the part of the loss from target width (b_w), using sqrt() to reduce skew of large boxes:
            localization_loss += nn.MSELoss()(torch.sqrt(pred[..., 2:3][target_mask]), torch.sqrt(target[..., 2:3][target_mask]))

        # Confidence loss:
        # The confidence loss is the MSE of the confidence (p_c) for each box, but split into two parts: for boxes that
        # are assigned to a segment in training, and a (scaled) amount for those that aren't assigned to a segment:
        conf_loss_with_targets = torch.tensor(0.)
        if target_mask.any():  # only compute if at least one box with target:
            conf_loss_with_targets = nn.MSELoss()(pred[..., 0:1][target_mask], target[..., 0:1][target_mask])

        conf_loss_no_targets = torch.tensor(0.)
        if not target_mask.all():  # only compute if there is at least one non-target box
            conf_loss_no_targets = nn.MSELoss()(pred[..., 0:1][~target_mask], target[..., 0:1][~target_mask])

        # Classification loss:
        # Compute the MSE loss of class probabilities for boxes where there is a target:
        class_loss = torch.tensor(0.)
        if target_mask.any():  # only compute if at least one box with target:
            class_loss = nn.MSELoss()(pred[..., 3:][target_mask], target[..., 3:][target_mask])

        # Combine the losses, using scaling:
        total_loss = self.lambda_localization * localization_loss + conf_loss_with_targets + self.lambda_noobj * conf_loss_no_targets + class_loss

        return total_loss


def compute_iou(gt_start, gt_end, pred_start, pred_end):
    inter_start = max(gt_start, pred_start)
    inter_end = min(gt_end, pred_end)
    inter = max(0, inter_end - inter_start)
    union = (gt_end-gt_start)+(pred_end-pred_start)-inter
    epsilon = 1e-6
    return inter / (union +epsilon)

import numpy as np

def evaluate_segments(model, dataloader, label_encoder, iou_thresh=0.05):
    model.eval()
    total_windows = 0
    windows_with_no_gt = 0
    windows_with_no_pred = 0

    total_gt = 0
    total_pred = 0
    matched = 0
    all_ious = []
    
    # Store true and predicted class labels for a final report
    all_true_classes = []
    all_pred_classes = []

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        with torch.no_grad():
            _, outputs = model(x_batch)

        batch_size = outputs.shape[0]
        total_windows += batch_size
        
        # NOTE: The predicted output needs to be converted back to the format used in `segments_to_detections`.
        # This is because the output of the model is a tensor of activations, not the segments themselves.
        # This part of the code is also fundamentally wrong, as it's trying to get the `start` and `end` from a tensor
        # that doesn't represent them directly.
        
        for i in range(batch_size):
            true_segs_tensor = y_batch[i]
            pred_segs_tensor = outputs[i]

            # Reconstruct true segments from the ground truth tensor
            true_list = []
            for j in range(true_segs_tensor.shape[0]):
                if true_segs_tensor[j, 0] > 0:  # Check for confidence > 0
                    p_c, b_x, b_w = true_segs_tensor[j, :3].tolist()
                    class_probs = true_segs_tensor[j, 3:].tolist()
                    true_class_idx = np.argmax(class_probs)
                    
                    # Convert to actual frame numbers
                    box_width = WINDOW_SIZE / NUM_DETECTORS
                    box_start = j * box_width
                    
                    segment_midpoint = box_start + box_width * b_x
                    segment_length = box_width * b_w
                    
                    seg_start = max(0, int(segment_midpoint - segment_length / 2))
                    seg_end = min(WINDOW_SIZE - 1, int(segment_midpoint + segment_length / 2))
                    
                    true_list.append((seg_start, seg_end, true_class_idx))
            
            # Reconstruct predicted segments from the model's output tensor
            # Apply non-max suppression here if desired, as in `construct_segments_from_detections`
            # For simplicity, we'll just check for high confidence.
            
            pred_list = []
            for j in range(pred_segs_tensor.shape[0]):
                p_c, b_x, b_w = pred_segs_tensor[j, :3].tolist()
                class_probs = pred_segs_tensor[j, 3:].tolist()
                
                # Check for confidence
                if p_c > 0.5:
                    pred_class_idx = np.argmax(class_probs)
                    
                    box_width = WINDOW_SIZE / NUM_DETECTORS
                    box_start = j * box_width
                    
                    segment_midpoint = box_start + box_width * b_x
                    segment_length = box_width * b_w
                    
                    seg_start = max(0, int(segment_midpoint - segment_length / 2))
                    seg_end = min(WINDOW_SIZE - 1, int(segment_midpoint + segment_length / 2))
                    
                    pred_list.append((seg_start, seg_end, pred_class_idx))
            

            if not true_list:
                windows_with_no_gt += 1
            if not pred_list:
                windows_with_no_pred += 1

            total_gt += len(true_list)
            total_pred += len(pred_list)

            ious = []
            matched_gt = set()

            for pj, pred_seg in enumerate(pred_list):
                for gj, gt_seg in enumerate(true_list):
                    if gj in matched_gt:
                        continue
                    
                    iou = compute_iou(gt_seg[0], gt_seg[1], pred_seg[0], pred_seg[1])
                    ious.append(float(iou))
                    
                    if iou >= iou_thresh and pred_seg[2] == gt_seg[2]:
                        matched += 1
                        matched_gt.add(gj)
                        all_true_classes.append(gt_seg[2])
                        all_pred_classes.append(pred_seg[2])
                        break
            
            # Add all unmatched ground truths and predictions to the list
            for gt_idx, gt_seg in enumerate(true_list):
                if gt_idx not in matched_gt:
                    all_true_classes.append(gt_seg[2])
                    all_pred_classes.append(label_encoder.transform(['Other'])[0])

            all_ious.extend(ious)

    precision = matched / total_pred if total_pred > 0 else 0.0
    recall = matched / total_gt if total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    best_iou = max(all_ious) if all_ious else 0.0
    mean_iou = np.mean(all_ious) if all_ious else 0.0

    print("\n=== Segment-Level Evaluation ===")
    print(f"Total windows evaluated: {total_windows}")
    print(f"Windows with no ground-truth segments:   {windows_with_no_gt}")
    print(f"Windows with no predictions:   {windows_with_no_pred}")
    print(f"Total ground-truth segments:   {total_gt}")
    print(f"Total predicted segments:      {total_pred}")
    print(f"Matched segments:              {matched}\n")
    print(f"Precision:                     {precision:.3f}")
    print(f"Recall:                        {recall:.3f}")
    print(f"F1 Score:                      {f1:.3f}")
    print(f"Best IoU:                      {best_iou:.3f}")
    print(f"Mean IoU:                      {mean_iou:.3f}")
    
    print("\n=== Classification Report for Segments ===")
    print(classification_report(all_true_classes, all_pred_classes, target_names=label_encoder.classes_, zero_division=0))



def train(model, dataloader, optimizer, criterion, epochs=10, num_classes=5):
    print("Starting training...")

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits, preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

def evaluate_plot(model, dataset, window_size, stride, le):
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

    all_preds = []
    all_targets = []

    for x_batch, y_batch in eval_dataloader:
        x_batch = x_batch.to(device)

        with torch.no_grad():
            logits, preds = model(x_batch)

            # Move values back to cpu and split batch out into individual arrays per window:
            preds_by_window = list(preds.to('cpu').numpy())
            targets_by_window = list(y_batch.numpy())

            all_preds.extend(preds_by_window)
            all_targets.extend(targets_by_window)

    pred_segments = []

    for window_index, window_detections in enumerate(all_preds):
        segments_for_window = construct_segments_from_detections(window_detections, le, window_size, 0.05, 0.0)

        # Convert the segment indices to overall indices using the window_index:
        for segment in segments_for_window:
            seg_start_in_window, seg_end_in_window, p_c, label = segment
            seg_start_in_sequence = window_index * window_size + seg_start_in_window
            seg_end_in_sequence = window_index * window_size + seg_end_in_window

            pred_segments.append((seg_start_in_sequence, seg_end_in_sequence, p_c, label))

    actual_segments = []

    for window_index, window_detections in enumerate(all_targets):
        segments_for_window = construct_segments_from_detections(window_detections, le, window_size, 0.05, 0.0)

        # Convert the segment indices to overall indices using the window_index:
        for segment in segments_for_window:
            seg_start_in_window, seg_end_in_window, p_c, label = segment
            seg_start_in_sequence = window_index * window_size + seg_start_in_window
            seg_end_in_sequence = window_index * window_size + seg_end_in_window

            actual_segments.append((seg_start_in_sequence, seg_end_in_sequence, p_c, label))

    num_windows = len(all_targets)
    num_detectors = all_targets[0].shape[0]
    plot_segments(actual_segments, pred_segments, num_windows, window_size, num_detectors, le)

def plot_segments(actual_segments, pred_segments, num_windows, window_size, num_detectors, le):
    """Plot the actual vs predicted segments."""

    total_length = num_windows * window_size
    detector_size = window_size // num_detectors

    classes = list(le.classes_)

    cmap = plt.get_cmap('tab20', len(classes))

    fig, ax = plt.subplots(figsize=(20, 3), constrained_layout=True)

    # Plot the actual segments:
    for segment in actual_segments:
        seg_start, seg_end, p_c, label = segment
        label_index = classes.index(label)

        ax.barh(0, left=seg_start, width=seg_end-seg_start, height=0.5, color=cmap(label_index))

    # Plot the predicted segments:
    for segment in pred_segments:
        seg_start, seg_end, p_c, label = segment
        label_index = classes.index(label)

        ax.barh(-0.5, left=seg_start, width=seg_end-seg_start, height=0.5, color=cmap(label_index))

    # Drop a separation:
    ax.axhline(-0.25, color='black')


    # Set axis limits:
    ax.set_xlim(0, total_length)
    ax.set_ylim(-0.75, 0.25)
    ax.set_yticks([])
    ax.set_xlabel('Frames Since Start')
    ax.set_title('Activity Segments')

    # Create legend:
    legend_patches = [mpatches.Patch(color=cmap(label_index), label=label) for label_index, label in enumerate(sorted(classes))]
    ax.legend(handles=legend_patches, loc='upper center', ncol=len(classes)/3 + 1, bbox_to_anchor=(0.5, -0.25))

    # plt.subplots_adjust(bottom=0.65)
    plt.tight_layout()
    plt.show()

from collections import deque
import numpy as np

def construct_segments_from_detections(detections, le, window_size, min_p_c, allowed_iou):
    """
    Reconstruct label segments in a window based on the detection box predictions (or ground truths), along with the
    window size.

    :param detections: list of detections for the window as a #detectors x (3 + #classes) array (each row is a detector
        output of [p_c, b_x, b_w, c_1, ..., c_N]
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
        p_c = detection[0]  # confidence
        b_x = detection[1]  # center of detected segment (relative to detection box)
        b_w = detection[2]  # width of detected segment (relative to detection box)
        p_classes = detection[3:]  # probability of classes (note: includes Other)

        if p_c < min_p_c:
            # Not high enough confidence, skip
            continue

        # Calculate the start/end indices (relative to the window) of the detected segment:
        detector_start = detector_index * detection_width

        segment_midpoint = detector_start + detection_width * b_x  # midpoint using relative size to detector
        segment_length = detection_width * b_w  # segment length relative to detector size

        segment_start = max(int(segment_midpoint - segment_length / 2), 0)
        segment_end = min(int(segment_midpoint + segment_length / 2), window_size - 1)

        proposed_segments.append((segment_start, segment_end, p_c, p_classes))
    
    # Perform non-max-suppression on the proposed segments:
    # Sort the segments from highest to lowest p_c:
    remaining_proposed = deque(sorted(proposed_segments, key=lambda segment: segment[2], reverse=True))

    final_segments = []

    while remaining_proposed:
        # Get the remaining segment with highest confidence, and add it to final segments:
        highest_confidence_segment = remaining_proposed.popleft()
        seg_start, seg_end, p_c, p_classes = highest_confidence_segment
        class_idx = int(np.argmax(p_classes))
        segment_class_name = le.inverse_transform([class_idx])[0]

        final_segments.append((seg_start, seg_end, p_c, segment_class_name))

        # Filter out overlapping ones (IOU > allowed_iou)
        filtered = []
        for (s_start, s_end, sc, cl_probs) in remaining_proposed:
            if iou((seg_start, seg_end), (s_start, s_end)) <= allowed_iou:
                filtered.append((s_start, s_end, sc, cl_probs))

        remaining_proposed = deque(filtered)

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
    csv_files = glob.glob("*.chest.csv")#adjust this if you have different path
    X, Y, label_encoder = load_and_window_data(csv_files, window_size=WINDOW_SIZE, stride=STRIDE, num_detectors=NUM_DETECTORS)

    dataset = YodaSensorDataset(X, Y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    num_classes = len(label_encoder.classes_)
    model = Yoda(input_channels=X.shape[2], num_classes=num_classes, num_detectors=NUM_DETECTORS)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Move model and optimizer to GPU:
    model.to(device)

    train(model, train_loader, optimizer, YODALoss(), epochs=EPOCHS,num_classes=num_classes)
    print("evaluate training set")
    evaluate_segments(model, train_loader, label_encoder, iou_thresh=0.05)
    print("evaluate testing set")
    evaluate_segments(model, test_loader, label_encoder, iou_thresh=0.05)
    evaluate_plot(model, dataset, window_size=WINDOW_SIZE, stride=STRIDE, le=label_encoder)