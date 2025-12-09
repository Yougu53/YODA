import torch
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import math
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report,accuracy_score
import numpy as np
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
        
        class_id = torch.argmax(data[3:]).item()
        class_label = le.inverse_transform([class_id])[0]
        
        gt_segments.append((start, end, 1.0, class_id, class_label))
        
    return gt_segments
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
    
def evaluate_plot(model, dataset, dataset_name, window_size, stride, le,anchors, device, background_label = 'Other'):
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
        
        class_id = torch.argmax(data[3:]).item()
        class_label = le.inverse_transform([class_id])[0]
        
        gt_segments.append((start, end, 1.0, class_id, class_label))
        
    return gt_segments
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
            
            class_probs = torch.sigmoid(detection[3:])
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