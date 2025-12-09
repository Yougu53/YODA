import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
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
    def __init__(self, input_channels, num_classes, num_detectors, num_anchors, device, pretrained_path):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_detectors = num_detectors
        # The output size for each anchor: 1 (confidence) + 1 (center_x) + 1 (width) + num_classes
        self.num_outputs_per_anchor = 3 + self.num_classes
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading pretrained encoder...")
            # The following is the example for CPC pretrained model, modify it with yours
            from cpc import EnhancedEncoder1D
            encoder = EnhancedEncoder1D(in_channels=input_channels)
            state = torch.load(pretrained_path, map_location=device,weights_only=True) # Be careful using weights_only
            encoder.load_state_dict(state, strict=False)
            self.feature_extractor = encoder.net
            print("pretrained encoder loaded.")
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
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
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
        # Classification loss:
        # Compute the loss as the BCE loss over the prediction values, only for cells with a target segment.
        classification_loss = self.bce_loss(pred[..., 3:], target[..., 3:]) * target_mask.unsqueeze(-1)  # unsqueeze to match dims

        # Combine the losses using scaling factors, and normalize by the batch size (to get mean across batch):
        num_boxes = pred.shape[0] * pred.shape[1]
        total_loss = (
            confidence_loss +
            self.lambda_localization * localization_loss_x.sum() +
            self.lambda_localization * localization_loss_w.sum() +
            classification_loss.sum()
        ) / num_boxes

        return total_loss