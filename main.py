from data_preprocess import load_and_window_segment, find_all_segments, load_and_window_single_segment
from yoda import YodaSensorDataset, Yoda, YODALoss
from evaluation import evaluate_plot
import torch
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# features to exclude for GSUR
EXCLUDE_FEATURES = [
    "stamp", "heart_rate_motion_context", 
    "course", "speed","respiratory_rate",
    "sleep_stage", "battery_state"
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_COLUMN = "user_activity_label"
BATCH_SIZE = 32
EPOCHS = 50
NUM_DETECTORS = 20
WINDOW_SIZE = 8192
STRIDE = 8192
ANCHOR_BOXES = torch.tensor([[0.5], [1.0], [2.0]]).to(device)
NUM_ANCHORS = len(ANCHOR_BOXES)
TARGET = 'Walk'
# Define similar activities for the new evaluation metric
SIMILAR_ACTIVITIES = {
    'Sit': ['LieDown','Stand'],
    'LieDown': ['Sit','Stand'],
    'Stand':['LieDown','Sit'],
    'ClimbUpStairs': ['ClimbDownStairs'],
    'ClimbDownStairs': ['ClimbUpStairs']
}
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

if __name__ == "__main__":
    test_files = ['gsur2504.csv','gsur2501.csv','gsur2503.csv']
    train_files = ['gsur2510.csv','gsur2511.csv','gsur2514.csv','gsur2505.csv','gsur2506.csv','gsur2507.csv','gsur2509.csv']
    pretrained_path = "" # fill it with pretrained model and network code address
    train_X, train_Y, label_encoder = load_and_window_segment(train_files,TARGET,WINDOW_SIZE,STRIDE,'Other', EXCLUDE_FEATURES, LABEL_COLUMN, NUM_DETECTORS,ANCHOR_BOXES)
    train_dataset = YodaSensorDataset(train_X, train_Y)
    train_loader = DataLoader(train_dataset, batch_size=32)
    
    num_classes = len(label_encoder.classes_)
    model = Yoda(input_channels=train_X.shape[2], num_classes=num_classes, num_detectors=NUM_DETECTORS, num_anchors=NUM_ANCHORS, device=device, pretrained_path=pretrained_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)

    train(model, train_loader, optimizer, YODALoss(), epochs=EPOCHS)
    evaluate_plot(model, train_dataset, 'Training', window_size=WINDOW_SIZE, stride=STRIDE, le=label_encoder,anchors=ANCHOR_BOXES, device=device)

    all_test_segments = find_all_segments(test_files, EXCLUDE_FEATURES, LABEL_COLUMN)
    # Iterate through each segment and perform evaluation
    print("\n--- Evaluating each segment in the test set ---")
    
    for activity, start, duration in all_test_segments:
        # Ignore "Other" activities
        if activity != TARGET:
            continue
            
        similar_activities = SIMILAR_ACTIVITIES.get(activity, [])
        non_A = 'Other'
            
        # Create a temporary, isolated dataset for this segment
        temp_X, temp_Y, temp_le = load_and_window_single_segment(
            test_files, 
            target_activity=activity, 
            target_start_frame=start, 
            target_duration=duration,
            similar =similar_activities,
            nonA = non_A,
            le=label_encoder,
            exclude_features=EXCLUDE_FEATURES, 
            label_column=LABEL_COLUMN,
            window_size=WINDOW_SIZE,
            num_detection=NUM_DETECTORS,
            anchor=ANCHOR_BOXES
        )
        if temp_X is not None:
            temp_dataset = YodaSensorDataset(temp_X, temp_Y)
            
            # Since the dataset only contains one window, we need to adjust the print statements
            print(f"\nEvaluating segment: Activity={activity}, Start={start}, Duration={duration}")
            
            # Call the evaluation function
            evaluate_plot(
                model, 
                temp_dataset, 
                f'Testing - {activity} Segment ({start})', 
                window_size=duration, 
                stride=duration, 
                le=label_encoder, 
                anchors=ANCHOR_BOXES
            )