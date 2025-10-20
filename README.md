Note: 

Evaluation

Start with one activity A in isolation (walk, stand, up/down stairs, sit, ironing/brush teeth -- not included in PAMAP2)

Train on  A plus Other

Repeat for every occurrence of A in the test dataset

Remove all but that occurrence in the dataset

Remove similar activities (for walk these would be NordicWalking, Running)

Label all other activities as Other

Evaluate with metrics (acc, IoU, f1, latency, offset, precision, recall)

Aggregated datasets (PAMAP2, UCI-HAR)

Capture24, GSUR datasets, aggregate labeled instances

Compare with baselines

Report per activity, per dataset, global across activities and datasets (TO DO)
