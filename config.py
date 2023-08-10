# Training hyperparameters
seed = 42
save = "./saved_models/"
embedding_size= 128
temperature = 0.5
batch_size = 64
max_epochs = 1000

gradient_accumulation_steps = 4
learning_rate = 3e-4
weight_decay = 1e-6

# Dataset
data_dir = "data/"
num_workers = 4

# Compute related
accelerator = "gpu"
devices = [0]
precision = 32