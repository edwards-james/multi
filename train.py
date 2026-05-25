import pickle
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
# from tqdm import tqdm

from src.config import config
from src.model import Model
from src.utils import save_checkpoint


torch.manual_seed(1)



OUTPUT_DIR = "/home/james/multi"

# load the data
with open(f"{OUTPUT_DIR}/data/headings_data.pkl", "rb") as f:
    headings_data = pickle.load(f)

with open(f"{OUTPUT_DIR}/data/rel_positions_data.pkl", "rb") as f:
    rel_positions_data = pickle.load(f)

with open(f"{OUTPUT_DIR}/data/rel_traffic_data.pkl", "rb") as f:
    rel_traffic_data = pickle.load(f)


# replace the inf values with 0
replace_dict = {np.inf: 0}
for key, value in replace_dict.items():
    rel_positions_data[rel_positions_data == key] = value
    rel_traffic_data[rel_traffic_data == key] = value

# print(rel_traffic_data.shape)

rel_traffic_data = rel_traffic_data.reshape(len(headings_data), config["context_length"]+1, 2*config["n_aircraft"])
# rel_traffic_data = rel_traffic_data.squeeze(axis=2)

print(f"headings_data: {headings_data.shape}")
print(f"rel_positions_data: {rel_positions_data.shape}")
print(f"rel_traffic_data: {rel_traffic_data.shape}")



class HeadingTokeniser:

    def __init__(self, vocab):
        self.s_to_i = vocab
        self.i_to_s = { i:s for s,i in vocab.items()}
    
    def encode(self, text):
        ids = np.vectorize(self.s_to_i.__getitem__)(text)
        return ids

    def decode(self, ids):
        text = np.vectorize(self.i_to_s.__getitem__)(ids)
        return text


# construct vocab dictionary (note: 0 maps to 0, -1 maps to 37)
heading_options = [i*10 for i in range(1,37)]
heading_options.insert(0,0)
heading_options.append(-1)
voc = {heading_options[i] : i for i in range(len(heading_options))}
tokeniser = HeadingTokeniser(voc)
vocab_size = len(voc)

# encode and split the data into training and validation
headings_data = torch.tensor(tokeniser.encode(headings_data), dtype=torch.long)
n = int(0.9*len(headings_data))
train_data = {
    "heading" : headings_data[:n],
    "position" : torch.tensor(rel_positions_data[:n], dtype=torch.float32),
    "traffic" : torch.tensor(rel_traffic_data[:n], dtype=torch.float32)
}
val_data = {
    "heading" : headings_data[n:],
    "position" : torch.tensor(rel_positions_data[n:], dtype=torch.float32),
    "traffic" : torch.tensor(rel_traffic_data[n:], dtype=torch.float32)
}



def get_batch(split):
    """ 
    Generate a batch of data x (inputs), y (targets)
    """

    match split:
        case "train":
            data = train_data
        case "validate":
            data = val_data
    
    # make a tensor with batch_size random numbers (to pick out example trajectories)
    ix = torch.randint(len(data["heading"]), (config["batch_size"],))
    batch = {}

    # get the data for headings, positions or targets
    # for each, build lists with the inputs (x) and the targets (y)
    for key in data.keys():
        
        dataset = data[key]

        if key == "heading":
            x = torch.tensor(np.asarray([dataset[i][:config["context_length"]] for i in ix]), dtype=torch.long)
            y = torch.tensor(np.asarray([dataset[i][1:config["context_length"]+1] for i in ix]), dtype=torch.long)
        else:
            x = torch.tensor(np.asarray([dataset[i][:config["context_length"]] for i in ix]), dtype=torch.float32)
            y = torch.tensor(np.asarray([dataset[i][1:config["context_length"]+1] for i in ix]), dtype=torch.float32)
        
        batch[key] = (x,y)

    return batch





# disable gradient calculations
@torch.no_grad()
def estimate_loss():

    out = {}

    # put the model into evaluation mode
    model.eval()

    for split in ["train", "validate"]:
        losses = torch.zeros(config["eval_iters"])
        for k in range(config["eval_iters"]):

            batch_dict = get_batch(split)
            X_heading, Y_heading = batch_dict["heading"]
            X_position, _ = batch_dict["position"]
            X_traffic, _ = batch_dict["traffic"]

            X_heading = X_heading.to(config["device"])
            X_position = X_position.to(config["device"])
            X_traffic = X_traffic.to(config["device"])
            Y_heading = Y_heading.to(config["device"])

            # evaluate the logits, loss
            _, loss = model(X_heading, X_position, X_traffic, Y_heading)

            losses[k] = loss.item()

        out[split] = losses.mean()
    
    # put the model back into training mode
    model.train()

    return out






start_time = time.time()

model = Model(config).to(config["device"])
optimiser = torch.optim.AdamW(model.parameters())

train_losses, validate_losses, steps, track_lrs = [], [], [], []

initial_lr = config["initial_lr"]
peak_lr = config["peak_lr"]
min_lr = 0.1 * initial_lr
warmup_steps = int(0.2 * config["max_iters"])
lr_increment = (peak_lr - initial_lr) / warmup_steps

for global_step in range(config["max_iters"]):

    # eval at regular intervals
    if global_step % config["eval_interval"] == 0:

        losses = estimate_loss()
        print(
            f"step {global_step}: "
            f"train loss {losses['train']:.4f}, "
            f"validate loss {losses['validate']:.4f}, "
            f"lr {peak_lr if global_step == 0 else track_lrs[-1]:.6f}"
        )
        steps.append(global_step)
        train_losses.append(losses["train"])
        validate_losses.append(losses["validate"])
        model.train()  # restore train mode after estimate_loss

    # get batch
    batch_dict = get_batch("train")
    x_heading, y_heading = batch_dict["heading"]
    x_position, _ = batch_dict["position"]
    x_traffic, _ = batch_dict["traffic"]

    x_heading  = x_heading.to(config["device"])
    x_position = x_position.to(config["device"])
    x_traffic  = x_traffic.to(config["device"])
    y_heading  = y_heading.to(config["device"])

    # adjust learning rate
    if global_step < warmup_steps:
        lr = initial_lr + global_step * lr_increment
    else:
        progress = (global_step - warmup_steps) / (config["max_iters"] - warmup_steps)
        lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    for param_group in optimiser.param_groups:
        param_group["lr"] = lr
    track_lrs.append(lr)

    # forward pass
    optimiser.zero_grad(set_to_none=True)
    logits, loss = model(x_heading, x_position, x_traffic, y_heading)

    # backward pass
    loss.backward()

    # gradient clipping after warmup
    if global_step >= warmup_steps:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimiser.step()


    if global_step % config["save_interval"] == 0:
        save_checkpoint(model, optimiser, global_step, config["save_dir"])


# save the final trained model
save_checkpoint(model, optimiser, config["max_iters"], config["save_dir"])


end_time = time.time()
print(f"training completed in {(end_time - start_time) / 60:.1f} minutes")


# save the losses and lrs from training
with open(f"{OUTPUT_DIR}/train_stats/train_losses.pkl", "wb") as f:
    pickle.dump(np.array(train_losses), f)

with open(f"{OUTPUT_DIR}/train_stats/validate_losses.pkl", "wb") as f:
    pickle.dump(np.array(validate_losses), f)

with open(f"{OUTPUT_DIR}/train_stats/track_lrs.pkl", "wb") as f:
    pickle.dump(np.array(track_lrs), f)

with open(f"{OUTPUT_DIR}/train_stats/steps.pkl", "wb") as f:
    pickle.dump(np.array(steps), f)
