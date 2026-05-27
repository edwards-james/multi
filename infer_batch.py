import numpy as np
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.model import Model
from src.config import config


device = config["device"]

checkpoint_path = "checkpoints/ckpt_100000.pt"


class HeadingTokeniser:

    def __init__(self, vocab):
        self.s_to_i = vocab
        self.i_to_s = {i: s for s, i in vocab.items()}

    def encode(self, text):
        return np.vectorize(self.s_to_i.__getitem__)(text)

    def decode(self, ids):
        return np.vectorize(self.i_to_s.__getitem__)(ids)


heading_options = [i * 10 for i in range(1, 37)]
heading_options.insert(0, 0)
heading_options.append(-1)
voc = {heading_options[i]: i for i in range(len(heading_options))}
tokeniser = HeadingTokeniser(voc)


# load model
model = Model(config).to(device)
ckpt = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"loaded model weights from: {checkpoint_path}", flush=True)


def point_on_heading(point, aircraft_heading, aircraft_speed, time, wind_direction=0, wind_speed=0):
    """Compute the position an aircraft will finish in when flown for a set time"""
    aircraft_angle = math.radians(90 - aircraft_heading)
    wind_angle = -math.radians(wind_direction + 90)

    aircraft_vx = aircraft_speed * math.cos(aircraft_angle)
    aircraft_vy = aircraft_speed * math.sin(aircraft_angle)
    wind_vx = wind_speed * math.cos(wind_angle)
    wind_vy = wind_speed * math.sin(wind_angle)

    x = point[0] + (aircraft_vx + wind_vx) * time
    y = point[1] + (aircraft_vy + wind_vy) * time

    return np.array([x, y])


def create_disc(centre, radius):
    """Creates a quasi circle around a centre point (aircraft)"""
    centre_x, centre_y = centre[0], centre[1]
    angles = np.linspace(0, 2 * np.pi, 21)
    coords = [
        [centre_x + radius * np.cos(angle), centre_y + radius * np.sin(angle)]
        for angle in angles
    ]
    coords.reverse()
    return coords


def generate(model, config, headx, posx, trafx, abs_traff):

    prob_list = []
    max_steps = config["context_length"]
    step = 0

    # position of the target - extract once without moving whole tensor
    target = posx[0].cpu().numpy()

    while (len(headx[0]) == 1 or int(headx[0, -1]) != 0) and step < max_steps:
        step += 1

        headx_cond = headx[:, -config["context_length"]:]
        posx_cond  = posx[-config["context_length"]:, :].unsqueeze(0)
        trafx_cond = trafx[-config["context_length"]:, :].unsqueeze(0)

        logits, _ = model(headx_cond, posx_cond, trafx_cond)
        logits = logits[:, -1, :]
        # Do not let the sampler choose the padding token: it is not a heading, and
        # treating it as one corrupts the generated trajectory state.
        logits[:, config["pad_token_id"]] = -torch.inf
        probs = F.softmax(logits, dim=-1)
        prob_list.append(probs)
        headx_next = torch.multinomial(probs, num_samples=1)

        # only pull the last row to CPU for numpy arithmetic, not the whole tensor
        # Advance the physical position using the heading just sampled, so the
        # conditioning state given back to the model matches the plotted route.
        next_token = int(headx_next[0, 0].item())
        current_pos = target - posx[-1].cpu().numpy()
        if next_token == 0:
            next_pos = current_pos
        else:
            next_pos = point_on_heading(
                current_pos,
                next_token * 10,
                config["speed"],
                config["tt"],
                0, 0
            )

        posx_next = target - next_pos
        trafx_next = (
            abs_traff - np.array([next_pos for _ in range(config["n_aircraft"])])
        ).reshape(1, 2 * config["n_aircraft"])

        # convert new positions to tensors and move to device once
        posx_next  = torch.tensor(posx_next,  dtype=torch.float32).unsqueeze(0).to(device)
        trafx_next = torch.tensor(trafx_next, dtype=torch.float32).to(device)
        # ------------ zeroing test
        # posx_next = torch.zeros_like(posx_next)
        # ------------

        headx = torch.cat((headx, headx_next), dim=1)
        posx  = torch.cat((posx,  posx_next),  dim=0)
        trafx = torch.cat((trafx, trafx_next), dim=0)

    return headx, prob_list, posx


eps   = 1e-8
exit  = np.asarray((2.5 - eps, 0))
entry = np.asarray((-2.5 + eps, 0))










# Match the safety radius used to generate the training data in data/gen2.py.
sep_radius = config["sep_radius"]
# sector boundaries
x_lim = 2.5
y_lim = 0.4

# random traffic positions
traffic_positions = [
    np.array([np.random.uniform(-x_lim, x_lim), np.random.uniform(-y_lim, y_lim)])
    for _ in range(config["n_aircraft"])
]

# traffic_positions = [
#     np.array([0, 0])
#     for _ in range(config["n_aircraft"])
# ]

print(traffic_positions)

Z_boundary      = np.array(((-x_lim,-y_lim),(x_lim,-y_lim),(x_lim,y_lim),(-x_lim,y_lim),(-x_lim,-y_lim)))
holes           = [create_disc(tp, sep_radius) for tp in traffic_positions]
hole_boundaries = [np.array(hole) for hole in holes]


fig, ax = plt.subplots(1, 1, figsize=(15, 15))

for _ in tqdm(range(100)):

    head_0 = torch.zeros((1, 1), dtype=torch.long, device=device)
    pos_0  = torch.zeros((1, 2), dtype=torch.float32, device=device)
    pos_0[0, 0] = exit[0] - entry[0]
    pos_0[0, 1] = exit[1] - entry[1]
    traf_0 = np.array(traffic_positions) - np.array([entry for _ in range(config["n_aircraft"])])
    traf_0 = torch.tensor(traf_0.reshape(1, 2 * config["n_aircraft"]), dtype=torch.float32, device=device)
    # -------------- zeroing test
    # pos_0 = torch.zeros_like(pos_0)
    # --------------

    with torch.no_grad():
        # Keep the absolute traffic argument in the same entry-relative frame as
        # traf_0, otherwise later relative-traffic updates mix coordinate frames.
        gen = generate(model, config, headx=head_0, posx=pos_0, trafx=traf_0, abs_traff=np.array(traffic_positions) - entry)

    route     = gen[0][0].tolist()
    prob_list = gen[1]

    route    = tokeniser.decode(route)
    # Drop the initial BOS token and remove EOS only when the model actually
    # emitted it; otherwise keep every sampled heading for reconstruction.
    headings = list(route)[1:]
    if headings and headings[-1] == 0:
        headings = headings[:-1]

    # reconstruct trajectory
    positions   = [entry]
    current_pos = entry
    for heading in headings:
        pos = point_on_heading(current_pos, int(heading), config["speed"], config["tt"], 0, 0)
        positions.append(pos)
        current_pos = pos

    positions = np.asarray(positions)

    # plot the trajectory
    ax.plot(positions[:, 0],  positions[:, 1],  color='red', alpha=0.2)
    # ax.scatter(positions[:, 0], positions[:, 1], color='red', zorder=5, alpha=0.2)


ax.plot(Z_boundary[:, 0], Z_boundary[:, 1], "-k")
for hole_boundary in hole_boundaries:
    ax.plot(hole_boundary[:, 0], hole_boundary[:, 1], "-k", alpha=0.3)

ax.scatter(
    np.array(traffic_positions)[:, 0],
    np.array(traffic_positions)[:, 1],
    color='blue', marker="P"
)

ax.set_aspect(1.0)

plt.savefig("trajectories.png", bbox_inches="tight")
print("plot saved to trajectories.png")
