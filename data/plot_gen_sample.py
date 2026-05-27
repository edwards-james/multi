import math
import pickle
import numpy as np
from tqdm import tqdm
import scipy.special as sp
from skspatial.objects import Vector
from extremitypathfinder import PolygonEnvironment
from scipy.spatial import Delaunay
from joblib import Parallel, delayed
import torch
import matplotlib.pyplot as plt

OUTPUT_DIR = "/home/je540/multi"

import sys
import os

# Add the project root to Python's search path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.config import config

# load the heading data
with open(f"{OUTPUT_DIR}/data/headings_data.pkl", "rb") as f:
    headings_data = pickle.load(f)

with open(f"{OUTPUT_DIR}/data/abs_traffic_data.pkl", "rb") as f:
    abs_traffic_data = pickle.load(f)


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






eps   = 1e-8
entry = np.asarray((-2.5 + eps, 0))

# pick a random example from the dataset
i = np.random.randint(0,len(headings_data))

# traffic at a fixed position
traffic_positions = abs_traffic_data[i][0]

print(traffic_positions)



# Match the safety radius used to generate the training data in data/gen2.py.
sep_radius = config["sep_radius"]
# sector boundaries
x_lim = 2.5
y_lim = 0.4
Z_boundary      = np.array(((-x_lim,-y_lim),(x_lim,-y_lim),(x_lim,y_lim),(-x_lim,y_lim),(-x_lim,-y_lim)))
holes           = [create_disc(tp, sep_radius) for tp in traffic_positions]
hole_boundaries = [np.array(hole) for hole in holes]


fig, ax = plt.subplots(1, 1, figsize=(15, 15))

route = headings_data[i]
headings = list(route)[1:]
headings = headings[:headings.index(0)]

print(headings)
print(len(headings))

# reconstruct trajectory
positions   = [entry]
current_pos = entry
for heading in headings:
    pos = point_on_heading(current_pos, int(heading), config["speed"], config["tt"], 0, 0)
    positions.append(pos)
    current_pos = pos

positions = np.asarray(positions)

# plot the trajectory
ax.plot(positions[:, 0],  positions[:, 1],  color='red', alpha=1.0)
ax.scatter(positions[:, 0], positions[:, 1], color='red', zorder=5, alpha=1.0)

ax.plot(Z_boundary[:, 0], Z_boundary[:, 1], "-k")
for hole_boundary in hole_boundaries:
    ax.plot(hole_boundary[:, 0], hole_boundary[:, 1], "-k")

ax.scatter(
    np.array(traffic_positions)[:, 0],
    np.array(traffic_positions)[:, 1],
    color='blue', marker="P"
)

ax.set_aspect(1.0)

plt.savefig("trajectory.png", bbox_inches="tight")
print("plot saved to trajectory.png")
