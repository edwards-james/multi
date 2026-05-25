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

OUTPUT_DIR = "/home/je540/multi"

import sys
import os

# Add the project root to Python's search path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.config import config


def point_on_heading(point, aircraft_heading, aircraft_speed, time, wind_direction=0, wind_speed=0):

    """
    Compute the position an aircraft will finish in when flown for a set time
    """

    aircraft_angle = math.radians(90 - aircraft_heading)
    wind_angle = - math.radians(wind_direction + 90)

    aircraft_vy = aircraft_speed * math.sin(aircraft_angle)
    aircraft_vx = aircraft_speed * math.cos(aircraft_angle)

    wind_vy = wind_speed * math.sin(wind_angle)
    wind_vx = wind_speed * math.cos(wind_angle)

    x = point[0] + (aircraft_vx + wind_vx) * time
    y = point[1] + (aircraft_vy + wind_vy) * time
    
    return np.array([x,y])


def shortest_route_path(point1: np.array, point2: np.array, environment: PolygonEnvironment) -> np.array:
    """
    Find the shortest path from point1 to point2 through the pre-built environment
    """
    path, _ = environment.find_shortest_path(tuple(point1), tuple(point2))
    return path


def segments_cross(seg_start: np.ndarray, seg_end: np.ndarray,
                   edge_start: np.ndarray, edge_end: np.ndarray,
                   eps: float = 1e-10) -> bool:
    """
    Returns True if two line segments strictly cross each other.
    Endpoint touches and parallel/collinear cases return False.

    Parameters
    ----------
    seg_start, seg_end   : endpoints of the first segment (aircraft path)
    edge_start, edge_end : endpoints of the second segment (polygon edge)
    eps                  : tolerance for excluding endpoint touches
    """
    seg_dir  = seg_end  - seg_start   # direction vector of the flight segment
    edge_dir = edge_end - edge_start  # direction vector of the polygon edge

    # cross product of the two direction vectors
    # if zero, the segments are parallel or collinear — no crossing
    cross = seg_dir[0] * edge_dir[1] - seg_dir[1] * edge_dir[0]
    if abs(cross) < eps:
        return False

    # t is how far along the flight segment the intersection occurs (0=start, 1=end)
    # u is how far along the polygon edge the intersection occurs (0=start, 1=end)
    offset = edge_start - seg_start
    t = (offset[0] * edge_dir[1] - offset[1] * edge_dir[0]) / cross
    u = (offset[0] * seg_dir[1]  - offset[1] * seg_dir[0])  / cross

    # eps < t < 1-eps  : exclude the aircraft path's own endpoints
    # -eps < u < 1-eps : include polygon vertex hits (u=0), exclude shared end
    #                    vertex (u=1) so each polygon vertex is owned by one edge
    return eps < t < 1 - eps and -eps < u < 1 - eps


def doIntersectPolygon(segment: list[np.ndarray], polygon: np.ndarray) -> bool:
    """
    Returns True if a line segment strictly crosses any edge of a polygon.
    Endpoint touches are ignored — only true crossings count.

    Parameters
    ----------
    segment : [start, end] defining the aircraft's proposed path
    polygon : (N, 2) array of vertices; assumed closed (polygon[0] == polygon[-1])
    """
    assert np.allclose(polygon[0], polygon[-1]), \
        "Polygon must be closed (first and last vertex must match)"

    seg_start, seg_end = segment[0], segment[1]

    for i in range(len(polygon) - 1):
        edge_start = polygon[i]
        edge_end   = polygon[i + 1]
        if segments_cross(seg_start, seg_end, edge_start, edge_end):
            return True

    return False


def create_disc(centre, radius):
    centre_x, centre_y = centre[0], centre[1]
    angles = np.linspace(0, 2 * np.pi, 21)
    coords = np.stack([
        centre_x + radius * np.cos(angles),
        centre_y + radius * np.sin(angles),
    ], axis=1)
    return coords[::-1]    # reversed, closed (first == last)


def in_hull(p, hull):
    # returns False
    # otherwise returns True if hull.find_simplex(p) >= 0
    return hull.find_simplex(p) >= 0


def make_data(config):

    n_aircraft = config["n_aircraft"]
    
    # list of possible heading options from any point
    heading_options = [i*10 for i in range(1,37)]
    # time between ticks
    tt = config["tt"]
    # aircraft speed
    speed = config["speed"]
    # wind direction
    wind_direction = config["wind_direction"]
    # wind speed
    wind_speed = config["wind_speed"]
    # tolerance for reaching exit position, dependent on speed
    # set to half the distance travelled in 1 time step (without wind)
    exit_eps = 0.5 * speed * tt
    # temperature scaling for heading sampling
    temperature = config["temperature"]
    # bias towards continuing present heading
    bias = config["bias"]
    # safe area around each aircraft
    sep_radius = config["sep_radius"]

    # sector boundaries
    x_lim = 2.5
    y_lim = 0.4

    eps = 1e-8
    exit = np.asarray((2.5-eps, 0))
    entry = np.asarray((-2.5+eps, 0))

    # rectangular boundary
    Z_boundary = np.array(((-x_lim,-y_lim),(x_lim,-y_lim),(x_lim,y_lim),(-x_lim,y_lim),(-x_lim,-y_lim)))

    # generate a list of n_aircraft traffic positions and make discs around them
    traffic_positions = [np.array([np.random.uniform(-x_lim,x_lim),np.random.uniform(-y_lim,y_lim)]) for _ in range(n_aircraft)]
    holes = [create_disc(traffic_pos, sep_radius) for traffic_pos in traffic_positions]

    # set up polygon environment
    environment = PolygonEnvironment()
    environment.store(
        Z_boundary[:-1],
        [hole[:-1] for hole in holes],
        validate=False,
    )

    Z_hull = Delaunay(Z_boundary[:-1])   # drop duplicate closing vertex
    hole_hulls = [Delaunay(np.array(hole[:-1])) for hole in holes]

    # check whether the start or finish positions are in any of the holes
    # if you are starting or finishing in a hole, don't accept this - return None
    start_in_hole = [in_hull(entry, hull) for hull in hole_hulls]
    finish_in_hole = [in_hull(exit, hull) for hull in hole_hulls]
    if any(start_in_hole):
        return None

    if any(finish_in_hole):
        return None

    # confirm that there is a possible path through the sector
    if len(shortest_route_path(entry, exit, environment)) == 0:
        return None

    # set the current position as the sector entry position
    current_pos = entry

    # make a log of the aircraft positions for the entire trajectory
    positions = [current_pos]

    # make a log of the headings followed
    headings = [0]
    # log of arrays showing the relative position of all other aircraft at each timestep
    rel_traffic_positions_list = [np.array(traffic_positions) - np.array([current_pos for _ in range(n_aircraft)])]


    # while the aircraft is further from the exit position than eps, keep selecting new headings
    while np.linalg.norm(current_pos - exit) > exit_eps:

        # list of the negative deviations from shortest track
        deviation_scores = []

        # along shortest route vector
        path = np.array(shortest_route_path(current_pos, exit, environment))
        a = path[1] - path[0]
        
        # for each possible heading, find the end position and compute the distance to the exit
        for heading in heading_options:

            # find the position that would be reached on the given heading
            next_pos = point_on_heading(current_pos, heading, speed, tt, wind_direction, wind_speed)
            # the line segment followed if the proposed heading is taken up
            seg = [current_pos, next_pos]

            # if the proposed heading will end you out of the sector
            if not in_hull(next_pos, Z_hull):
                deviation_scores.append(-math.inf)
            # if the proposed heading will end you in any of the holes
            elif any(in_hull(next_pos, hull) for hull in hole_hulls):
                deviation_scores.append(-math.inf)
            # if the proposed heading will lead to the trajectory intersecting a hole or sector boundary
            elif any(doIntersectPolygon(seg, hole) for hole in holes) or doIntersectPolygon(seg, Z_boundary):
                deviation_scores.append(-math.inf)
            # otherwise this is a valid heading to take up, so compute the correct deviation score
            else:
                # vector for the new heading path
                b = next_pos - path[0]
                # deviation angle from the shortest path
                deviation = math.degrees(Vector(a).angle_between(b))

                # check whether the candidate heading is the same as the current one
                # if it is a continuation, apply the continuation bias to the deviation score
                if heading == headings[-1]:
                    deviation_scores.append(-deviation * bias)
                else:
                    deviation_scores.append(-deviation)
        
        # if there is no move you can make (all deviation_scores are -inf)
        if len(set(deviation_scores)) == 1:
            return None

        # construct probability distribution for new heading and sample from it
        probs = sp.softmax(np.asarray(deviation_scores)/temperature)
        selected_heading = np.random.choice(heading_options, p=probs)
        headings.append(selected_heading)

        # update the current position based on following that heading for step_dist
        current_pos = point_on_heading(current_pos, selected_heading, speed, tt, wind_direction, wind_speed)
        positions.append(current_pos)

        rel_traffic_positions = np.array(traffic_positions) - np.array([current_pos for _ in range(n_aircraft)])
        rel_traffic_positions_list.append(rel_traffic_positions)

        # if the sequence will end up being longer than the context length for the model, return None
        if len(headings)>30:
            return None


    heading_tokens = headings.copy()
    heading_tokens.append(0)

    position_tokens = positions.copy()

    target_tokens = [exit for _ in range(len(position_tokens))]

    rel_pos_tokens = list(np.array(target_tokens) - np.array(position_tokens))

    abs_traffic_tokens = [np.array(traffic_positions) for _ in range(len(position_tokens))]

    rel_traffic_tokens = rel_traffic_positions_list.copy()

    trajectory_data = {
        "headings" : heading_tokens,
        "abs_positions" : position_tokens,
        "targets" : target_tokens,
        "rel_positions" : rel_pos_tokens,
        "abs_traffic" : abs_traffic_tokens,
        "rel_traffic" : rel_traffic_tokens
    }

    return trajectory_data


def make_batch(n, config):
    return [make_data(config) for _ in range(n)]


n_aircraft = config["n_aircraft"]

print("generating data...")

# generate 1000 batches of 1000 trajectories (= 1m)
batch_size = 1000
n_batches = 1000

results_batched = Parallel(n_jobs=-1, return_as="generator")(
    delayed(make_batch)(batch_size, config) for _ in range(n_batches)
)

print("separating data...")

headings_list, abs_positions_list, targets_list, rel_positions_list, abs_traffic_list, rel_traffic_list = [], [], [], [], [], []

for batch in tqdm(results_batched, total=n_batches):
    for trajectory in batch:
        if trajectory is not None:
            headings_list.append(trajectory["headings"])
            abs_positions_list.append(trajectory["abs_positions"])
            targets_list.append(trajectory["targets"])
            rel_positions_list.append(trajectory["rel_positions"])
            abs_traffic_list.append(trajectory["abs_traffic"])
            rel_traffic_list.append(trajectory["rel_traffic"])


print("padding sequences...")

max_length = len(max(headings_list, key=len))
context_length = 2 ** max_length.bit_length()

# heading_tokens:  [initial_0, h1, ..., hN, terminal_0]  →  N+2 elements
# position_tokens: [entry, p1, ..., pN]                  →  N+1 elements
# Both are padded to (context_length + 1) total elements
target_len = context_length + 1

for i in range(len(headings_list)):

    headings      = headings_list[i]
    abs_positions = abs_positions_list[i]
    targets       = targets_list[i]
    rel_positions = rel_positions_list[i]
    abs_traffic   = abs_traffic_list[i]
    rel_traffic   = rel_traffic_list[i]

    heading_pad = target_len - len(headings)       # = context_length - N - 1
    pos_pad     = target_len - len(abs_positions)  # = context_length - N

    headings_list[i]      = headings      + [-1]                               * heading_pad
    abs_positions_list[i] = abs_positions + [np.asarray((np.inf, np.inf))]     * pos_pad
    targets_list[i]       = targets       + [np.asarray((np.inf, np.inf))]     * pos_pad
    rel_positions_list[i] = rel_positions + [np.asarray((np.inf, np.inf))]     * pos_pad
    abs_traffic_list[i]   = abs_traffic   + [np.full((n_aircraft, 2), np.inf)] * pos_pad
    rel_traffic_list[i]   = rel_traffic   + [np.full((n_aircraft, 2), np.inf)] * pos_pad


print("saving data...")

with open(f"{OUTPUT_DIR}/data/headings_data.pkl", "wb") as f:
    pickle.dump(np.array(headings_list), f)

with open(f"{OUTPUT_DIR}/data/abs_positions_data.pkl", "wb") as f:
    pickle.dump(np.array(abs_positions_list), f)

with open(f"{OUTPUT_DIR}/data/targets_data.pkl", "wb") as f:
    pickle.dump(np.array(targets_list), f)

with open(f"{OUTPUT_DIR}/data/rel_positions_data.pkl", "wb") as f:
    pickle.dump(np.array(rel_positions_list), f)

with open(f"{OUTPUT_DIR}/data/abs_traffic_data.pkl", "wb") as f:
    pickle.dump(np.array(abs_traffic_list), f)

with open(f"{OUTPUT_DIR}/data/rel_traffic_data.pkl", "wb") as f:
    pickle.dump(np.array(rel_traffic_list), f)


print("complete!")