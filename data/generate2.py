import math
import pickle
import numpy as np
from tqdm import tqdm
import scipy.special as sp
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from extremitypathfinder import PolygonEnvironment
# from skspatial.objects import Vector
# from scipy.spatial import Delaunay, ConvexHull
from joblib import Parallel, delayed
# import torch
import sys
import os

OUTPUT_DIR = "/home/je540/multi"

# Add the project root to Python's search path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.config import config

# geometry constants (true for every trajectory)

# sector bounaries 
x_lim, y_lim = 2.5, 0.4
_eps = 1e-8
exit  = np.asarray((x_lim - _eps, 0.0))
entry = np.asarray((-x_lim + _eps, 0.0))
Z_boundary = np.array((
    (-x_lim, -y_lim), (x_lim, -y_lim),
    (x_lim,  y_lim ), (-x_lim, y_lim),
    (-x_lim, -y_lim),
))

# config-derived constants

tt             = config["tt"]
speed          = config["speed"]
sep_radius     = config["sep_radius"]
sep_radius_sq  = sep_radius ** 2
exit_eps       = 0.5 * speed * tt
exit_eps_sq    = exit_eps ** 2

heading_options = [i*10 for i in range(1, 37)]
heading_arr     = np.array(heading_options)
_angles_rad     = np.radians(90 - heading_arr)
_heading_dirs   = np.stack([np.cos(_angles_rad), np.sin(_angles_rad)], 1)  # (36, 2)
_wind_angle     = -math.radians(config["wind_direction"] + 90)
_wind_v         = config["wind_speed"] * np.array([math.cos(_wind_angle), math.sin(_wind_angle)])

step_v          = (config["speed"] * _heading_dirs + _wind_v) * config["tt"]   # (36, 2)
disp_angles_rad = np.arctan2(step_v[:, 1], step_v[:, 0])               # (36,) — also constant


# ------------ geometry ------------


def create_disc(centre, radius):
    centre_x, centre_y = centre[0], centre[1]
    angles = np.linspace(0, 2 * np.pi, 21)
    coords = np.stack([
        centre_x + radius * np.cos(angles),
        centre_y + radius * np.sin(angles),
    ], axis=1)
    return coords[::-1]    # reversed, closed (first == last)


def in_sector(p):
    return (
        (abs(p[0]) <= x_lim and abs(p[1]) <= y_lim)
        or ((p[0] - exit[0])**2 + (p[1] - exit[1])**2 <= exit_eps_sq)
    )


def in_sector_batch(P):                                            # P: (k, 2)
    in_rect = (np.abs(P[:, 0]) <= x_lim) & (np.abs(P[:, 1]) <= y_lim)
    in_exit = ((P - exit) ** 2).sum(1) <= exit_eps_sq
    return in_rect | in_exit                                       # (k,)


def in_any_hole(p, traffic_arr):
    diff = traffic_arr - p
    return ((diff * diff).sum(1) <= sep_radius_sq).any()


def in_any_hole_batch(P, traffic_arr):                             # P: (k, 2)
    diff = P[:, None, :] - traffic_arr[None, :, :]                 # (k, n, 2) broadcasting
    return ((diff * diff).sum(-1) <= sep_radius_sq).any(1)         # (k,)


# ------------ other ------------


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


def in_hull(p, hull):
    # returns False
    # otherwise returns True if hull.find_simplex(p) >= 0
    return hull.find_simplex(p) >= 0


def in_region(p, polygon):
    return polygon.covers(Point(p))


# ------------ generate data ------------

def _make_data(config):
    n_aircraft  = config["n_aircraft"]
    temperature = config["temperature"]
    bias        = config["bias"]

    # traffic positions as one (n, 2) array
    traffic_arr = np.random.uniform(
        low=[-x_lim, -y_lim], high=[x_lim, y_lim], size=(n_aircraft, 2)
    )
    holes = [create_disc(pos, sep_radius) for pos in traffic_arr]

    # sector boundary union — still needed for environment.store
    # and the segment-vs-boundary intersection check
    exit_circle      = create_disc(exit, exit_eps)
    sector_union     = unary_union([Polygon(Z_boundary), Polygon(exit_circle)])
    Z_boundary_union = np.array(sector_union.exterior.coords)

    environment = PolygonEnvironment()
    environment.store(
        Z_boundary_union[:-1],
        [hole[:-1] for hole in holes],
        validate=False,
    )

    # discard if entry/exit overlap a hole, or no path exists
    if in_any_hole(entry, traffic_arr):
        return None
    if in_any_hole(exit, traffic_arr):
        return None
    if len(shortest_route_path(entry, exit, environment)) == 0:
        return None

    # initialise trajectory
    current_pos = entry
    positions   = [current_pos]
    headings    = [0]
    rel_traffic_positions_list = [traffic_arr - current_pos]

    while np.linalg.norm(current_pos - exit) > exit_eps:

        path = np.array(shortest_route_path(current_pos, exit, environment))
        if len(path) < 2:
            return None
        a = path[1] - path[0]

        # all 36 candidate next positions in one numpy op
        next_positions = current_pos + step_v                       # (36, 2)

        # cheap vectorised containment checks
        valid_mask = (
            in_sector_batch(next_positions)
            & ~in_any_hole_batch(next_positions, traffic_arr)
        )

        # expensive per-segment intersection check, only on survivors
        for i in np.where(valid_mask)[0]:
            seg = [current_pos, next_positions[i]]
            if (any(doIntersectPolygon(seg, h) for h in holes)
                or doIntersectPolygon(seg, Z_boundary_union)):
                valid_mask[i] = False

        if not valid_mask.any():
            return None

        # all 36 deviation angles in one shot
        angle_a  = math.atan2(a[1], a[0])
        delta    = (disp_angles_rad - angle_a + math.pi) % (2 * math.pi) - math.pi
        devs_deg = np.abs(np.degrees(delta))

        # continuation bias, then -inf out the rejected candidates
        bias_mult        = np.where(heading_arr == headings[-1], bias, 1.0)
        deviation_scores = -devs_deg * bias_mult
        deviation_scores[~valid_mask] = -math.inf

        # sample by index — avoids a lookup back to the heading value
        probs        = sp.softmax(deviation_scores / temperature)
        selected_idx = np.random.choice(36, p=probs)
        headings.append(heading_options[selected_idx])

        # advance position by indexing into precomputed step_v
        current_pos = current_pos + step_v[selected_idx]
        positions.append(current_pos)
        rel_traffic_positions_list.append(traffic_arr - current_pos)

        if len(headings) > 30:
            return None

    # package tokens
    heading_tokens     = headings + [0]
    position_tokens    = positions
    target_tokens      = [exit] * len(position_tokens)
    rel_pos_tokens     = list(np.asarray(target_tokens) - np.asarray(position_tokens))
    abs_traffic_tokens = [traffic_arr] * len(position_tokens)
    rel_traffic_tokens = rel_traffic_positions_list

    return {
        "headings":      heading_tokens,
        "abs_positions": position_tokens,
        "targets":       target_tokens,
        "rel_positions": rel_pos_tokens,
        "abs_traffic":   abs_traffic_tokens,
        "rel_traffic":   rel_traffic_tokens,
    }

# def make_data(config):
#     try:
#         return _make_data(config)   # rename current function to _make_data
#     except Exception:
#         return None



def make_batch(n, config):
    # change to make_data!
    return [_make_data(config) for _ in range(n)]


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
    abs_positions_list[i] = abs_positions + [np.asarray((0, 0))]     * pos_pad
    targets_list[i]       = targets       + [np.asarray((0, 0))]     * pos_pad
    rel_positions_list[i] = rel_positions + [np.asarray((0, 0))]     * pos_pad
    abs_traffic_list[i]   = abs_traffic   + [np.full((n_aircraft, 2), 0)] * pos_pad
    rel_traffic_list[i]   = rel_traffic   + [np.full((n_aircraft, 2), 0)] * pos_pad


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