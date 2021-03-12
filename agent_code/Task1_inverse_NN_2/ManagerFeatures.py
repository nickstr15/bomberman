import torch
import numpy as np
from collections import deque
import bisect

STEP = np.array([[1,0], [-1,0], [0,1], [0,-1]])

def state_to_features(game_state: dict) -> np.array:

    max_len_wanted_fields = 9 # only the coins

    # at the beginning and the end:

    if game_state is None:
        return None

    def possible_neighbors(pos):
        result = []
        for new_pos in (pos + STEP):
            if game_state["field"][new_pos[0], new_pos[1]] == 0:
                result.append(new_pos.tolist())

        return result

    player_pos = np.array(game_state["self"][3])
    wanted_fields = np.array(game_state["coins"])
    if len(wanted_fields) == 0:
        return torch.tensor([1,1,1,1]).float().unsqueeze(0)
    # if the len of wanted fields changes, we receive an error
    # => fill it with not reachable entries (e.g. [16,16]) and shuffle afterward to prevent a bias.
    fake_entries = []
    for _ in range(max_len_wanted_fields - len(wanted_fields)):
        fake_entries.append([16,16])
    if len(fake_entries) != 0:
        wanted_fields = np.append(wanted_fields, fake_entries, axis=0)
        np.random.shuffle(wanted_fields) # prevent a bias by having the fake entries always on the end.
        # all of the coin fields should have the same influence since the order in game_state is arbitrary

    possible_next_pos = possible_neighbors(player_pos)
    features = []
    for pos in (player_pos + STEP):
        new_distances = np.empty(len(wanted_fields))
        pos = pos.tolist()

        if pos not in possible_next_pos:
            features = np.append(features, -1)
            continue

        new_distances.fill(np.inf) # if no way can be found we consider the distance to be infinite

        # analyse the change of the distances of the shortest paths to all coins if we do a STEP
        visited = [player_pos.tolist()]
        q = deque()
        q.append([pos, 1])

        while len(q) != 0:

            pos, distance = q.popleft()
            if pos in visited:
                continue
            visited.append(pos)

            new_distances[np.argwhere((wanted_fields==pos).all(axis=1))] = distance
            assert sum((wanted_fields==pos).all(axis=1)) <= 1
            neighbors = possible_neighbors(pos)
            for node in neighbors:              
                q.append([node, distance+1])

        features = np.append(features, sum(1/new_distances**2))
    features = torch.from_numpy(features).float()
    return features.unsqueeze(0)
