"""
BurningForest Gymnasium environment for wildfire-mitigation curriculum learning.

The agent (a firetruck) navigates an 8×8 grid from Start (S) to Goal (G),
collecting water (W) and avoiding burning tiles (B).  Tile types and their
reward indices are: S/R (road) = 0, V (vegetation) = 1, B (burning) = 2,
W (water) = 3, G (goal) = 4.
"""

from __future__ import annotations

from contextlib import closing
from io import StringIO
from os import path

import numpy as np

import gymnasium as gym
from gymnasium import Env, spaces, utils
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import seeding

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
tile_key = b"SRVBWG"
reward_index = {
    b"S": 0,
    b"R": 0,
    b"V": 1,
    b"B": 2,
    b"W": 3,
    b"G": 4,
}


MAPS = {
    "4x4": ["SVVR", "RBVB", "RVWV", "RBVG"]
}
# ["SRRV", "BBVB", "RVWV", "RBVG"]
# MAPS = {
#     "4x4": [
#     "SRRV",
#     "BRVB",
#     "RVWV",
#     "BRVG",
# ]
# }


def is_valid(board: list[list[str]], max_size: int) -> bool:
    """Return True if there is a path from S(0,0) to G(n-1,n-1) avoiding B tiles (DFS)."""
    frontier, discovered = [], set()
    frontier.append((0, 0))
    while frontier:
        r, c = frontier.pop()
        if (r, c) not in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                    continue
                if board[r_new][c_new] == "G":
                    return True
                if board[r_new][c_new] != "B":
                    frontier.append((r_new, c_new))
    return False


def generate_random_map(
    size: int = 8,
    p_road: float = 0.4,
    p_veg: float = 0.3,
    n_water: int = 1,
    seed: int | None = None,
) -> list[str]:
    """Generates a random valid map (one that has a path from start to goal,
    avoiding burning tiles).
    """
    p_burn = 1.0 - p_veg - p_road
    valid = False
    board = []  # initialize to make pyright happy
    np_random, _ = seeding.np_random(seed)

    while not valid:
        board = np_random.choice(
            ["R", "V", "B"],
            size=(size, size),
            p=[p_road, p_veg, p_burn],
        )
        board[0][0] = "S"
        board[-1][-1] = "G"
        valid = is_valid(board, size)

    interior_cells = [
        (r, c)
        for r in range(size)
        for c in range(size)
        if board[r][c] not in ("S", "G")
    ]
    water_positions = np_random.choice(
        len(interior_cells),
        size=min(n_water, len(interior_cells)),
        replace=False,
    )
    for idx in water_positions:
        r, c = interior_cells[idx]
        board[r][c] = "W"
    return ["".join(row) for row in board]


class BurningForest(Env):
    """
    Wildfire-mitigation grid-world environment.

    The agent navigates an n×n grid from S(0,0) to G(n-1,n-1).
    Collecting a water tile (W) for the first time yields a bonus reward;
    stepping on a burning tile (B) terminates the episode with a penalty.
    Observation space is Discrete(n²) when no water tiles are present,
    or Discrete(2·n²) to encode whether water has been collected.
    """
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        desc: list[str] = None,
        map_name: str = "4x4",
        is_slippery: bool = False,
        success_rate: float = 0.95,
        # reward_schedule: tuple[int, int, int, int, int, int,int ] = (0, 0, -0.02, -0.5, 0.5, 1),
        reward_schedule: tuple[int, int, int, int, int] = (0, -0.02, -0.5, 0.5, 1),
    ):
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (min(reward_schedule), max(reward_schedule))
        self.reward_schedule = reward_schedule

        nA = 4
        nS = nrow * ncol

        self.initial_state_distrib = np.array(desc == b"S").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        self._water_positions = set(
            int(s) for s in range(nS) if desc.flat[s] == b"W"
        )
        self._n_positions = nS
        self._collected_water = set()

        n_water = len(self._water_positions)
        if n_water > 0:
            self.observation_space = spaces.Discrete(nS * 2)
        else:
            self.observation_space = spaces.Discrete(nS)

        fail_rate = (1.0 - success_rate) / 2.0

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):
            new_row, new_col = inc(row, col, action)
            new_state = to_s(new_row, new_col)
            new_letter = bytes(desc[new_row, new_col])
            terminated = new_letter in (b"G", b"B")
            idx = reward_index[new_letter]
            reward = reward_schedule[idx]
            return new_state, reward, terminated

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    letter = bytes(desc[row, col])
                    if letter in (b"G", b"B"):
                        idx = reward_index[letter]
                        reward = reward_schedule[idx]
                        li.append((1.0, s, reward, True))
                    else:
                        if is_slippery:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                li.append(
                                    (
                                        success_rate if b == a else fail_rate,
                                        *update_probability_matrix(row, col, b),
                                    )
                                )
                        else:
                            li.append((1.0, *update_probability_matrix(row, col, a)))

        # self.observation_space = spaces.Discrete(nS)
        # self.observation_space = spaces.Dict(
        #     {
        #         "agent": spaces.Box(0, max(nrow, ncol) - 1, shape=(2,), dtype=np.float32),
        #         "goal": spaces.Box(0, max(nrow, ncol) - 1, shape=(2,), dtype=np.float32),
        #     }
        # )
        self.action_space = spaces.Discrete(nA)

        self.render_mode = render_mode

        # goal position
        goal_location = np.argwhere(desc == b"G")
        self._goal_location = np.array([goal_location[0][1], goal_location[0][0]], dtype=np.int64)

        # # set _collected_water
        # self._collected_water = set()

        # pygame utils
        self.window_size = (min(64 * ncol, 512), min(64 * nrow, 512))
        self.cell_size = (
            self.window_size[0] // self.ncol,
            self.window_size[1] // self.nrow,
        )
        self.window_surface = None
        self.clock = None
        self.road_img = None
        self.start_img = None
        self.goal_img = None
        self.burning_img = None
        self.water_img = None
        self.veg_img = None
        self.truck_images = None
        # self.hole_img = None
        # self.cracked_hole_img = None
        # self.ice_img = None
        # self.elf_images = None
        # self.goal_img = None
        # self.start_img = None

    # def _get_obs(self):
    #     row, col = self.s // self.ncol, self.s % self.ncol
    #     return {
    #         "agent": np.array([col, row], dtype=np.float32),
    #         "goal": self._goal_location.astype(np.float32),
    #     }
    def _get_obs(self):
        """Return the current observation, encoding water-collection state when applicable."""
        if len(self._water_positions) > 0:
            water_flag = 1 if self._collected_water else 0
            return int(self.s + self._n_positions * water_flag)
        return int(self.s)

    def step(self, a):
        """Execute action *a* and return (obs, reward, terminated, truncated, info)."""
        a = int(a)
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.s = s
        self.lastaction = a

        # Water tile: first visit gives full reward, subsequent visits give road reward
        # TODO: consider if we should change the letter from W to R?
        if self.desc.flat[s] == b"W":
            if s in self._collected_water:
                r = self.reward_schedule[0] # set to road reward
            else:
                self._collected_water.add(s)

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return self._get_obs(), r, t, False, {"prob": p}
        # return int(s), r, t, False, {"prob": p}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        """Reset the environment to the start state and return (obs, info)."""
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None
        self._collected_water = set()

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {"prob": 1}
        # return int(self.s), {"prob": 1}

    def render(self):
        """Render the environment according to render_mode ('human', 'ansi', or 'rgb_array')."""
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.render_mode == "ansi":
            return self._render_text()
        else:  # self.render_mode in {"human", "rgb_array"}:
            return self._render_gui(self.render_mode)

    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[toy-text]"`'
            ) from e

        if self.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Burning Forest")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert self.window_surface is not None, (
            "Something went wrong with pygame. This should never happen."
        )

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.road_img is None:
            file_name = path.join(path.dirname(__file__), "img/road.webp")
            self.road_img = pygame.transform.scale(pygame.image.load(file_name), self.cell_size)

        if self.start_img is None:
            file_name = path.join(path.dirname(__file__), "img/start.webp")
            self.start_img = pygame.transform.scale(pygame.image.load(file_name), self.cell_size)

        if self.goal_img is None:
            file_name = path.join(path.dirname(__file__), "img/goal.png")
            self.goal_img = pygame.transform.scale(pygame.image.load(file_name), self.cell_size)

        if self.burning_img is None:
            file_name = path.join(path.dirname(__file__), "img/burning.webp")
            self.burning_img = pygame.transform.scale(pygame.image.load(file_name), self.cell_size)

        if self.water_img is None:
            file_name = path.join(path.dirname(__file__), "img/water.png")
            self.water_img = pygame.transform.scale(pygame.image.load(file_name), self.cell_size)

        if self.veg_img is None:
            file_name = path.join(path.dirname(__file__), "img/vegetation.webp")
            self.veg_img = pygame.transform.scale(pygame.image.load(file_name), self.cell_size)

        if self.truck_images is None:
            # TODO: change the elf later
            trucks = [
                path.join(path.dirname(__file__), "img/firetruck.webp"),
                path.join(path.dirname(__file__), "img/firetruck.webp"),
                path.join(path.dirname(__file__), "img/firetruck.webp"),
                path.join(path.dirname(__file__), "img/firetruck.webp"),
            ]
            self.truck_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in trucks
            ]

        desc = self.desc.tolist()
        assert isinstance(desc, list), f"desc should be a list or an array, got {desc}"
        for y in range(self.nrow):
            for x in range(self.ncol):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                rect = (*pos, *self.cell_size)

                if desc[y][x] == b"R":
                    self.window_surface.blit(self.road_img, pos)
                if desc[y][x] == b"B":
                    self.window_surface.blit(self.burning_img, pos)
                elif desc[y][x] == b"W":
                    self.window_surface.blit(self.water_img, pos)
                elif desc[y][x] == b"V":
                    self.window_surface.blit(self.veg_img, pos)
                elif desc[y][x] == b"G":
                    self.window_surface.blit(self.goal_img, pos)
                elif desc[y][x] == b"S":
                    self.window_surface.blit(self.start_img, pos)

                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)

        # paint the truck
        bot_row, bot_col = self.s // self.ncol, self.s % self.ncol
        cell_rect = (bot_col * self.cell_size[0], bot_row * self.cell_size[1])
        last_action = self.lastaction if self.lastaction is not None else 1
        truck_img = self.truck_images[last_action]

        if desc[bot_row][bot_col] == b"B":
            self.window_surface.blit(self.burning_img, cell_rect)
        else:
            self.window_surface.blit(truck_img, cell_rect)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )

    @staticmethod
    def _center_small_rect(big_rect, small_dims):
        offset_w = (big_rect[2] - small_dims[0]) / 2
        offset_h = (big_rect[3] - small_dims[1]) / 2
        return (
            big_rect[0] + offset_w,
            big_rect[1] + offset_h,
        )

    def _render_text(self):
        desc = self.desc.tolist()
        outfile = StringIO()

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write(f"  ({['Left', 'Down', 'Right', 'Up'][self.lastaction]})\n")
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        with closing(outfile):
            return outfile.getvalue()

    def close(self):
        """Shut down the pygame window if it was opened."""
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()


# Elf and stool from https://franuka.itch.io/rpg-snow-tileset
# All other assets by Mel Tillery http://www.cyaneus.com/