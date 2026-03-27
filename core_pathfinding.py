"""
core_pathfinding.py — Weighted A* Pathfinding for AnatolianGrid

Uses a heapq-backed min-heap for the open set and respects per-cell
terrain movement costs.  Diagonal movement is permitted at √2 × the
destination cell's base cost.  Cells that are out-of-bounds or marked
IMPASSABLE_RIVER (255) are excluded from neighbour expansion.
"""

from __future__ import annotations

import heapq
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core_grid import AnatolianGrid

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SQRT2: float = math.sqrt(2)
_IMPASSABLE: int = 255

# 8-directional offsets: (dx, dy, is_diagonal)
_DIRECTIONS: tuple[tuple[int, int, bool], ...] = (
    ( 0, -1, False),  # N
    ( 1,  0, False),  # E
    ( 0,  1, False),  # S
    (-1,  0, False),  # W
    ( 1, -1, True),   # NE
    ( 1,  1, True),   # SE
    (-1,  1, True),   # SW
    (-1, -1, True),   # NW
)


# ---------------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------------
def _octile_distance(x0: int, y0: int, x1: int, y1: int) -> float:
    """Octile distance — the optimal heuristic for 8-direction grids.

    Consistent and admissible: it never over-estimates the true cost
    when diagonal movement costs √2 and orthogonal movement costs 1.

    Parameters
    ----------
    x0, y0 : int
        Source coordinates.
    x1, y1 : int
        Goal coordinates.

    Returns
    -------
    float
        Estimated minimum traversal cost.
    """
    dx: int = abs(x1 - x0)
    dy: int = abs(y1 - y0)
    return float(dx + dy) + (_SQRT2 - 2.0) * min(dx, dy)


# ---------------------------------------------------------------------------
# A* search
# ---------------------------------------------------------------------------
def find_path(
    grid: "AnatolianGrid",
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    """Compute the lowest-cost path between two grid coordinates.

    Uses a weighted A* algorithm with an octile-distance heuristic.
    Terrain costs from ``grid.get_cost()`` are used as edge weights.
    Impassable cells (cost 255) and out-of-bounds cells (cost -1)
    are never expanded.

    Parameters
    ----------
    grid : AnatolianGrid
        The terrain matrix to search.
    start : tuple[int, int]
        ``(x, y)`` origin coordinate.
    goal : tuple[int, int]
        ``(x, y)`` destination coordinate.

    Returns
    -------
    list[tuple[int, int]]
        Ordered list of ``(x, y)`` waypoints from *start* to *goal*
        (inclusive).  Returns an empty list when no viable path exists.
    """
    sx, sy = start
    gx, gy = goal

    # --- Quick-reject impossible endpoints --------------------------------
    start_cost: int = grid.get_cost(sx, sy)
    goal_cost: int = grid.get_cost(gx, gy)
    if start_cost in (-1, _IMPASSABLE) or goal_cost in (-1, _IMPASSABLE):
        return []

    # Trivial case
    if start == goal:
        return [start]

    # --- Data structures ---------------------------------------------------
    # open_heap entries: (f_score, tie_breaker, (x, y))
    # The tie-breaker is a monotonically increasing counter that ensures
    # FIFO ordering among nodes with identical f-scores and avoids
    # comparing tuples directly (which would fail on equal coordinates).
    counter: int = 0
    open_heap: list[tuple[float, int, tuple[int, int]]] = []
    heapq.heappush(open_heap, (0.0, counter, start))

    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {start: 0.0}

    # Closed set — once a node is popped from the heap it is finalised.
    closed: set[tuple[int, int]] = set()

    # --- Main loop ---------------------------------------------------------
    while open_heap:
        _f, _c, current = heapq.heappop(open_heap)
        cx, cy = current

        if current == goal:
            # Reconstruct the path by walking came_from backwards.
            path: list[tuple[int, int]] = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        if current in closed:
            continue
        closed.add(current)

        current_g: float = g_score[current]

        for dx, dy, is_diag in _DIRECTIONS:
            nx: int = cx + dx
            ny: int = cy + dy
            neighbour: tuple[int, int] = (nx, ny)

            if neighbour in closed:
                continue

            cell_cost: int = grid.get_cost(nx, ny)
            if cell_cost == -1 or cell_cost == _IMPASSABLE:
                continue

            # Edge weight = destination cell cost × distance multiplier.
            move_cost: float = cell_cost * (_SQRT2 if is_diag else 1.0)
            tentative_g: float = current_g + move_cost

            if tentative_g < g_score.get(neighbour, math.inf):
                g_score[neighbour] = tentative_g
                f_score: float = tentative_g + _octile_distance(nx, ny, gx, gy)
                came_from[neighbour] = current
                counter += 1
                heapq.heappush(open_heap, (f_score, counter, neighbour))

    # Exhausted the search space with no path found.
    return []


# ---------------------------------------------------------------------------
# Quick demo / smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from core_grid import AnatolianGrid, TERRAIN_COSTS

    SIZE: int = 64
    grid = AnatolianGrid(SIZE, SIZE)
    grid.generate_debug_terrain()

    start = (10, 32)
    goal = (50, 32)

    print(f"Grid:  {grid!r}")
    print(f"Start: {start}  (cost {grid.get_cost(*start)})")
    print(f"Goal:  {goal}  (cost {grid.get_cost(*goal)})")
    print(f"Terrain costs: {TERRAIN_COSTS}\n")

    path = find_path(grid, start, goal)

    if path:
        print(f"Path found — {len(path)} waypoints:")
        for i, (x, y) in enumerate(path):
            cost = grid.get_cost(x, y)
            print(f"  [{i:3d}]  ({x:3d}, {y:3d})  cost={cost}")
    else:
        print("No path found.")
