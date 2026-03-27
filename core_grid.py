"""
core_grid.py — Headless Server Terrain Matrix

A data-oriented, NumPy-backed terrain grid designed for RTS simulation backends.
Uses contiguous np.uint8 arrays for CPU cache efficiency and minimal memory footprint.
No rendering or visual logic is included.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Terrain cost constants
# ---------------------------------------------------------------------------
TERRAIN_COSTS: dict[str, int] = {
    "STEPPE": 1,
    "FOOTHILLS": 3,
    "TAURUS_MOUNTAINS": 5,
    "IMPASSABLE_RIVER": 255,
}


class AnatolianGrid:
    """A fixed-size 2-D terrain matrix backed by a NumPy uint8 array.

    Parameters
    ----------
    width : int
        Number of columns in the grid.
    height : int
        Number of rows in the grid.
    default_cost : int, optional
        The base movement cost written to every cell on initialisation.
        Defaults to ``TERRAIN_COSTS["STEPPE"]`` (1).

    Attributes
    ----------
    width : int
    height : int
    grid : numpy.ndarray
        Shape ``(height, width)``, dtype ``np.uint8``.
    """

    __slots__ = ("width", "height", "grid")

    def __init__(
        self,
        width: int,
        height: int,
        default_cost: int = TERRAIN_COSTS["STEPPE"],
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Grid dimensions must be positive integers, got "
                f"width={width}, height={height}."
            )

        self.width: int = width
        self.height: int = height

        # Single contiguous block of uint8 — cache-friendly and compact.
        self.grid: np.ndarray = np.full(
            (height, width),
            fill_value=default_cost,
            dtype=np.uint8,
        )

    # ------------------------------------------------------------------
    # Fast coordinate query
    # ------------------------------------------------------------------
    def get_cost(self, x: int, y: int) -> int:
        """Return the movement cost at grid coordinate ``(x, y)``.

        Boundary checking is performed with a branchless comparison so
        out-of-bounds queries return ``-1`` instead of raising an exception.

        Parameters
        ----------
        x : int
            Column index (0-based).
        y : int
            Row index (0-based).

        Returns
        -------
        int
            The uint8 terrain cost at ``(x, y)``, or ``-1`` if the
            coordinate lies outside the grid boundaries.
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            return -1

        return int(self.grid[y, x])

    # ------------------------------------------------------------------
    # Debug / test helpers
    # ------------------------------------------------------------------
    def generate_debug_terrain(self) -> None:
        """Inject a hardcoded mountain range and river into the grid centre.

        The mountain range is a horizontal stripe of ``TAURUS_MOUNTAINS`` (5)
        placed at the vertical midpoint.  The river is a vertical stripe of
        ``IMPASSABLE_RIVER`` (255) placed at the horizontal midpoint.

        This method is intended exclusively for quick visual / unit-test
        verification and should **not** be called in production paths.
        """
        mid_y: int = self.height // 2
        mid_x: int = self.width // 2

        mountain_cost: int = TERRAIN_COSTS["TAURUS_MOUNTAINS"]
        river_cost: int = TERRAIN_COSTS["IMPASSABLE_RIVER"]

        # --- Mountain range: horizontal line centred on mid_y ---------------
        mountain_half_len: int = min(self.width // 4, 10)
        x_start: int = max(mid_x - mountain_half_len, 0)
        x_end: int = min(mid_x + mountain_half_len, self.width)
        self.grid[mid_y, x_start:x_end] = mountain_cost

        # --- River: vertical line centred on mid_x --------------------------
        river_half_len: int = min(self.height // 4, 10)
        y_start: int = max(mid_y - river_half_len, 0)
        y_end: int = min(mid_y + river_half_len, self.height)
        self.grid[y_start:y_end, mid_x] = river_cost

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"AnatolianGrid(width={self.width}, height={self.height}, "
            f"dtype={self.grid.dtype}, "
            f"memory={self.grid.nbytes} bytes)"
        )
