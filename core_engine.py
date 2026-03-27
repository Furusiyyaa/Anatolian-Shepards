"""
core_state.py — Data-Oriented Entities & Centralised Game State

All entity definitions use ``@dataclass(slots=True)`` to avoid per-instance
``__dict__`` overhead.  No update loops or movement logic lives here — this
module is strictly a state container layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core_grid import AnatolianGrid


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class UnitState(Enum):
    """Finite-state label for a single unit."""

    IDLE = auto()
    MOVING = auto()
    HARVESTING = auto()


# ---------------------------------------------------------------------------
# Entity dataclasses
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ResourceNode:
    """An immovable resource deposit on the map.

    Attributes
    ----------
    node_id : int
        Unique identifier for this resource node.
    x : int
        Column position on the grid.
    y : int
        Row position on the grid.
    resource_type : str
        Category label (e.g. ``"SHEEP"``, ``"TIMBER"``).
    amount : int
        Remaining harvestable quantity.
    """

    node_id: int
    x: int
    y: int
    resource_type: str
    amount: int


@dataclass(slots=True)
class Unit:
    """A controllable entity belonging to a player.

    Attributes
    ----------
    unit_id : int
        Unique identifier for this unit.
    owner_id : int
        Player who controls the unit.
    x : int
        Current column position on the grid.
    y : int
        Current row position on the grid.
    state : UnitState
        Behavioural state machine label.
    current_path : list[tuple[int, int]]
        Waypoint sequence the unit is currently following.
        Empty when idle or harvesting.
    """

    unit_id: int
    owner_id: int
    x: int
    y: int
    state: UnitState = UnitState.IDLE
    current_path: list[tuple[int, int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Centralised game state
# ---------------------------------------------------------------------------
class GameState:
    """Top-level container that owns the grid and every live entity.

    Parameters
    ----------
    grid : AnatolianGrid
        The terrain matrix backing this game session.

    Attributes
    ----------
    grid : AnatolianGrid
    units : dict[int, Unit]
        Mapping of ``unit_id`` → :class:`Unit`.
    resources : dict[int, ResourceNode]
        Mapping of ``node_id`` → :class:`ResourceNode`.
    """

    __slots__ = ("grid", "units", "resources")

    def __init__(self, grid: "AnatolianGrid") -> None:
        self.grid: AnatolianGrid = grid
        self.units: dict[int, Unit] = {}
        self.resources: dict[int, ResourceNode] = {}

    # ----- mutators -------------------------------------------------------

    def add_unit(
        self,
        unit_id: int,
        owner_id: int,
        x: int,
        y: int,
        state: UnitState = UnitState.IDLE,
    ) -> Unit:
        """Create a :class:`Unit`, register it, and return it.

        Raises
        ------
        ValueError
            If ``unit_id`` is already present or the position is invalid.
        """
        if unit_id in self.units:
            raise ValueError(f"Unit with id {unit_id} already exists.")
        cost = self.grid.get_cost(x, y)
        if cost == -1:
            raise ValueError(f"Position ({x}, {y}) is out of grid bounds.")

        unit = Unit(unit_id=unit_id, owner_id=owner_id, x=x, y=y, state=state)
        self.units[unit_id] = unit
        return unit

    def add_resource(
        self,
        node_id: int,
        x: int,
        y: int,
        resource_type: str,
        amount: int,
    ) -> ResourceNode:
        """Create a :class:`ResourceNode`, register it, and return it.

        Raises
        ------
        ValueError
            If ``node_id`` is already present or the position is invalid.
        """
        if node_id in self.resources:
            raise ValueError(f"ResourceNode with id {node_id} already exists.")
        cost = self.grid.get_cost(x, y)
        if cost == -1:
            raise ValueError(f"Position ({x}, {y}) is out of grid bounds.")

        node = ResourceNode(
            node_id=node_id,
            x=x,
            y=y,
            resource_type=resource_type,
            amount=amount,
        )
        self.resources[node_id] = node
        return node

    # ----- dunder helpers --------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"GameState(grid={self.grid!r}, "
            f"units={len(self.units)}, "
            f"resources={len(self.resources)})"
        )


# ---------------------------------------------------------------------------
# Quick demo / smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from core_grid import AnatolianGrid

    grid = AnatolianGrid(64, 64)
    grid.generate_debug_terrain()

    state = GameState(grid)

    # Add a Player-1 unit near the left side of the map.
    unit = state.add_unit(unit_id=1, owner_id=1, x=10, y=32)
    print(f"Added unit:     {unit}")

    # Add a sheep resource on the steppe.
    sheep = state.add_resource(
        node_id=1, x=20, y=20, resource_type="SHEEP", amount=150,
    )
    print(f"Added resource: {sheep}")

    print(f"\nGameState: {state!r}")
    print(f"  units     = {state.units}")
    print(f"  resources = {state.resources}")
