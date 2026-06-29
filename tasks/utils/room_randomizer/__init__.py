"""Room randomizer package for pick and place tasks."""

__all__ = ["randomize_pickplace_room_layout"]


def __getattr__(name: str):
    if name == "randomize_pickplace_room_layout":
        from .room_events import randomize_pickplace_room_layout

        return randomize_pickplace_room_layout
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
