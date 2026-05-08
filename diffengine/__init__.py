from .version import __version__


def cli() -> None:
    from .entry_point import cli as _cli

    _cli()


__all__ = ["__version__", "cli"]
