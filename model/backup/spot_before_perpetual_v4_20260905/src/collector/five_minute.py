"""Compatibility entry point; the active dataset is now one-minute Spot data."""

from collector.one_minute import collect, main


if __name__ == "__main__":
    main()
