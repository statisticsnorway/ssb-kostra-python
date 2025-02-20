"""Command-line interface."""

import click


@click.command()
@click.version_option()
def main() -> None:
    """SSB Kostra Python."""


if __name__ == "__main__":
    main(prog_name="ssb-kostra-python")  # pragma: no cover
