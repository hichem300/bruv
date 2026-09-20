import typer

app = typer.Typer(help="no yap, only fax")


@app.callback()
def root() -> None:
    """no yap, only fax"""


def main() -> None:
    app()
