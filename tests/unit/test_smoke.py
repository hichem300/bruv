from typer.testing import CliRunner

from bruv.cli import app


def test_help_works_without_credentials() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "no yap, only fax" in result.stdout
