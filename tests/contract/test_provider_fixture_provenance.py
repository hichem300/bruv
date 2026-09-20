from pathlib import Path


def test_contract_notes_pin_source_and_access_date() -> None:
    for name in ("typesafe.md", "simple-jev.md"):
        text = Path("docs/provider-contracts", name).read_text()
        assert "Source:" in text
        assert "Accessed: 2026-09-20" in text
        assert "Version/commit:" in text
