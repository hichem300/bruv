import json
import re
from pathlib import Path

CONTRACT_FIXTURES = {
    "typesafe.md": "tests/fixtures/typesafe/success.json",
    "simple-jev.md": "tests/fixtures/simple_jev/success.json",
}


def test_contract_notes_and_fixtures_have_provenance() -> None:
    for document, fixture in CONTRACT_FIXTURES.items():
        text = Path("docs/provider-contracts", document).read_text()

        for marker in ("Source", "Accessed", "Version/commit"):
            values = re.findall(rf"^{re.escape(marker)}:\s*(.+)$", text, re.MULTILINE)
            assert values and all(value.strip() for value in values)

        assert re.findall(r"^Accessed: .+$", text, re.MULTILINE) == ["Accessed: 2026-09-20"]

        payload = json.loads(Path(fixture).read_text())
        assert isinstance(payload, dict)
        assert re.search(rf"`{re.escape(fixture)}`[^\n]*\bsynthetic\b", text, re.IGNORECASE)
