"""hexo.did.science import: normalization against captured fixtures, the
free-edit fallback, corrupt-record refusal, and the HTTP route (fetch mocked —
the suite never talks to the community site)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from showcase import lab_import
from showcase.lab_import import LabImportError, import_position

from test_showcase_api import fresh_ip

DATA = Path(__file__).resolve().parent
SANDBOX = json.loads((DATA / "didscience_sandbox_1ciyxmx.json").read_text(encoding="utf-8"))
GAME = json.loads((DATA / "didscience_game_c50b8f05.json").read_text(encoding="utf-8"))


def test_sandbox_fixture_imports_as_a_sequence():
    out = import_position("sandbox", "1ciyxmx", fetch=lambda k, i: SANDBOX)
    assert out["source"] == "hexo.did.science"
    assert out["kind"] == "sandbox" and out["id"] == "1ciyxmx"
    assert out["name"] == "Game Analysis"
    assert "stones" not in out
    assert len(out["moves"]) == 68
    assert out["moves"][0] == [0, 0]
    assert out["terminal"] is False


def test_game_fixture_imports_with_its_winning_move():
    out = import_position("game", "c50b8f05-0cc2-4c5f-bd9e-fc435812b050",
                          fetch=lambda k, i: GAME)
    assert out["name"] == "Guest 4B82 vs Guest F828"
    assert len(out["moves"]) == 39
    assert out["moves"][0] == [0, 0]
    assert out["moves"][-1] == [11, -2]
    assert out["terminal"] is True, "the record ends on the six-in-a-row placement"


def test_out_of_parity_sandbox_falls_back_to_free_edit():
    doc = json.loads(json.dumps(SANDBOX))
    # Swap one owner: the record no longer follows the 1-2-2 structure.
    doc["gamePosition"]["cells"][1]["player"] = "player-1"
    doc["gamePosition"]["cells"][3]["player"] = "player-2"
    out = import_position("sandbox", "1ciyxmx", fetch=lambda k, i: doc)
    assert "moves" not in out
    assert out["to_move"] == 0  # currentTurnPlayer: player-1
    n = len(out["stones"]["p0"]) + len(out["stones"]["p1"])
    assert n == 68


def test_corrupt_game_record_is_refused_not_guessed():
    doc = json.loads(json.dumps(GAME))
    doc["moves"][1]["playerId"] = doc["moves"][0]["playerId"]
    with pytest.raises(LabImportError, match="turn structure"):
        import_position("game", "x", fetch=lambda k, i: doc)


def test_bad_ids_and_kinds_are_rejected_before_any_fetch():
    def boom(kind, i):
        raise AssertionError("must not fetch")
    with pytest.raises(LabImportError, match="kind"):
        import_position("profile", "abc", fetch=boom)
    with pytest.raises(LabImportError, match="not a"):
        import_position("sandbox", "../etc", fetch=boom)
    with pytest.raises(LabImportError, match="not a"):
        import_position("sandbox", "", fetch=boom)


def test_import_route_normalizes_and_maps_errors(client, monkeypatch):
    docs = {"sandbox": SANDBOX, "game": GAME}
    monkeypatch.setattr(lab_import, "fetch_source", lambda kind, i: docs[kind])
    resp = client.get("/api/lab/import/didscience?kind=sandbox&id=1ciyxmx",
                      headers=fresh_ip())
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["moves"]) == 68
    resp = client.get("/api/lab/import/didscience?kind=game&id=c50b8f05-0cc2-4c5f-bd9e-fc435812b050",
                      headers=fresh_ip())
    assert resp.status_code == 200, resp.text
    assert resp.json()["terminal"] is True

    def missing(kind, i):
        raise LabImportError("hexo.did.science has no sandbox 'nope'", 404)
    monkeypatch.setattr(lab_import, "fetch_source", missing)
    resp = client.get("/api/lab/import/didscience?kind=sandbox&id=nope",
                      headers=fresh_ip())
    assert resp.status_code == 404
    resp = client.get("/api/lab/import/didscience?kind=bogus&id=nope",
                      headers=fresh_ip())
    assert resp.status_code == 422
