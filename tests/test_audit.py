import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402

import agent  # noqa: E402
from audit import audit_receipt  # noqa: E402
from schema import Receipt  # noqa: E402

SAMPLES = ROOT / "sample_data"


def load(name: str) -> Receipt:
    return Receipt.model_validate(json.loads((SAMPLES / f"{name}.json").read_text()))


def codes(name):
    return {f.code for f in audit_receipt(load(name)).findings}


def test_clean_receipt_has_no_serious_findings():
    rep = audit_receipt(load("01_clean_grocery"))
    assert [f for f in rep.findings if f.severity != "low"] == []
    assert rep.overcharge_estimate == 0
    assert rep.computed_total == 11.42


def test_double_scan():
    rep = audit_receipt(load("02_double_scan"))
    assert {f.code for f in rep.findings} == {"POSSIBLE_DOUBLE_SCAN"}
    assert rep.overcharge_estimate == 7.49


def test_tax_above_texas_max():
    rep = audit_receipt(load("03_tax_too_high"))
    assert {f.code for f in rep.findings} == {"TAX_TOO_HIGH"}
    assert rep.overcharge_estimate == 0.54


def test_total_mismatch():
    rep = audit_receipt(load("04_total_mismatch"))
    assert {f.code for f in rep.findings} == {"TOTAL_MISMATCH"}
    assert rep.overcharge_estimate == 1.00


def test_double_gratuity():
    rep = audit_receipt(load("05_double_gratuity"))
    assert {f.code for f in rep.findings} == {"DOUBLE_GRATUITY"}
    assert rep.overcharge_estimate == 7.00


def test_line_math_with_coupon():
    rep = audit_receipt(load("06_line_math"))
    assert {f.code for f in rep.findings} == {"LINE_MATH"}
    assert rep.overcharge_estimate == 1.00


def test_graph_offline_clean_skips_llm():
    rep = agent.run(receipt=load("01_clean_grocery"), provider=None)
    assert rep.verdict.startswith("Looks right")


def test_graph_calls_llm_only_when_needed():
    fake = FakeListChatModel(responses=["About $1.00 at stake: you were charged $18.00 but it adds up to $17.00."])
    rep = agent.run(receipt=load("04_total_mismatch"), provider=None, llm=fake)
    assert "$1.00" in rep.verdict
