"""Deterministic receipt checks. No AI here: the math has to be exact and testable."""
from __future__ import annotations

from collections import Counter

from schema import AuditReport, Finding, Receipt

CENT = 0.01
TEXAS_MAX_TAX_RATE = 0.0825  # state 6.25% + local up to 2%


def _r(x: float) -> float:
    return round(x + 0.0, 2)


def audit_receipt(r: Receipt, max_tax_rate: float = TEXAS_MAX_TAX_RATE) -> AuditReport:
    findings: list[Finding] = []
    items_sum = _r(sum(i.line_total for i in r.items))

    # 1. Each line: quantity x unit price should equal the line total
    for i in r.items:
        if i.unit_price is not None:
            expected = _r(i.quantity * i.unit_price)
            diff = _r(i.line_total - expected)
            if abs(diff) > CENT:
                findings.append(Finding(
                    code="LINE_MATH", severity="high", amount=max(diff, 0),
                    message=f"'{i.description}': {i.quantity:g} x ${i.unit_price:.2f} = ${expected:.2f}, "
                            f"but the line says ${i.line_total:.2f}."))

    # 2. Possible double scans: identical single-quantity lines
    keys = Counter((i.description.strip().lower(), i.line_total) for i in r.items if i.quantity == 1)
    for (desc, price), n in keys.items():
        if n > 1:
            findings.append(Finding(
                code="POSSIBLE_DOUBLE_SCAN", severity="medium", amount=_r(price * (n - 1)),
                message=f"'{desc}' at ${price:.2f} appears {n} times. Check whether you bought {n}."))

    # 3. Subtotal should match the items (printed before or after discounts)
    if r.subtotal is not None:
        candidates = (items_sum, _r(items_sum - r.discounts))
        if all(abs(r.subtotal - c) > CENT for c in candidates):
            diff = _r(r.subtotal - candidates[1])
            findings.append(Finding(
                code="SUBTOTAL_MISMATCH", severity="high", amount=max(diff, 0),
                message=f"Items add up to ${items_sum:.2f} (${candidates[1]:.2f} after discounts), "
                        f"but the subtotal says ${r.subtotal:.2f}."))

    # 4. Tax rate above the legal maximum
    base = _r(items_sum - r.discounts)
    if r.tax and base > 0:
        rate = r.tax / base
        if rate > max_tax_rate + 0.0005:
            extra = _r(r.tax - base * max_tax_rate)
            findings.append(Finding(
                code="TAX_TOO_HIGH", severity="medium", amount=extra,
                message=f"Tax is {rate:.2%} of ${base:.2f}; the maximum is {max_tax_rate:.2%}. "
                        f"About ${extra:.2f} too much, even if every item were taxable."))

    # 5. Service charge and tip both applied
    if r.service_charge > 0 and r.tip > 0:
        findings.append(Finding(
            code="DOUBLE_GRATUITY", severity="medium", amount=_r(r.tip),
            message=f"A ${r.service_charge:.2f} service charge was added and a ${r.tip:.2f} tip too. "
                    f"You may have tipped twice."))

    # 6. Printed total vs what the lines add up to
    computed = _r(base + (r.tax or 0) + r.service_charge + r.tip)
    diff_total = _r(r.total - computed)
    if abs(diff_total) > CENT:
        findings.append(Finding(
            code="TOTAL_MISMATCH", severity="high", amount=max(diff_total, 0),
            message=f"Everything adds up to ${computed:.2f}, but you were charged ${r.total:.2f} "
                    f"({'+' if diff_total > 0 else ''}{diff_total:.2f})."))

    if r.subtotal is None or r.tax is None:
        findings.append(Finding(code="MISSING_FIELDS", severity="low",
                                message="No subtotal or tax line found, so some checks were skipped."))

    # Money at stake, without double counting: a total mismatch already includes the line and subtotal errors
    if diff_total > CENT:
        over = diff_total
    else:
        over = sum(f.amount for f in findings if f.code in ("LINE_MATH", "SUBTOTAL_MISMATCH"))
    over += sum(f.amount for f in findings if f.code in ("POSSIBLE_DOUBLE_SCAN", "TAX_TOO_HIGH", "DOUBLE_GRATUITY"))

    order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: (order[f.severity], -f.amount))
    return AuditReport(receipt=r, findings=findings, computed_total=computed, overcharge_estimate=_r(over))
