"""Data models for a receipt and for audit findings."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: str = Field(description="Item name exactly as printed on the receipt")
    quantity: float = Field(default=1, description="Quantity purchased; 1 if not printed")
    unit_price: Optional[float] = Field(default=None, description="Price per unit if printed")
    line_total: float = Field(description="Amount charged for this line, as printed")


class Receipt(BaseModel):
    merchant: str = Field(description="Store or restaurant name")
    date: Optional[str] = Field(default=None, description="Purchase date as YYYY-MM-DD if printed")
    items: list[LineItem] = Field(description="Every purchased line item, in printed order")
    discounts: float = Field(default=0, description="Total of discounts/coupons as a positive number")
    subtotal: Optional[float] = Field(default=None, description="Subtotal as printed")
    tax: Optional[float] = Field(default=None, description="Sales tax as printed")
    service_charge: float = Field(default=0, description="Auto-gratuity or service charge as printed")
    tip: float = Field(default=0, description="Tip as printed")
    total: float = Field(description="Final total charged, as printed")


Severity = Literal["high", "medium", "low"]


class Finding(BaseModel):
    code: str
    severity: Severity
    message: str
    amount: float = 0.0  # money at stake, positive = customer likely overcharged


class AuditReport(BaseModel):
    receipt: Receipt
    findings: list[Finding]
    computed_total: float
    overcharge_estimate: float
    verdict: str = ""
