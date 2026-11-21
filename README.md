# Receipt Auditor

Agent #1 of my agent series. It doesn't just read a receipt. It checks whether you were overcharged.

**Simple output:** the line items.
**Agent output:** "About $7.00 at stake: an 18% service charge was already added, and you tipped $7.00 on top."

## How it works

```
receipt text or photo
   -> extract   (LLM + structured output -> validated Receipt JSON, numbers copied as printed)
   -> audit     (plain Python math: line totals, subtotal, tax rate cap, double scans, double tips, total)
   -> explain   (LLM writes the verdict, only when there is something to explain)
```

Built with LangGraph. The model never does the math: it only reads the receipt and explains the findings.

## Checks

| Code | What it catches |
| --- | --- |
| LINE_MATH | quantity x unit price doesn't equal the line total |
| POSSIBLE_DOUBLE_SCAN | the same item and price appear more than once |
| SUBTOTAL_MISMATCH | items (before or after coupons) don't add up to the subtotal |
| TAX_TOO_HIGH | tax rate above the Texas maximum of 8.25% |
| DOUBLE_GRATUITY | a service charge and a tip were both added |
| TOTAL_MISMATCH | the total charged isn't what everything adds up to |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add your ANTHROPIC_API_KEY
ollama pull qwen2.5:7b      # optional, for the local model

pytest -q                   # 8 tests, no API key needed
streamlit run app.py        # the demo
python compare.py           # local vs. hosted scorecard on the 6 sample receipts
```

Sample receipts in `sample_data/` are made up for testing; each has a `.json` with the correct extraction.
