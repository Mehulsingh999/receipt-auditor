"""Streamlit demo: streamlit run app.py"""
import json
import pathlib
import tempfile

import streamlit as st
from dotenv import load_dotenv

import agent
from schema import Receipt

load_dotenv()
SAMPLES = pathlib.Path(__file__).parent / "sample_data"

st.set_page_config(page_title="Receipt Auditor", page_icon="🧾")
st.title("Receipt Auditor")
st.caption("Doesn't just read your receipt. Checks whether you were overcharged.")

provider = st.sidebar.radio("Model", ["anthropic", "ollama"],
                            format_func=lambda p: "Claude (hosted)" if p == "anthropic" else "Ollama (local)")
mode = st.radio("Input", ["Sample receipt", "Paste text", "Upload photo"], horizontal=True)

text, image_path = None, None
if mode == "Sample receipt":
    name = st.selectbox("Sample", sorted(p.stem for p in SAMPLES.glob("*.txt")))
    text = (SAMPLES / f"{name}.txt").read_text()
    st.code(text)
elif mode == "Paste text":
    text = st.text_area("Receipt text", height=220)
else:
    up = st.file_uploader("Receipt photo", type=["jpg", "jpeg", "png", "webp"])
    if up:
        st.image(up, width=320)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=pathlib.Path(up.name).suffix)
        tmp.write(up.getvalue())
        image_path = tmp.name

if st.button("Audit", type="primary", disabled=not (text or image_path)):
    with st.spinner("Extracting and checking..."):
        try:
            rep = agent.run(text=text, image_path=image_path, provider=provider)
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
            st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Charged", f"${rep.receipt.total:.2f}")
    c2.metric("Adds up to", f"${rep.computed_total:.2f}")
    c3.metric("Money at stake", f"${rep.overcharge_estimate:.2f}")

    st.subheader("Verdict")
    st.write(rep.verdict)

    st.subheader("Findings")
    if not rep.findings:
        st.success("No issues found.")
    for f in rep.findings:
        icon = {"high": "🔴", "medium": "🟠", "low": "⚪"}[f.severity]
        st.write(f"{icon} **{f.code}**: {f.message}")

    with st.expander("Extracted data (structured output)"):
        st.json(json.loads(rep.receipt.model_dump_json()))
