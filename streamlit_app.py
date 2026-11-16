"""Streamlit front end + model trainer. `streamlit run streamlit_app.py` also starts the scoring API on port 5000."""
import random
import socket
import threading

import pandas as pd
import streamlit as st

import model

st.set_page_config(page_title="LR-DDoS Python Backend", layout="wide")


@st.cache_resource
def start_api(port=5000):
    """Run the Flask API in a background thread (once) so PHP can reach it."""
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            return f"already running on port {port}"
    import api
    threading.Thread(target=lambda: api.app.run(host="127.0.0.1", port=port, use_reloader=False), daemon=True).start()
    return f"started on http://127.0.0.1:{port}"


st.title("Low-Rate DDoS Detection – Python backend")
st.caption(f"Scoring API for the PHP dashboard: {start_api()}  (PHP works without this; it just upgrades the engine.)")

tab_train, tab_test, tab_link = st.tabs(["Train model", "Test a traffic window", "Connect to PHP"])

with tab_train:
    src = st.radio("Training data", ["Built-in synthetic traffic", "Upload CSV (features + label)"], horizontal=True)
    c1, c2 = st.columns(2)
    n_est = c1.slider("Trees (n_estimators)", 50, 500, 200, 50)
    depth = c2.slider("Max depth (0 = unlimited)", 0, 30, 0)
    df = None
    if src.startswith("Built-in"):
        n = st.slider("Windows to generate", 500, 10000, 2000, 500)
        if st.button("Train", type="primary"):
            df = model.synthetic_dataset(n)
    else:
        up = st.file_uploader("CSV with columns: " + ", ".join(model.FEATURES) + ", label (0 = normal, 1 = attack)", type="csv")
        st.caption("Export one from the PHP Settings page, or build it from CIC-DDoS2019 / BCCC-cPacket-Cloud-DDoS-2024 after feature extraction.")
        if up is not None and st.button("Train", type="primary"):
            df = pd.read_csv(up)
            missing = [c for c in model.FEATURES + ["label"] if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}"); df = None
    if df is not None:
        with st.spinner("Training..."):
            _, m = model.train(df, n_est, depth or None)
        st.success("Model trained and saved. The API now serves it.")
        a, b, c, d, e = st.columns(5)
        a.metric("Accuracy", f"{m['accuracy']:.2%}"); b.metric("Precision", f"{m['precision']:.2%}")
        c.metric("Recall", f"{m['recall']:.2%}"); d.metric("F1", f"{m['f1']:.2%}"); e.metric("False-positive rate", f"{m['fpr']:.2%}")
        cm = m["confusion"]
        st.write("Confusion matrix (rows = actual, columns = predicted)")
        st.table(pd.DataFrame([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], index=["Normal", "Attack"], columns=["Normal", "Attack"]))
        st.write("Feature importance")
        st.bar_chart(pd.Series(m["importance"]))
    elif model.MODEL_PATH.exists():
        st.info("A trained model already exists. Train again to replace it.")

with tab_test:
    bundle = model.load()
    kind = st.selectbox("Simulate a window", ["normal", "flash", "lowrate"],
                        format_func={"normal": "Normal user traffic", "flash": "Legitimate flash crowd / polling", "lowrate": "Low-rate DDoS"}.get)
    if st.button("Generate and classify"):
        pk, truth = model.simulate(kind, random.Random())
        f = model.extract_features(pk); p = model.predict_proba(bundle, f)
        st.metric("Attack probability", f"{p:.1%}", "ATTACK" if p >= 0.5 else "normal")
        st.json({k: round(v, 4) for k, v in f.items()})
    st.divider()
    st.write("Or enter features manually")
    cols = st.columns(3); vals = {}
    defaults = dict(mean_iat=0.3, iat_cv=1.0, burst_freq=0.1, burst_period_cv=1.5, pkt_size_std=250.0, duration=18.0)
    for i, k in enumerate(model.FEATURES):
        vals[k] = cols[i % 3].number_input(k, value=float(defaults[k]), format="%.4f")
    if st.button("Classify features"):
        st.metric("Attack probability", f"{model.predict_proba(bundle, vals):.1%}")

with tab_link:
    st.markdown("""
1. Keep this app running (`streamlit run streamlit_app.py`). The scoring API listens on **http://127.0.0.1:5000**.
2. In the PHP dashboard open **Settings**, set *Detection engine* to **Auto** and the *Python API URL* to the address above.
3. The dashboard's **Active detection engine** badge turns green: *Python scikit-learn API*.
4. Stop this app at any time. PHP falls back to its own Random Forest on the next window, with no errors.
""")
