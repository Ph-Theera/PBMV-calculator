import math
import streamlit as st

st.set_page_config(page_title="PBMV Success Calculator", page_icon="🫀", layout="centered")

st.title("PBMV Success Probability Calculator")
st.caption("Logistic regression model → predicted probability of PBMV success")

with st.expander("Model formula (for transparency)"):
    st.latex(
        r"""
        \text{logit}(P)=(-0.4064\cdot Wilkins8)+(-0.5569\cdot Wilkins9)+(-0.2557\cdot Wilkins\ge 10)
        -(0.0086\cdot Age)
        +(0.9879\cdot FCIV)
        -(0.0038\cdot AF)
        -(1.2366\cdot PriorComm)
        -(0.0021\cdot RVSP)
        -(0.4761\cdot SevereTR)
        +(2.2395\cdot MVApre)
        -1.2917
        """
    )
    st.write("Predicted probability = 1 / (1 + exp(-logit(P)))")

st.subheader("Inputs")

# --- Wilkins score handling (mutually exclusive dummies) ---
wilkins_group = st.selectbox(
    "Wilkins score category",
    options=["≤ 7", "8", "9", "≥ 10"],
    index=0,
    help="The model uses dummy variables for 8, 9, and ≥10 (≤7 is the reference)."
)
Wilkins8 = 1 if wilkins_group == "8" else 0
Wilkins9 = 1 if wilkins_group == "9" else 0
Wilkins10 = 1 if wilkins_group == "≥ 10" else 0

# --- Numeric inputs ---
age = st.number_input("Age (years)", min_value=0.0, max_value=120.0, value=55.0, step=1.0)
rvsp = st.number_input("RVSP (mmHg)", min_value=0.0, max_value=200.0, value=45.0, step=1.0)
mva_pre = st.number_input("Pre-PBMV Mitral Valve Area (cm²)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

# --- Binary inputs ---
col1, col2 = st.columns(2)
with col1:
    fciv = st.checkbox("Functional class IV (FC IV)", value=False)
    prior_comm = st.checkbox("Prior commissurotomy (surgical/percutaneous)", value=False)
with col2:
    af = st.checkbox("History of atrial fibrillation (AF)", value=False)
    severe_tr = st.checkbox("Severe tricuspid regurgitation (Severe TR)", value=False)

FCIV = 1 if fciv else 0
AF = 1 if af else 0
PriorComm = 1 if prior_comm else 0
SevereTR = 1 if severe_tr else 0

# --- Compute logit and probability ---
logitP = (
    (-0.4064 * Wilkins8) +
    (-0.5569 * Wilkins9) +
    (-0.2557 * Wilkins10) +
    (-0.0086 * age) +
    (0.9879 * FCIV) +
    (-0.0038 * AF) +
    (-1.2366 * PriorComm) +
    (-0.0021 * rvsp) +
    (-0.4761 * SevereTR) +
    (2.2395 * mva_pre) +
    (-1.2917)
)

# Numerically stable sigmoid
if logitP >= 0:
    prob = 1.0 / (1.0 + math.exp(-logitP))
else:
    exp_lp = math.exp(logitP)
    prob = exp_lp / (1.0 + exp_lp)

st.divider()
st.subheader("Result")

st.metric("Predicted probability of PBMV success", f"{prob*100:.1f}%")

with st.expander("Show calculation details"):
    st.write(f"Wilkins8={Wilkins8}, Wilkins9={Wilkins9}, Wilkins≥10={Wilkins10}")
    st.write(f"logit(P) = {logitP:.4f}")
    st.write(f"P = {prob:.6f}")

st.caption("Educational/research tool only. Clinical decisions should not rely on this calculator alone.")