import math
import streamlit as st

st.set_page_config(page_title="PBMV Success Calculator", page_icon="🫀", layout="centered")

# ---- TOP TITLE (first line) ----
st.markdown("## The Wilkins-Integrated Clinical PBMV score for PBMV success")

# ---- CALCULATOR HEADING ----
st.title("PBMV Success Probability Calculator")
st.caption("Logistic regression model → predicted probability of PBMV success")

with st.expander("Model formula (for transparency)"):
    st.latex(
        r"""
        \text{logit}(P)=
        (-0.0676\cdot Wilkins)
        -(0.0088\cdot Age)
        +(0.9011\cdot FCIV)
        -(0.012\cdot AF)
        -(1.1874\cdot PriorComm)
        -(0.0015\cdot RVSP)
        -(0.4646\cdot SevereTR)
        +(2.276\cdot MVApre)
        -1.0877
        """
    )
    st.write("Predicted probability = 1 / (1 + exp(-logit(P)))")

st.subheader("Inputs")

# --- Numeric inputs ---
wilkins = st.number_input(
    "Wilkins score (range 4–16)",
    min_value=4.0, max_value=16.0, value=8.0, step=1.0
)
age = st.number_input("Age (years)", min_value=0.0, max_value=120.0, value=55.0, step=1.0)
rvsp = st.number_input("RVSP (mmHg)", min_value=0.0, max_value=200.0, value=45.0, step=1.0)
mva_pre = st.number_input("Pre-BMV mitral valve area (cm²)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

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
    (-0.0676 * wilkins) +
    (-0.0088 * age) +
    (0.9011 * FCIV) +
    (-0.0120 * AF) +
    (-1.1874 * PriorComm) +
    (-0.0015 * rvsp) +
    (-0.4646 * SevereTR) +
    (2.2760 * mva_pre) +
    (-1.0877)
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
    st.write(f"Wilkins={wilkins}")
    st.write(f"FCIV={FCIV}, AF={AF}, PriorComm={PriorComm}, SevereTR={SevereTR}")
    st.write(f"logit(P) = {logitP:.4f}")
    st.write(f"P = {prob:.6f}")

# ---- LOWER PART OF PAGE (definition + dropdown citation) ----
st.divider()
st.caption(
    "PBMV success was defined as post-procedural MVA ≥ 1.5 cm², irrespective of the percentage increase, "
    "and MR ≤ grade 2, with no more than a 1-grade increment in severity and without in-hospital complications."
)

with st.expander("Citation"):
    st.write(
        "Manoret P, Thonghong T, Meemook K, Kosallavat S, Aroonsiriwattana S, Songsangjinda T, "
        "Suwanugsorn S, Nilmoje T, Cheewatanakornkul S, Wisaratapong T, Limumpornpetch S, Lohawijarn W, "
        "Thungthienthong M, Chamnarnphol N, Chandavimol M, Suwannasom P, Jintapakorn W, Chichareon P. "
        "Impact of Procedural Success Definitions on Long-Term Outcomes in Patients With Rheumatic Mitral Stenosis "
        "Treated With Percutaneous Balloon Mitral Valvuloplasty: A Multicenter, Retrospective Cohort Study. "
        "J Am Heart Assoc. 2024;13(16):e031433."
    )
st.caption("Developed by **Theerapat Buppodom, MD** · Division of Cardiology, Prince of Songkla University")
st.caption("Educational/research tool only. Clinical decisions should not rely on this calculator alone.")
