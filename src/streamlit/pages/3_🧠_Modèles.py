import streamlit as st
import pandas as pd

st.title("🧠 Architecture & Plan")
st.markdown("""
### 📋 Plan d'implémentation
1. **Preprocessing** : Normalisation & DINO Resize (518px).
2. **Features** : Extraction DINOv2 Global + ResNet Texture.
3. **Moteur** : Entraînement XGBoost Champion.
4. **Décision** : Soft-Voting pondéré (XGB:4, DINO:2, EffNet:1).
""")

c1, c2 = st.columns(2)
with c1:
    st.subheader("🖼️ Stream Image (92%)")
    st.table(pd.DataFrame({
        "Modèle": ["XGBoost", "DINOv3", "EffNet"],
        "Score": ["80.1%", "79.1%", "75.4%"]
    }))

with c2:
    st.subheader("📝 Stream Texte (84%)")
    st.table(pd.DataFrame({
        "Modèle": ["SVM", "RF", "LogReg"],
        "Score": ["84.1%", "72.0%", "69.5%"]
    }))
