import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.real_classifier import MultimodalClassifier

st.title("🔍 Démonstration")
@st.cache_resource
def get_clf(): return MultimodalClassifier()
clf = get_clf()

t1, t2, t3 = st.tabs(["Texte", "Image", "Fusion"])

with t1:
    txt = st.text_area("Description")
    if st.button("Analyser Texte") and txt:
        res = clf.predict_text(txt)
        if res:
            st.metric("Meilleure Catégorie", res[0]['name'], f"{res[0]['confidence']:.1%}")
        else: st.error("Fichiers texte (.joblib) introuvables dans /models")

with t2:
    f = st.file_uploader("Image", type=['jpg', 'png'])
    if f:
        with open("temp.jpg", "wb") as out: out.write(f.getbuffer())
        res = clf.predict_image("temp.jpg")
        if res:
            st.metric("Meilleure Catégorie", res[0]['name'], f"{res[0]['confidence']:.1%}")
            for r in res:
                st.write(f"**{r['name']}**")
                st.progress(r['confidence'])
        else: st.error("Moteur Image non chargé")

with t3:
    st.info("Utilisez les deux onglets précédents pour une analyse complète.")
