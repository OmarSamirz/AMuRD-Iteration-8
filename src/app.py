import torch
import streamlit as st
from teradataml import *

from modules.db import TeradataDatabase
from utils import load_tfidf_model, load_embedding_model
from constants import TFIDF_CLASSIFIER_CONFIG_PATH, E5_LARGE_INSTRUCT_CONFIG_PATH


# --- PAGE CONFIG ---
st.set_page_config(page_title="Prodify - Product Classifier", layout="centered")


# --- CACHED LOADERS ---
@st.cache_resource
def get_db():
    db = TeradataDatabase()
    db.connect()
    return db


@st.cache_resource
def get_models():
    embed_model = load_embedding_model(E5_LARGE_INSTRUCT_CONFIG_PATH)
    tfidf_model = load_tfidf_model(TFIDF_CLASSIFIER_CONFIG_PATH)
    return embed_model, tfidf_model


@st.cache_data
def get_classes():
    return DataFrame.from_table("classes").to_pandas().sort_values(by="id")["class_name"].tolist()


@st.cache_data
def get_products():
    df = DataFrame.from_table("products").to_pandas()
    df["Select"] = [False] * len(df)
    return df


# --- CLASSIFICATION LOGIC ---
def classify_product(model, product, classes):
    if hasattr(model, "predict"):  # TF-IDF XGBoost
        prediction = model.predict([product])[0]
        return classes[prediction]

    elif hasattr(model, "get_scores"):  # Embedding model
        prediction = model.get_scores(product, classes)
        prediction = torch.argmax(prediction, dim=1)[0]
        return classes[prediction]

    else:
        return "⚠️ Unknown model type"


# --- MAIN APP ---
def main():
    db = get_db()
    embed_model, tfidf_model = get_models()
    classes = get_classes()
    df = get_products()

    # --- TOP BAR ---
    left, right = st.columns([5, 2])
    with left:
        st.markdown("# Prodify")

    with right:
        st.write("")
        colr1, colr2 = st.columns(2)

        if "model_name" not in st.session_state:
            st.session_state["model_name"] = None

        if colr1.button("Embedding Model"):
            st.session_state["model"] = embed_model
            st.session_state["model_name"] = "Embedding Model"

        if colr2.button("TF-IDF XGBoost Classifier"):
            tfidf_model.load()
            st.session_state["model"] = tfidf_model
            st.session_state["model_name"] = "TF-IDF XGBoost Classifier"

        # Highlight the selected model
        if st.session_state["model_name"]:
            st.markdown(
                f"<p style='color: green; font-weight: bold;'>✅ Using {st.session_state['model_name']}</p>",
                unsafe_allow_html=True
            )


    # --- PRODUCT LIST ---
    st.markdown("#### Product Table")
    edited_df = st.data_editor(
        df[["product_name", "Select"]],
        hide_index=True,
        use_container_width=True,
        num_rows="fixed"
    )

    selected_rows = edited_df[edited_df["Select"] == True]["product_name"].tolist()
    selected_product = selected_rows[0] if selected_rows else None

    if selected_product:
        st.write("You selected:", selected_product)

    # --- USER INPUT ---
    user_input = st.text_input("Or type a product name:")

    # --- CLASSIFICATION OUTPUT ---
    st.markdown("---")
    st.markdown("#### Classification Output")
    classification_output = st.empty()

    # --- CLASSIFICATION BUTTON ---
    if st.button("Classify Product"):
        product_to_classify = user_input if user_input else selected_product

        if not product_to_classify:
            classification_output.markdown("⚠️ Please select or type a product.")
        else:
            model = st.session_state.get("model", None)
            if not model:
                classification_output.markdown("⚠️ Please select a model first.")
            else:
                try:
                    result = classify_product(model, product_to_classify, classes)
                    classification_output.markdown(f"**Prediction:** {result}")
                except Exception as e:
                    classification_output.markdown(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
