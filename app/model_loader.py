import os
import pickle
import torch

from app.model_arch import GRU4RecTorch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models")
ENCODER_DIR = os.path.join(BASE_DIR, "encoders")


def load_encoder(name):
    path = os.path.join(ENCODER_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing encoder: {path}")
    return pickle.load(open(path, "rb"))


def load_model_and_encoders():
    # ---- Load encoders first ----
    product_enc  = load_encoder("product_encoder_fyp.pkl")
    category_enc = load_encoder("category_encoder_fyp.pkl")
    brand_enc    = load_encoder("brand_encoder_fyp.pkl")
    event_enc    = load_encoder("event_encoder_fyp.pkl")

    print("🔥 LOADED category_encoder.classes_ =", category_enc.classes_)
    print("🔥 LOADED product_encoder.classes_ =", product_enc.classes_[:10])

    # ---- Build model architecture ----
    n_products   = len(product_enc.classes_)
    n_categories = len(category_enc.classes_)
    n_brands     = len(brand_enc.classes_)
    n_events     = len(event_enc.classes_)
    n_targets    = n_categories        # model predicts categories

    model = GRU4RecTorch(
        n_targets=n_targets,
        n_products=n_products,
        n_categories=n_categories,
        n_brands=n_brands,
        n_events=n_events,
        seq_len=17
    )

    # ---- Load state_dict ----
    model_path = os.path.join(MODEL_DIR, "gru4rec_fyp_best.pth")
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()

    return model, product_enc, category_enc, brand_enc, event_enc
