import numpy as np
import torch
from app.model_loader import load_model_and_encoders

# Load model + encoders ONCE globally
model, product_encoder, category_encoder, brand_encoder, event_encoder = load_model_and_encoders()

MAX_LEN = 17


def safe_transform(encoder, values):
    """Safely transform values using LabelEncoder.
       Unknown values are mapped to first known class."""
    known = set(encoder.classes_)
    safe_values = []

    for v in values:
        if v in known:
            safe_values.append(v)
        else:
            # map unseen values to first class
            safe_values.append(encoder.classes_[0])

    return encoder.transform(safe_values).tolist()


def pad(seq):
    seq = list(seq)

    # truncate to max length from the right (keep most recent)
    if len(seq) > MAX_LEN:
        seq = seq[-MAX_LEN:]

    # right-align sequence and left-pad with zeros
    padded = np.zeros(MAX_LEN, dtype=np.int64)
    padded[-len(seq):] = np.array(seq, dtype=np.int64)

    # [1, MAX_LEN] tensor
    return torch.from_numpy(padded).unsqueeze(0)  # shape (1, MAX_LEN)


def predict_next_categories(product_ids, category_ids, brand_ids, event_types):
    try:
        # 1. CLEAN INPUTS
        product_ids = [str(x) for x in product_ids]
        category_ids = [str(x) if x else "" for x in category_ids]
        brand_ids = [str(x) if x else "" for x in brand_ids]
        event_types = [str(x) for x in event_types]

        # 2. SAFE ENCODING (prevents crashes)
        enc_prod = safe_transform(product_encoder, product_ids)
        enc_cat = safe_transform(category_encoder, category_ids)
        enc_brand = safe_transform(brand_encoder, brand_ids)
        enc_event = safe_transform(event_encoder, event_types)

        # 3. PAD SEQUENCES
        t_prod = pad(enc_prod)
        t_cat = pad(enc_cat)
        t_brand = pad(enc_brand)
        t_event = pad(enc_event)

        # 4. RUN MODEL
        with torch.no_grad():
            logits = model(t_prod, t_cat, t_brand, t_event)  # shape [1, n_categories]

            # make sure k is not bigger than number of categories
            num_classes = logits.size(1)
            k = min(5, num_classes)  # or min(10, num_classes) if you really want up to 10

            top_indices = logits.topk(k, dim=1).indices.squeeze(0).cpu().numpy().tolist()

        # 5. DECODE CATEGORY IDS
        decoded = category_encoder.inverse_transform(top_indices)
        return decoded.tolist()

    except Exception as e:
        print("🔥 Prediction ERROR:", e)
        return []
