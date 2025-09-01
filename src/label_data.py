import pandas as pd
from tqdm.auto import tqdm

from gpc_agent import build_graph
from utils import classify_product, load_translation_model
from constants import SAMPLE_PRODUCTS_PATH, OPUS_TRANSLATION_CONFIG_PATH, LABELED_PRODUCTS_PATH


def label_products(df, agent, translation_model):
    segment_lst = family_lst = class_lst = brick_lst = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        result = classify_product(agent, translation_model, row["product_name"])
        segment_lst.append(result["segment"])
        family_lst.append(result["family"])
        class_lst.append(result["class_"])
        brick_lst.append(result["brick"])

    df["segment"] = segment_lst
    df["family"] = family_lst
    df["class"] = class_lst
    df["brick"] = brick_lst

    return df

def main():
    translation_model = load_translation_model(OPUS_TRANSLATION_CONFIG_PATH)
    agent = build_graph()

    df_sample = pd.read_csv(SAMPLE_PRODUCTS_PATH)
    df_labeled = label_products(df_sample, agent, translation_model)

    df_labeled.to_csv(LABELED_PRODUCTS_PATH, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()