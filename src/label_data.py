import pandas as pd
from tqdm.auto import tqdm

from gpc_agent import build_graph
from utils import classify_product, load_translation_model
from constants import LABELED_GPC_PATH


def label_products(df, product_col, agent, translation_model = None):
    segment_lst = family_lst = class_lst = brick_lst = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        result = classify_product(row[product_col], agent, translation_model)
        segment_lst.append(result["segment"])
        family_lst.append(result["family"])
        class_lst.append(result["class_"])
        brick_lst.append(result["brick"])

    df["segment_pred"] = segment_lst
    df["family_pred"] = family_lst
    df["class_pred"] = class_lst
    df["brick_pred"] = brick_lst

    return df

def main():
    # translation_model = load_translation_model(OPUS_TRANSLATION_CONFIG_PATH)
    agent = build_graph()

    df_sample = pd.read_csv(LABELED_GPC_PATH)
    df_labeled = label_products(df_sample, "translated_name", agent)

    df_labeled.to_csv(LABELED_GPC_PATH, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()