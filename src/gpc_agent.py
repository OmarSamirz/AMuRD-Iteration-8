from teradataml import *
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from langgraph.graph import StateGraph, START, END

import time
import json
from typing import Dict
from dataclasses import dataclass

from modules.db import TeradataDatabase
from utils import load_translation_model
from constants import OPUS_TRANSLATION_CONFIG_PATH


# MODEL_NAME = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     torch_dtype="auto",
#     device_map="auto",
# )


MODEL_NAME = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
llm = LLM(model=MODEL_NAME)

td_db = TeradataDatabase()
td_db.connect()
df = DataFrame.from_table("gpc").to_pandas()


@dataclass
class GpcState(Dict):
    product_description: str
    segment: str = ""
    family: str = ""
    class_: str = ""
    brick: str = ""


def classify_node(text, candidates):
    if len(candidates) == 1:
        return candidates[0]

    few_shot_examples = """
    Example 1:
    Input:
    {
    "product_description": "Chicken Breasts (0.5Kg)",
    "segment": "Food/Beverage",
    "family": "Meat/Poultry/Other Animals",
    "class_": "Meat/Poultry/Other Animals - Unprepared/Unprocessed",
    "brick": "Alternative Meat/Poultry/Other Animal Species - Unprepared/Unprocessed"
    }

    Example 2:
    Input:
    {
    "product_description": "Indian Menshawi Mango (0.5Kg)",
    "segment": "Food/Beverage",
    "family": "Fruits - Unprepared/Unprocessed (Fresh)",
    "class_": "Fruits - Unprepared/Unprocessed (Fresh) Variety Packs",
    "brick": "Fruits - Unprepared/Unprocessed (Fresh) Variety Packs"
    }

    Example 3:
    Input:
    {
    "product_description": "bonomi cocoa butter biscuits - 150 g",
    "segment": "Food/Beverage",
    "family": "Bread/Bakery Products",
    "class_": "Biscuits/Cookies",
    "brick": "Biscuits/Cookies (Shelf Stable)"
    }
    """
    prompt = (
        "You are an expert, unbiased product classification AI designed to assign the most accurate product category based on the given product name and a list of possible categories.\n"
        "Follow these steps for every query:\n"
        "1. Analyze the product name for descriptive keywords and context clues.\n"
        "2. Determine its primary purpose and intended use.\n"
        "3. Compare the product to the categories, explaining your semantic reasoning.\n"
        "4. If uncertain, pick the most specific or least ambiguous fit and state your uncertainty via the confidence score.\n"
        "5. Respond only in the following JSON format.\n"
        f"{few_shot_examples}\n"
        f"CLASSIFY THIS PRODUCT:\n"
        f"Product Name: \"{text}\"\n"
        "Categories:\n" + "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)]) +
        "\n\nRespond ONLY in strict JSON format:\n"
        "{\n"
        "  \"category_number\": <number>,\n"
        "  \"category_name\": \"<name>\",\n"
        "  \"confidence\": <0.0-1.0>,\n"
        "  \"reasoning\": \"<short explanation>\"\n"
        "}\n"
        "If none fit, select the closest, set confidence below 0.5, and explain in reasoning."
    )

    sampling_params = SamplingParams(
        temperature=0.0,   # deterministic
        max_tokens=100,    # like max_new_tokens
    )

    outputs = llm.generate([prompt], sampling_params)
    output_text = outputs[0].outputs[0].text.strip()

    try:
        data = json.loads(output_text)
        return data["category_name"]
    except json.JSONDecodeError:
        output_text_lower = output_text.lower()
        for c in candidates:
            if c.lower() in output_text_lower:
                return c
        return candidates[0] 

def segment_node(state):
    state["segment"] = classify_node(state["product_description"], df["SegmentTitle"].drop_duplicates().tolist())
    return state

def family_node(state):
    candidates = df[df["SegmentTitle"] == state["segment"]]["FamilyTitle"].drop_duplicates().tolist()
    state["family"] = classify_node(state["product_description"], candidates)
    return state

def class_node(state):
    candidates = df[df["FamilyTitle"] == state["family"]]["ClassTitle"].drop_duplicates().tolist()
    state["class_"] = classify_node(state["product_description"], candidates)
    return state

def brick_node(state):
    candidates = df[df["ClassTitle"] == state["class_"]]["BrickTitle"].drop_duplicates().tolist()
    state["brick"] = classify_node(state["product_description"], candidates)
    return state

def build_graph():
    workflow = StateGraph(GpcState)
    workflow.add_node("segment", segment_node)
    workflow.add_node("family", family_node)
    workflow.add_node("class", class_node)
    workflow.add_node("brick", brick_node)

    workflow.add_edge(START, "segment")
    workflow.add_edge("segment", "family")
    workflow.add_edge("family", "class")
    workflow.add_edge("class", "brick")
    workflow.add_edge("brick", END)

    agent = workflow.compile()
    return agent

def main():
    agent = build_graph()
    translation_model = load_translation_model(OPUS_TRANSLATION_CONFIG_PATH)

    start = time.time()
    product_name = "Ahmad Tea Fruit And Herb Selection Herbal Tea bags - 20 Pieces"
    translated_name = translation_model.translate(product_name).lower()
    result = agent.invoke({"product_description": translated_name})
    print(result)
    end = time.time()
    print(f"\nThe duration of the run is {end-start:.4f} seconds")


if __name__ == "__main__":
    main()