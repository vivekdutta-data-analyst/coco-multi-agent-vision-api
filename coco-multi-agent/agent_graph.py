"""
agent_graph.py
--------------
3-node LangGraph workflow:

    cnn_node            -> runs the trained MultiLabelCOCONet, returns top-5 preds
    multimodal_llm_node -> sends image + CNN predictions to a multimodal LLM,
                            asks it to describe the scene and validate the preds
    description_node    -> combines both into the final enhanced description

State schema is exactly as specified in the task:
    {
        "image": PIL.Image,
        "cnn_predictions": dict[str, float],
        "multimodal_llm_response": str,
        "final_description": str,
    }

Multimodal LLM provider: Anthropic Claude by default (ANTHROPIC_API_KEY env var),
but swappable — see `call_multimodal_llm()`, which is the only function you need
to edit to point at OpenAI / Gemini / Grok / HF Inference instead.
"""

import os
import io
import base64
from typing import TypedDict, Dict, Optional

from PIL import Image
from langgraph.graph import StateGraph, END

from model import load_trained_model, predict_top_k


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
class AgentState(TypedDict):
    image: Image.Image
    cnn_predictions: Dict[str, float]
    multimodal_llm_response: str
    final_description: str


# --------------------------------------------------------------------------- #
# Globals: load the CNN once at import time (cheap, reused across requests)
# --------------------------------------------------------------------------- #
_MODEL_CACHE = {"model": None, "classes": None, "device": None}


def _get_model(weights_path="coco_multilabel_cnn.pth"):
    if _MODEL_CACHE["model"] is None:
        model, classes, device = load_trained_model(weights_path)
        _MODEL_CACHE.update(model=model, classes=classes, device=device)
    return _MODEL_CACHE["model"], _MODEL_CACHE["classes"], _MODEL_CACHE["device"]


# --------------------------------------------------------------------------- #
# Node 1: cnn_node
# --------------------------------------------------------------------------- #
def cnn_node(state: AgentState) -> AgentState:
    model, classes, device = _get_model()
    top5 = predict_top_k(model, classes, device, state["image"], k=5)
    state["cnn_predictions"] = top5
    return state


# --------------------------------------------------------------------------- #
# Node 2: multimodal_llm_node
# --------------------------------------------------------------------------- #
def _image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_multimodal_llm(image: Image.Image, cnn_predictions: Dict[str, float]) -> str:
    """
    Sends the image + CNN predictions to a multimodal LLM and returns its
    scene description / validation as a plain string.

    Default provider: Anthropic Claude (multimodal, vision-capable).
    Swap the body of this function to use OpenAI / Gemini / Grok / HF instead
    -- the rest of the graph doesn't need to change.
    """
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Export it, or edit call_multimodal_llm() "
            "in agent_graph.py to use a different provider (OpenAI/Gemini/Grok/HF)."
        )

    client = anthropic.Anthropic(api_key=api_key)
    b64_image = _image_to_base64(image)

    preds_str = ", ".join(f"{k} ({v:.0%})" for k, v in cnn_predictions.items())
    prompt = (
        "A custom CNN classifier produced these top-5 multi-label predictions "
        f"for this image: {preds_str}.\n\n"
        "Look at the image and: (1) describe the scene in 1-2 sentences, "
        "(2) note whether the CNN predictions look accurate given what you see, "
        "flagging anything it likely missed or got wrong. Keep it concise."
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                              "data": b64_image}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()


def multimodal_llm_node(state: AgentState) -> AgentState:
    llm_text = call_multimodal_llm(state["image"], state["cnn_predictions"])
    state["multimodal_llm_response"] = llm_text
    return state


# --------------------------------------------------------------------------- #
# Node 3: description_node
# --------------------------------------------------------------------------- #
def description_node(state: AgentState) -> AgentState:
    preds_str = ", ".join(
        f"{cls}({prob:.0%})" for cls, prob in state["cnn_predictions"].items()
    )
    final = (
        f"CNN detected {preds_str}. "
        f"Multimodal LLM analysis: {state['multimodal_llm_response']}"
    )
    state["final_description"] = final
    return state


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #
def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("cnn_node", cnn_node)
    graph.add_node("multimodal_llm_node", multimodal_llm_node)
    graph.add_node("description_node", description_node)

    graph.set_entry_point("cnn_node")
    graph.add_edge("cnn_node", "multimodal_llm_node")
    graph.add_edge("multimodal_llm_node", "description_node")
    graph.add_edge("description_node", END)

    return graph.compile()


# Compiled, ready-to-invoke graph, imported by app.py
compiled_agent = None


def get_compiled_agent():
    global compiled_agent
    if compiled_agent is None:
        compiled_agent = build_agent_graph()
    return compiled_agent


def run_agent(image: Image.Image) -> AgentState:
    agent = get_compiled_agent()
    initial_state: AgentState = {
        "image": image,
        "cnn_predictions": {},
        "multimodal_llm_response": "",
        "final_description": "",
    }
    return agent.invoke(initial_state)


if __name__ == "__main__":
    # Quick manual test (requires coco_multilabel_cnn.pth + ANTHROPIC_API_KEY set)
    test_image = Image.open("test_image.jpg")
    result = run_agent(test_image)
    print("CNN predictions:", result["cnn_predictions"])
    print("LLM response:", result["multimodal_llm_response"])
    print("Final description:", result["final_description"])
