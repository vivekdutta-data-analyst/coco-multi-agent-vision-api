# Enhanced COCO Multi-Label Visual Agent API

## What's here
- `model.py` — `MultiLabelCOCONet` CNN (4 conv blocks → GlobalAvgPool → sigmoid) + training loop + Kaggle dataset loader
- `agent_graph.py` — 3-node LangGraph workflow (`cnn_node` → `multimodal_llm_node` → `description_node`), exact `AgentState` schema
- `app.py` — FastAPI app exposing `POST /enhanced-vision`
- `requirements.txt`
- `test_api.ipynb` — training demo + live API test
- `train_results.png` — generated after training (loss curves + sample top-5 predictions)

## How to run end-to-end

### 1. Train the CNN (Colab/Kaggle, free GPU tier)
```bash
!kaggle datasets download -d shubham2703/coco-dataset-for-multi-label-image-classification
!unzip -q coco-dataset-for-multi-label-image-classification.zip -d coco_data
```
Then in Python:
```python
from model import CocoMultiLabelDataset, train_model
train_ds = CocoMultiLabelDataset("coco_data/train_labels.csv", "coco_data/train_images", augment=True)
val_ds   = CocoMultiLabelDataset("coco_data/val_labels.csv", "coco_data/val_images", classes=train_ds.classes)
model, history = train_model(train_ds, val_ds, num_classes=len(train_ds.classes),
                              class_names=train_ds.classes, epochs=15)
```
This saves `coco_multilabel_cnn.pth`, `classes.json`, and `train_results.png`.

> **Note on dataset paths:** the Kaggle download's exact CSV/folder names vary by
> version — after unzipping, run `!find coco_data -maxdepth 2` and adjust the
> `csv_path`/`img_dir` arguments above to match what you actually get.
> `CocoMultiLabelDataset` auto-detects the image-id column and treats every other
> CSV column as a class label, so it adapts to most multi-label CSV layouts without
> code changes.

### 2. Set your multimodal LLM API key
```bash
export ANTHROPIC_API_KEY=sk-...
```
(Or edit `call_multimodal_llm()` in `agent_graph.py` to use OpenAI/Gemini/Grok/HF instead — only that one function needs to change.)

### 3. Copy `coco_multilabel_cnn.pth` next to `app.py`, then run the API
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 4. Test it
```bash
curl -X POST "http://localhost:8000/enhanced-vision" -F "file=@test_image.jpg"
```

Expected response shape:
```json
{
  "cnn_predictions": {"person": 0.92, "book": 0.78, "chair": 0.65},
  "multimodal_enhancement": "...",
  "final_enhanced_response": "..."
}
```

## Design notes
- **Sigmoid + BCELoss**: labels are independent (multi-hot), not mutually exclusive, so each output unit gets its own sigmoid and BCE loss rather than softmax/cross-entropy.
- **GlobalAvgPool** before the FC layer keeps the model size-agnostic and reduces overfitting vs. flattening a full feature map.
- **LangGraph state** is a plain `TypedDict` matching the spec exactly, so each node just reads/writes named keys — easy to unit-test in isolation (see the `if __name__ == "__main__"` blocks in `model.py`/`agent_graph.py`).
- **Model caching**: the CNN is loaded once at process start (`_MODEL_CACHE` in `agent_graph.py`), not on every request, so `/enhanced-vision` stays fast.
- **Provider-agnostic LLM call**: all multimodal-LLM-specific code lives in one function (`call_multimodal_llm`), so swapping providers doesn't touch the graph or API layers.
