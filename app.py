"""
app.py
------
FastAPI wrapper around the LangGraph agent.

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000

Test:
    curl -X POST "http://localhost:8000/enhanced-vision" -F "file=@test_image.jpg"
"""

import io
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from agent_graph import run_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enhanced-vision-api")

app = FastAPI(
    title="Enhanced COCO Multi-Label Visual Agent API",
    description="CNN multi-label predictions enhanced by a multimodal LLM via a LangGraph agent.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"status": "ok", "message": "POST an image to /enhanced-vision"}


@app.post("/enhanced-vision")
async def enhanced_vision(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        raw_bytes = await file.read()
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    try:
        result = run_agent(image)
    except Exception as e:
        logger.exception("Agent execution failed")
        raise HTTPException(status_code=500, detail=f"Agent pipeline failed: {e}")

    response = {
        "cnn_predictions": result["cnn_predictions"],
        "multimodal_enhancement": result["multimodal_llm_response"],
        "final_enhanced_response": result["final_description"],
    }
    return JSONResponse(content=response)
