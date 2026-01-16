import os
import logging

import torch
import mlflow
from ray import serve
from fastapi import FastAPI, Request, Body

# 1: Define a FastAPI app and wrap it in a deployment with a route handler.
app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ray-serve-fmnist")

def _prepare_tensor(instances):
    tensor = torch.tensor(instances, dtype=torch.float32)
    if tensor.ndim == 2 and tensor.shape[1] == 28 * 28:
        tensor = tensor.view(-1, 1, 28, 28)
    elif tensor.ndim == 3 and tensor.shape[1:] == (28, 28):
        tensor = tensor.unsqueeze(1)
    elif tensor.ndim == 4:
        pass
    else:
        raise ValueError("Unsupported input shape for instances")
    return tensor


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1})
@serve.ingress(app)
class FmnistServe:
    def __init__(self):
        model_uri = os.getenv("MODEL_URI")
        if not model_uri:
            raise RuntimeError("MODEL_URI env var is required")
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self.model = mlflow.pytorch.load_model(model_uri)
        self.model.eval()
        logger.info("Loaded model from %s", model_uri)

    @app.post("/predict")
    async def predict(self, payload: dict = Body(...)):
        instances = payload.get("instances")
        if instances is None:
            return {"error": "Missing 'instances' in request body"}

        try:
            inputs = _prepare_tensor(instances)
        except ValueError as exc:
            return {"error": str(exc)}

        with torch.no_grad():
            logits = self.model(inputs)
            predictions = logits.argmax(dim=1).tolist()
        return {"predictions": predictions}

app = FmnistServe.bind()
