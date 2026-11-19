
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
EXPORT_DIR = ROOT / "exports"
EXPORT_DIR.mkdir(exist_ok=True)

CHECKPOINT = ROOT / "checkpoints/mlp_12workloads.pt"
OUTPUT = EXPORT_DIR / "mlp_12workloads.onnx"

checkpoint = torch.load(CHECKPOINT, map_location="cpu")

H = checkpoint["history_length"]
EMBED_DIM = checkpoint["embedding_dim"]
pc_to_id = checkpoint["pc_to_id"]


class BranchMLP(nn.Module):
    def __init__(self, num_pcs):
        super().__init__()

        self.embedding = nn.Embedding(
            num_pcs + 1, EMBED_DIM, padding_idx=0
        )

        self.layers = nn.Sequential(
            nn.Linear(EMBED_DIM + H, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, pc, history):
        features = torch.cat(
            [self.embedding(pc), history], dim=1
        )
        return self.layers(features).squeeze(1)


model = BranchMLP(len(pc_to_id))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Example inputs for tracing the model
dummy_pc = torch.tensor([1], dtype=torch.long)
dummy_history = torch.zeros((1, H), dtype=torch.float32)

# Export the trained model
torch.onnx.export(
    model,
    (dummy_pc, dummy_history),
    str(OUTPUT),
    export_params=True,
    opset_version=17,
    input_names=["pc_id", "history"],
    output_names=["logit"],
    dynamic_axes={
        "pc_id": {0: "batch"},
        "history": {0: "batch"},
        "logit": {0: "batch"}
    }
)

print("Exported:", OUTPUT)

# Save the PC encoding for future C++ inference
with open(EXPORT_DIR / "pc_vocab.json", "w") as f:
    json.dump(
        {str(pc): idx for pc, idx in pc_to_id.items()},
        f
    )

# Validate the ONNX graph
onnx_model = onnx.load(str(OUTPUT))
onnx.checker.check_model(onnx_model)
print("ONNX graph validation: PASSED")

# Compare PyTorch and ONNX Runtime predictions
session = ort.InferenceSession(
    str(OUTPUT),
    providers=["CPUExecutionProvider"]
)

torch.manual_seed(42)
test_pc = torch.tensor([1, 2, 3, 4], dtype=torch.long)
test_history = torch.randint(0, 2, (4, H)).float()

with torch.no_grad():
    torch_logits = model(
        test_pc, test_history
    ).numpy()

onnx_logits = session.run(
    ["logit"],
    {
        "pc_id": test_pc.numpy(),
        "history": test_history.numpy()
    }
)[0]

difference = np.max(
    np.abs(torch_logits - onnx_logits)
)

np.testing.assert_allclose(
    torch_logits, onnx_logits,
    rtol=1e-5, atol=1e-5
)

print("Maximum output difference:", difference)
print("PyTorch/ONNX Runtime parity: PASSED")
print("ONNX model size:",
      round(OUTPUT.stat().st_size / 1024, 2), "KiB")
