import math

import torch
import wandb
from transformers import AutoModel, AutoVideoProcessor

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from models.decoder import VJEPA2Decoder

run = wandb.init(project="vjepa2-decoder-experiment")

hf_repo = "facebook/vjepa2-vitl-fpc64-256"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModel.from_pretrained(hf_repo).to(device)
for param in model.parameters():
    param.requires_grad = False
processor = AutoVideoProcessor.from_pretrained(hf_repo)
decoder = VJEPA2Decoder(model.config).to(device)

dataset = LeRobotDataset("lerobot/droid_100")
camera_key = dataset.meta.camera_keys[0]
delta_timestamps = {
    camera_key: [t / dataset.fps for t in range(16)],
    "episode_index":  [t / dataset.fps for t in range(16)],
}
dataset = LeRobotDataset("lerobot/droid_100", delta_timestamps=delta_timestamps)

optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-4, weight_decay=0.1)
dataloader = torch.utils.data.DataLoader(
    dataset,
    # num_workers=4,
    # batch_size=1024,
    batch_size=32,
    shuffle=True,
    pin_memory=device.type != "cpu",
    drop_last=True,
)

# def lr_lambda(current_step: int):
#     if current_step < 2000:
#         # Linear warmup
#         return float(current_step) / 2000.0
#     else:
#         # Cosine decay
#         progress = float(current_step - 2000) / (150000 - 2000)
#         return 0.5 * (1.0 + math.cos(math.pi * progress))
#
#
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

decoder.train()
num_steps = 2000
run.watch(decoder)
for step in range(num_steps):
    for batch in dataloader:
        images = batch[camera_key]
        pixel_values = processor(images, return_tensors="pt")["pixel_values_videos"].to(model.device)

        with torch.no_grad():
            outputs = model(pixel_values, skip_predictor=True)
        reconstructed = decoder(outputs.last_hidden_state)

        loss = torch.nn.functional.mse_loss(reconstructed, pixel_values)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)

        optimizer.step()
        optimizer.zero_grad()

        print(f"Step {step + 1}/{num_steps}, Loss: {loss.item()}")
        run.log({"loss": loss.item()})

    # scheduler.step()
    torch.save(decoder.state_dict(), f"decoder_{step}.pt")

torch.save(decoder.state_dict(), "decoder_final.pt")
