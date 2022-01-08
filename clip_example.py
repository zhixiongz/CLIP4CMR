import torch
import clip
from PIL import Image
import re
import string

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50", device=device)
params_to_update = list(model.parameters())   # 返回模型的可训练参数
total = sum([param.nelement() for param in params_to_update])
print("Number of parameter: %.2fM" % (total / 1e6))
image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)  # torch.Size([1, 3, 224, 224])
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]