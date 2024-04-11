import torch
import InternVideo

# text_cand = ["an airplane is taking off", "an airplane is flying", "a dog is chasing a ball"]
# video_path = './data/demo.mp4'

text_cand = ["a blue car is approaching another car, hitting a male, and driving through", "a blue car is approaching another car and driving through", "a total solar eclipse is happening"]
video_path = './data/demo_WTS.mp4'

device = torch.device("cuda:1")

video = InternVideo.load_video(video_path).to(device)

model = InternVideo.load_model("./models/InternVideo-MM-L-14.ckpt").to(device)
text = InternVideo.tokenize(
    text_cand
).to(device)

with torch.no_grad():
    text_features = model.encode_text(text)
    video_features = model.encode_video(video.unsqueeze(0))

    video_features = torch.nn.functional.normalize(video_features, dim=1)
    text_features = torch.nn.functional.normalize(text_features, dim=1)
    t = model.logit_scale.exp()
    probs = (video_features @ text_features.T * t).softmax(dim=-1).cpu().numpy()

    allocated_memory = torch.cuda.memory_allocated()
    print(f"Allocated GPU memory: {allocated_memory / 1024**3} GB ")

print("Label probs: ")  # [[9.5619422e-01 4.3805469e-02 2.0393253e-07]]
for t, p in zip(text_cand, probs[0]):
    print("{:30s}: {:.4f}".format(t, p))

max_allocated_memory = torch.cuda.max_memory_allocated()
print(f"Max allocated GPU memory: {max_allocated_memory / 1024**3} GB ")