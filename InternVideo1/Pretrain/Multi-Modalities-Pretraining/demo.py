import torch
import InternVideo

text_cand = ["a vehicle has potential interaction with another pedestrian and causes abrupt change of pedestrian's behavior", "a vehicle is passing a pedestrian on the road without affecting pedestrian's movement"]
video_path = './data/demo_WTS.mp4'

device = torch.device("cuda:1")

video = InternVideo.load_video(video_path).to(device)

model = InternVideo.load_model("./models/InternVideo-MM-L-14.ckpt").to(device)
text = InternVideo.tokenize(
    text_cand
).to(device)

with torch.no_grad():
    text_features, attention = model.encode_text(text)
    print(video.shape)
    video_features = model.encode_video(video.unsqueeze(0))
    print(video_features.shape)

    video_features = torch.nn.functional.normalize(video_features, dim=1)
    text_features = torch.nn.functional.normalize(text_features, dim=1)
    t = model.logit_scale.exp()
    probs = (video_features @ text_features.T * t).softmax(dim=-1).cpu().numpy()

print("Label probs: ")  # [[9.5619422e-01 4.3805469e-02 2.0393253e-07]]
for t, p in zip(text_cand, probs[0]):
    print("{:30s}: {:.4f}".format(t, p))