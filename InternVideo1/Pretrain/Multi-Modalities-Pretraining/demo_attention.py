import torch
import InternVideo

import numpy as np
import matplotlib.pyplot as plt

def visualize_attention_map(attention_weights, text, save_path=None):

    text_option_1 = ["a", "blue", "car", "is", "driving", "and", "hitting", "a", "male", "near", "another", "car", "EOT"]
    text_option_2 = ["a", "blue", "car", "is", "driving", "and", "passing", "another", "car", "EOT"]

    num_tokens = len(text)
    # Plot the attention map
    plt.figure(figsize=(5, 5))
    attention = attention_weights[0]
    attention[1:num_tokens, num_tokens] = attention[num_tokens, 1:num_tokens]
    plt.imshow(attention[1:num_tokens,1:int(num_tokens+1)], cmap='Reds', interpolation='nearest')
    plt.xticks(np.arange(len(text_option_1)), text_option_1, rotation=45)
    plt.yticks(np.arange(len(text_option_1[:-1])), text_option_1[:-1], rotation=45)
    # plt.grid(True, which='both', color='black', linewidth=0.5)
    # plt.xlabel('Key')
    # plt.ylabel('Query')
    # plt.colorbar()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# text_cand = ["an airplane is taking off", "an airplane is flying", "a dog is chasing a ball"]
# video_path = './data/demo.mp4'

text_cand = ["a blue car is driving and hitting a male near another car", "a blue car is driving and passing another car", "a total solar eclipse is happening"]
video_path = './data/demo_WTS.mp4'

device = torch.device("cuda:1")

video = InternVideo.load_video(video_path).to(device)

model = InternVideo.load_model("./models/InternVideo-MM-L-14.ckpt").to(device)
text = InternVideo.tokenize(
    text_cand
).to(device)
print(text)

with torch.no_grad():
    text_features, attention = model.encode_text(text)
    print(attention.shape)
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

save_path = 'attention_map_1.png'
text_option_1 = ["a", "blue", "car", "is", "driving", "and", "hitting", "a", "male", "near", "another", "car", "EOT"]
text_option_2 = ["a", "blue", "car", "is", "driving", "and", "passing", "another", "car", "EOT"]

visualize_attention_map(attention.cpu().numpy(), text_option_1, save_path=save_path)