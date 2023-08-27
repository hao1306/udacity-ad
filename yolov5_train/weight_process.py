import torch

ckpt = torch.load("yolov5-master/runs/train/exp4/weights/best.pt", map_location = "cpu")
print(ckpt["model"])
csd = ckpt["model"].float().state_dict()
torch.save(csd,"yolov5-master/runs/train/exp4/weights/best_02.pt")
