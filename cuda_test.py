import torch

print(torch.cuda.is_available())
torch.zeros(1).cuda()

# https://stackoverflow.com/questions/60987997/why-torch-cuda-is-available-returns-false-even-after-installing-pytorch-with