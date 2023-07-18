# CUDA setup and installation guide
# https://stackoverflow.com/questions/60987997/why-torch-cuda-is-available-returns-false-even-after-installing-pytorch-with
#
import torch
import sys
from subprocess import call
import subprocess


def print_v(*args):
    try:
        print(*args)
    except Exception as e:
        print("Error while running: {}. {}".format(args, e))
        pass


def execute_command(command):
    try:
        # Execute the command
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, universal_newlines=True)

        # Print the command output
        print(output)
    except subprocess.CalledProcessError as e:
        # Handle any errors that occurred during command execution
        print("Error: '{}' running command  with  '{}'".format(e.output, command))


print_v('__Python VERSION:', sys.version)
print_v('__pyTorch VERSION:', torch.__version__)
print_v('__CUDA VERSION')
execute_command("nvcc --version")
print_v('__CUDNN VERSION:', torch.backends.cudnn.version())
print_v('__Number CUDA Devices:', torch.cuda.device_count())
print_v('__Devices')
execute_command("nvidia-smi --format=csv --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free")
print_v('Active CUDA Device: GPU', torch.cuda.current_device())
print_v('Available devices ', torch.cuda.device_count())
print_v('Current cuda device ', torch.cuda.current_device())
