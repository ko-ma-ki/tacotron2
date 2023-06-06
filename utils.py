import numpy as np
from scipy.io.wavfile import read
import torch


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.LongTensor(max_len).to(DEVICE))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()
    #test code
    x = x.to(DEVICE, non_blocking=True)
    #if torch.cuda.is_available():
    #    x = x.to(DEVICE, non_blocking=True)
    return torch.autograd.Variable(x)

def get_device(pref:str = None):
    if pref == "cpu":
        device = torch.device("cpu")
    elif pref == "cuda":
         device = torch.device("cuda")
    elif pref == "mps":
         device = torch.device("mps")
    elif pref == "dml":
        try:
            import torch_directml
            if torch_directml.is_available():
                device = torch_directml.device()
        except ModuleNotFoundError:
            print("In order ti use DML, you need to install torchtorch_directml")
            print("conda install pytorch cpuonly -c pytorch")
            print("pip install torch-directml")
    #prefが空欄なら自動でデバイスを決める
    elif pref == None:
        try:
            import torch_directml
            if torch_directml.is_available():
                device = torch_directml.device()
        except ModuleNotFoundError:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_avalrable():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

    global DEVICE
    DEVICE = device

    print('device is ' + str(device))

    return device