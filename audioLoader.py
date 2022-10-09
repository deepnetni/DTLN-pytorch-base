import os
import fnmatch
import wave
import soundfile as sf
import numpy as np
import torch
from random import shuffle, seed
from torch.utils.data import Dataset, IterableDataset, get_worker_info

class AudioDataset(Dataset):
    def __init__(self, mix_dir, sph_dir):
        super(AudioDataset, self).__init__()
        self.mix_dir = mix_dir
        self.sph_dir = sph_dir

        self.files = np.array(os.listdir(self.mix_dir))

    def __getitem__(self, idx):
        # single-process
        #work_info = get_worker_info()
        #if work_info is None:
        #    print("work info is None", idx)
        #else:
        #    print("work id ", work_info.id, idx)
        f_name = self.files[idx]
        mix_f = os.path.join(self.mix_dir, f_name)
        sph_f = os.path.join(self.sph_dir, f_name)

        mix, fs = sf.read(mix_f)
        sph, fs = sf.read(sph_f)

        return torch.from_numpy(mix), torch.from_numpy(sph)

    def __len__(self):
        return len(self.files)


class AudioLoader(object):
    def __init__(self, mix_dirname, sph_dirname, batch_sz, fs=16000, shuffle=False):
        self.mix_dirname = mix_dirname
        self.sph_dirname = sph_dirname
        self.batch_sz = batch_sz
        self.fs = fs
        self.shuffle = shuffle

        self.count_samples()

    def count_samples(self):
        self.total_samples = 0
        # list .wav files in directory, os.listdir won't list targets recursively.
        self.file_names = fnmatch.filter(os.listdir(self.mix_dirname), "*.wav")
        self.total_samples = len(self.file_names)

        #for f in self.file_names:
        #    with wave.open(os.path.join(self.mix_dirname, f)) as w:
        #        frames = w.getnframes()
        #        self.total_samples += int(np.fix(frames / self.n_sample))

    def pad(self, mix, sph, nlen):
        # mix and sph is a list
        tgt_L = max(nlen)
        for i in range(len(mix)):
            pad_L = tgt_L - len(mix[i])
            mix[i] = np.pad(mix[i], [0, pad_L])
            sph[i] = np.pad(sph[i], [0, pad_L])

    def to_tensor(self, x):
        '''
        convert numpy array to torch.tensor
        '''
        return torch.from_numpy(x).float()

    def batch(self):
        if self.shuffle is True:
            shuffle(self.file_names)

        # iterate over files
        for idx, f in enumerate(self.file_names, start=1):
            # batch start
            if idx % self.batch_sz == 1 or self.batch_sz == 1:
                train = []      # to store the mix records which will be sent to the net
                label = []      # to store the target records
                sample_L = []   # to store the real length of audio file

            # read the audio files, return numpy data
            mix, fs_1 = sf.read(os.path.join(self.mix_dirname, f))
            sph, fs_2 = sf.read(os.path.join(self.sph_dirname, f))

            if fs_1 != self.fs or fs_2 != self.fs:
                raise ValueError('Sampling rates do not match.')

            if mix.ndim != 1 or sph.ndim != 1:
                raise ValueError('Too many channels.')

            sample_L.append(len(sph))
            train.append(mix)
            label.append(sph)

            # get a total batch
            if idx % self.batch_sz == 0:
                # padding to the same length
                # self.pad(train, label, sample_L)
                train = torch.stack([self.to_tensor(x) for x in train], dim=0)
                label = torch.stack([self.to_tensor(x) for x in label], dim=0)
                sample_L = torch.tensor(sample_L, dtype=torch.int64)
                yield train, label, sample_L

    def __iter__(self):
        try:
            yield from self.batch()
        except ValueError:
            print("Value error")
        except StopIteration:
            pass

if __name__ == "__main__":
    dirname = 'E:\\datasets\\DNS-Challenge\\datasets'
    dirname = "E:\\deepnetni\\test_audio"
    mix = os.path.join(dirname, "mix")
    sph = os.path.join(dirname, "clean")

    loader = AudioLoader(mix, sph, 1)

    for mix, sph, n_sample in loader:
        print(n_sample)
