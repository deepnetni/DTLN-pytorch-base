import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import shutil
import soundfile as sf
from audioLoader import AudioLoader, AudioDataset
from matplotlib import pyplot as plt
from DTLNNet import DTLNNet
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from log import getLogger, setLevel
from utils import snr_cost, loss_mask

os.environ['max_split_size_mb'] = "512"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = '512'


class Model(object):
    def __init__(self, epoches=10, batch_sz=16,
                 n_frame=256, n_hop=128,
                 lr=0.001, lr_decay_period=2, lr_decay_factor=0.98,
                 val_step=1, trn_step=50, window=None):
        '''
        n_frame:    the nfft lenght, in samples, each time.
        n_hop:      the length of the frame shift calculated in samples.
        window:     represents the length of the window if it is a scale number;
                    otherwise, means the parameters of the window.
        '''
        self.n_block = n_hop
        self.n_frame = n_frame
        self.n_hop = n_hop
        self.n_overlap = n_frame - n_hop
        self.eps = torch.finfo(torch.float32).eps
        self.n_batch = batch_sz
        self.n_epoch = epoches
        self.lr = lr
        self.lr_decay_period = lr_decay_period
        self.lr_decay_factor = lr_decay_factor
        self.save_path = os.path.join(os.getcwd(), "pretrained")
        self.save_fname = "DTLN.pth"
        self.val_step = val_step
        self.trn_step = trn_step

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.device = torch.device('cuda:0') if torch.cuda.is_available() is True else torch.device('cpu')
        self.logger = getLogger("DTLN")
        setLevel(self.logger, "warning")

        if window is None:
            self.win = self.gen_window()

    @staticmethod
    def loss_fn(est, lbl):
        loss = snr_cost(est, lbl)
        return torch.mean(loss)

    def gen_window(self):
        if self.n_block * 2 > self.n_frame:
            n_hann = self.n_overlap * 2 - 1
            # matlab start index from 1, while python start from 0
            hann = np.sqrt(np.hanning(n_hann+2))[1:-1]
            win = np.concatenate((
                np.zeros(1),
                hann[:self.n_overlap-1],
                np.ones(self.n_frame - 2 * self.n_overlap - 1),
                hann[self.n_overlap-1:]
            ))
        else:
            n_hann = 2 * self.n_block - 1
            hann = np.sqrt(np.hanning(n_hann+2))[1:-1]
            win = np.concatenate((
                np.zeros(int(self.n_frame/2) - self.n_block + 1),
                hann,
                np.zeros(int(self.n_frame/2) - self.n_block)
            ))

        return win

    def to_frame(self, x):
        '''
        x, batch input with the shape of (batch, data of each file)
        split to (batch, frames, each frame)
        '''
        # padding first
        L = x.shape[1] + self.n_overlap
        x = F.pad(x, (self.n_overlap, 0))
        nframes = int((L - self.n_frame) / self.n_hop) + 1
        valid_L = nframes * self.n_hop
        index_in_frame = torch.arange(0, self.n_frame, 1)
        index_over_frame = torch.arange(0, valid_L, self.n_hop).view(-1,1)
        indices = index_over_frame + index_in_frame

        out = torch.stack([d[indices] for d in x], dim=0)

        return out

    def ola(self, x):
        for btx, batch_data in enumerate(x):
            for idx, frame in enumerate(batch_data):
                if idx == 0:
                    out = frame
                else:
                    out = torch.concat((out[:-self.n_overlap],
                                        out[-self.n_overlap:]+frame[:self.n_overlap],
                                        frame[self.n_overlap:]), dim=0)
            if btx == 0:
                out_ = torch.unsqueeze(out, dim=0)
            else:
                out_ = torch.concat((out_, torch.unsqueeze(out, dim=0)), dim=0)

        return out_

    def feed(self, x):
        '''
        x is the type of torch tensor which arranged by rows
        '''
        # split to blocks with the shape of (batch, n_frame, frame_samples)
        x = self.to_frame(x)

        # fft
        x = x * self.win
        x_fft = torch.fft.rfft(x, dim=-1)
        mag = torch.absolute(x_fft)
        phi = torch.atan2(x_fft.imag + self.eps, x_fft.real + self.eps)

        return mag, phi

    def resynthesis(self, mag, phi):
        x = mag * torch.exp(1j*phi)
        x = torch.fft.irfft(x, dim=-1).real
        x = x * self.win

        # overlap add methods
        out = self.ola(x)
        return out[self.n_overlap:]

    def train_with_dataset(self, mix_dirname, sph_dirname,
                           val_mix_dirname, val_sph_dirname,
                           fs=16000, reload=True):
        dataset = AudioDataset(mix_dirname, sph_dirname)
        val_dataset = AudioDataset(val_mix_dirname, val_sph_dirname)
        loader = DataLoader(dataset=dataset, batch_size=self.n_batch, num_workers=10, pin_memory=True, shuffle=True, drop_last=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.n_batch, num_workers=10, pin_memory=True, shuffle=True, drop_last=True)
        record_path = os.path.join('./log_training', time.strftime('%Y%d%m'))
        if os.path.exists(record_path):
            shutil.rmtree(record_path)
        #writer = SummaryWriter(log_dir=record_path)

        net = DTLNNet(self.n_frame, self.n_hop)
        if reload is True and os.path.exists(os.path.join(self.save_path, self.save_fname)) is True:
            net.load_state_dict(torch.load(os.path.join(self.save_path, self.save_fname)))
        net.to(self.device)

        loss_fn = Model.loss_fn
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, amsgrad=False)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=self.lr_decay_period,
                                                    gamma=self.lr_decay_factor)

        total_train_step = 0
        current_epoch = 0
        minimum_loss_val = torch.finfo(torch.float32).max
        while current_epoch < self.n_epoch:
            epoch_st = time.time()
            # set net to train mode
            net.train()
            acc_train_loss = 0
            acc_train_frames = 0
            for current_batch, (mix, sph) in enumerate(loader):
                bt_st = time.time()
                mix = mix.to(self.device)
                sph = sph.to(self.device)

                optimizer.zero_grad()
                with torch.enable_grad():
                    est = net(mix)
                #loss = loss_fn(est, lbl_mag, mask)
                loss = loss_fn(est, sph)
                loss.backward()
                optimizer.step()
                bt_ed = time.time()

                # save to tensorboard
                current_loss = loss.data.item()
                acc_train_loss += current_loss
                #writer.add_scalar('train/loss', current_loss, total_train_step)
                if current_batch % self.trn_step == 0:
                    self.logger.info("Epoch [{}/{}] trn-loss {:.4f}, time {:.2f} s".format(current_epoch+1, self.n_epoch,
                                                                                           acc_train_loss / (total_train_step+1),
                                                                                           bt_ed - bt_st))
                total_train_step += 1

            # validation every val_step times
            if current_epoch % self.val_step == 0:
                bt_st = time.time()
                net.eval()
                val_loss = 0
                n_val = 0
                for idx, (mix_val, sph_val) in enumerate(val_loader):
                    mix_val = mix_val.to(self.device)
                    sph_val = sph_val.to(self.device)
                    with torch.no_grad():
                        est = net(mix_val)

                    loss = loss_fn(est, lbl_mag_val)
                    val_loss += loss.data.item()
                    n_val +=1

                bt_ed = time.time()
                val_loss = val_loss / n_val
                if val_loss < minimum_loss_val:
                    minimum_loss_val = val_loss
                    # save model if it is better than before
                    torch.save(net.state_dict(), os.path.join(self.save_path, self.save_fname))
                #writer.add_scalar('val/loss', val_loss, total_train_step)
                self.logger.info("Epoch [{}/{}] val-loss {:.4f}, time {:.2f} s".format(current_epoch+1, self.n_epoch,
                                                                                        val_loss, bt_ed - bt_st))

            # update learning rate
            scheduler.step()
            epoch_ed = time.time()
            epoch_cost = round(epoch_ed - epoch_st, 2)
            if current_epoch % 100 == 0:
                self.logger.info("Epoch-{}/{}, time {:.2f} s".format(current_epoch+1, self.n_epoch, epoch_cost))
            current_epoch += 1
            #writer.close()

    def train(self, mix_dirname, sph_dirname,
              val_mix_dirname, val_sph_dirname,
              fs=16000, reload=True):
        loader = AudioLoader(mix_dirname, sph_dirname, self.n_batch, fs, True)
        loader_val = AudioLoader(val_mix_dirname, val_sph_dirname, self.n_batch, fs, True)
        record_path = os.path.join('./log_training', time.strftime('%Y%d%m'))
        if os.path.exists(record_path):
            shutil.rmtree(record_path)
        #writer = SummaryWriter(log_dir=record_path)

        net = DTLNNet(self.n_frame, self.n_hop)
        if reload is True and os.path.exists(os.path.join(self.save_path, self.save_fname)) is True:
            net.load_state_dict(torch.load(os.path.join(self.save_path, self.save_fname)))
        net.to(self.device)

        loss_fn = Model.loss_fn
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, amsgrad=False)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=self.lr_decay_period,
                                                    gamma=self.lr_decay_factor)

        total_train_step = 0
        current_epoch = 0
        minimum_loss_val = torch.finfo(torch.float32).max
        while current_epoch < self.n_epoch:
            epoch_st = time.time()
            # set net to train mode
            net.train()
            acc_train_loss = 0
            acc_train_frames = 0
            for current_batch, (mix, sph, n_sample) in enumerate(loader):
                bt_st = time.time()
                mix = mix.to(self.device)
                sph = sph.to(self.device)

                optimizer.zero_grad()
                with torch.enable_grad():
                    est = net(mix)
                loss = loss_fn(est, sph)
                loss.backward()
                optimizer.step()
                bt_ed = time.time()

                # save to tensorboard
                current_loss = loss.data.item()
                acc_train_loss += current_loss
                #writer.add_scalar('train/loss', current_loss, total_train_step)
                if current_batch % self.trn_step == 0:
                    self.logger.info("Epoch [{}-{}/{}] trn-loss {:.4f}, time {:.2f} s".format(current_batch+1, current_epoch+1,
                                                                                        self.n_epoch, acc_train_loss / (total_train_step+1),
                                                                                        bt_ed - bt_st))
                total_train_step += 1

            # validation every val_step times
            if current_batch % self.val_step == 0:
                bt_st = time.time()
                net.eval()
                val_loss = 0
                n_val = 0
                for idx, (mix_val, sph_val, n_sample_val) in enumerate(loader_val):
                    mix_val = mix_val.to(self.device)
                    sph_val = sph_val.to(self.device)
                    with torch.no_grad():
                        est = net(mix_val)

                    loss = loss_fn(est, lbl_mag_val)
                    val_loss += loss.data.item()
                    n_val +=1

                bt_ed = time.time()
                val_loss = val_loss / n_val
                if val_loss < minimum_loss_val:
                    minimum_loss_val = val_loss
                    # save model if it is better than before
                    torch.save(net.state_dict(), os.path.join(self.save_path, self.save_fname))
                #writer.add_scalar('val/loss', val_loss, total_train_step)
                self.logger.info("Epoch [{}/{}] val-loss {:.4f}, time {:.2f} s".format(current_epoch+1, self.n_epoch,
                                                                                        val_loss, bt_ed - bt_st))

            # update learning rate
            scheduler.step()
            epoch_ed = time.time()
            epoch_cost = round(epoch_ed - epoch_st, 2)
            if current_epoch % 100 == 0:
                self.logger.info("Epoch-{}/{}, time {:.2f} s".format(current_epoch+1, self.n_epoch, epoch_cost))
            current_epoch += 1
            #writer.close()

    def val(self):
        pass

    def test(self, test_dir):
        net = DTLNNet(self.n_frame, self.n_hop)
        net.load_state_dict(torch.load(os.path.join(self.save_path, self.save_fname)))
        net.to(self.device)
        net.eval()

        out_dir = os.path.join(os.path.dirname(test_dir), "post")
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        for f in os.listdir(test_dir):
            save_p = os.path.join(out_dir, f)
            fname = os.path.join(test_dir, f)
            data, fs = sf.read(fname)
            data = torch.from_numpy(data)
            data = torch.unsqueeze(data, dim=0).to(self.device)

            with torch.no_grad():
                est = net(data).cpu().numpy()
                est = est.squeeze()
            sf.write(save_p, est, fs)

if __name__ == "__main__":
    mix_dir = "E:\\datasets\\DNS-Challenge\\training_set_100h\\noisy"
    sph_dir = "E:\\datasets\\DNS-Challenge\\training_set_100h\\clean"
    val_mix_dir = "E:\\datasets\\DNS-Challenge\\training_set_100h\\val_noisy"
    val_sph_dir = "E:\\datasets\\DNS-Challenge\\training_set_100h\\val_clean"
    test_hdd_dir = "F:\\dataset\\val_clean"
    model = Model(epoches=10, batch_sz=64, n_frame=256, n_hop=128)
    #model.train(mix_dir, sph_dir, val_mix_dir, val_sph_dir)
    #model.train_with_dataset(mix_dir, sph_dir, val_mix_dir, val_sph_dir)
    #model.train(test_hdd_dir, test_hdd_dir, test_hdd_dir, test_hdd_dir)

    test_dir = "E:\\datasets\\DNS-Challenge\\datasets\\test_set\\real_recordings"
    model.test(test_dir)
