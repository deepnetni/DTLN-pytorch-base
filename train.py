import os

def train(self, save_p, train_mix_p, train_sph_p, val_mix_p, val_sph_p):
    save_p = './models_' + save_p + '/'
    if not os.path.exists(save_p):
        os.makedirs(save_p)

    # TODO create callback for the adaptive learning rate

    # TODO create callback for early stopping

    # calculate length of audio chunks in samples
    len_in_samples = int(np.fix(self.fs * self.len_samples / self.block_shift) * self.block_shift)

    # create data generator for training data
    gen_input = audio_generator(train_mix_p, train_sph_p,
                                len_in_samples, self.fs, shuffle=True)

    # calculate number of training steps in one epoch


if __name__ == '__main__':
    pass
