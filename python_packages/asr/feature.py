# @File : old_feature_processor.py
# @Author: bunjun.li
# @Date : 2023-4-27
# @Desc : feature processor for voice
# @E-Mail: bunjun.li@ubtrobot.com

import io
from loguru import logger as log
from pathlib import Path
import sys
import wave
from scipy.fftpack import dct
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import io
import gc

sys.path.append(str(Path(__file__).resolve().parents[2]))

from python_packages import objectStorage

matplotlib.use("agg")


# predefined parameters
pre_emphasis_alpha = 0.97
frame_size, frame_stride = 0.025, 0.01
figure_axis_rate = 1.536
spectrogram_rate_fix = 2.577229905764966799308639143449


def low_freq_filter(noisy, sr=44100):
    hamming_window = 0.02
    if len(noisy) < (sr * hamming_window * 2):
        return noisy
    frame_dur = hamming_window  # frame duration, 20ms hamming window
    frame_length = int(frame_dur * sr)  # frame length in samples
    hamming_window = np.hamming(frame_length)  # 20ms hamming window

    n_noisy_frames = len(noisy) // (frame_length // 2) - 1
    n_start = 0
    frame_filter = np.log(np.arange(np.e**0.01, np.e, step=frame_length))
    enhanced_signal = np.zeros(noisy.shape)
    for j in range(n_noisy_frames):
        noisy_frame = noisy[n_start : n_start + frame_length] * hamming_window
        noisy_frame_fft = np.fft.fft(noisy_frame, n=frame_length)
        noisy_frame_fft *= frame_filter
        enhanced_frame = np.real(np.fft.ifft(noisy_frame_fft, n=frame_length))
        if j == 0:
            enhanced_signal[n_start : n_start + frame_length // 2] = enhanced_frame[
                0 : frame_length // 2
            ]
            overlap = enhanced_frame[frame_length // 2 : frame_length]
        else:
            enhanced_signal[n_start : n_start + frame_length // 2] = (
                overlap + enhanced_frame[0 : frame_length // 2]
            )
            overlap = enhanced_frame[frame_length // 2 : frame_length]
        n_start += frame_length // 2
    enhanced_signal[n_start : n_start + frame_length // 2] = overlap

    return enhanced_signal


def spectrogram_plot(data, data_length, figure_path, cmap="rainbow"):
    fig = plt.figure(figsize=(data_length, 5), dpi=100)
    plt.pcolor(data.T, cmap=cmap)
    plt.axis("off")
    plt.show()
    fig.clear()
    plt.close(fig)
    del fig


def fig_plot(data, data_length, figure_path):
    fig = plt.figure(figsize=(data_length, 5), dpi=100)
    plt.plot(data)
    plt.axis("off")
    plt.show()
    fig.clear()
    plt.close(fig)
    del fig


def blank_skip(sig):
    # skip and the blank in the starting and ending of wave to improve the effect of denoising
    sig = np.array(sig)
    start_idx = (sig != 0).argmax(axis=0)
    end_idx = (sig[::-1] != 0).argmax(axis=0)
    return sig[start_idx : -end_idx if end_idx else None]


def pre_emphasis(signal):
    # pre-emphasis for wave
    new_signal = np.append(signal[0], signal[1:] - pre_emphasis_alpha * signal[:-1])
    return new_signal


def frame_pad(emphasized_signal, fs):
    # padding signal for process
    frame_length, frame_step = int(round(frame_size * fs)), int(
        round(frame_stride * fs)
    )

    signal_length = len(emphasized_signal)
    num_frames = int(np.ceil(np.abs(signal_length - frame_length) / frame_step)) + 1
    pad_signal_length = (num_frames - 1) * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)

    indices = np.arange(0, frame_length).reshape(1, -1) + np.arange(
        0, num_frames * frame_step, frame_step
    ).reshape(-1, 1)
    frames = pad_signal[indices]
    return frames, frame_length


def frame_fft(frames, nfft=512):
    # frame to fft
    mag_frames = np.absolute(np.fft.rfft(frames, nfft))
    pow_frames = (1.0 / nfft) * (mag_frames**2)
    return mag_frames, pow_frames


def frame_mfcc(filter_banks):
    # fbank to mfcc
    num_ceps = 12
    mfcc_data = dct(filter_banks, type=2, axis=1, norm="ortho")[:, 1 : (num_ceps + 1)]
    return mfcc_data


class FeatureProcessor(object):
    r"""ASR processor

    A processor for feature.

    Attributes:
        model: vosk result_manager
    """

    def __init__(self, config=None):
        if config is None:
            config = {}
        self.nfft = config.get("nfft", 512)
        self.hamming_window = config.get("hamming_window", 0.02)
        self.a_dd = 0.98  # smoothing factor in priori update

        self.low_freq_mel = 0

        self.nfilt = 40
        self.num_ceps = 12

    def wiener(self, noisy, sr=44100):
        if len(noisy) < (sr * self.hamming_window * 2):
            return noisy
        frame_dur = self.hamming_window  # frame duration, 20ms hamming window
        frame_length = int(frame_dur * sr)  # frame length in samples
        hamming_window = np.hamming(frame_length)  # 20ms hamming window
        U = (
            sum(hamming_window**2) / frame_length
        )  # normalization constant for welch method

        len_first_120ms = int(sr * 0.12)
        first_120ms = noisy[0:len_first_120ms]

        number_of_frames_first_120ms = len_first_120ms // (frame_length // 2) - 1
        noise_psd = np.zeros(
            [
                frame_length,
            ]
        )
        n_start = 0

        for _ in range(number_of_frames_first_120ms):
            noise = first_120ms[n_start : n_start + frame_length]
            noise = noise * hamming_window
            noise_fft = np.fft.fft(noise, n=frame_length)
            noise_psd += (np.abs(noise_fft) ** 2) / (frame_length * U)
            n_start += int(frame_length / 2)
        noise_psd /= number_of_frames_first_120ms

        noise_psd += 1e-7

        n_noisy_frames = len(noisy) // (frame_length // 2) - 1
        n_start = 0
        enhanced_signal = np.zeros(noisy.shape)
        for j in range(n_noisy_frames):
            noisy_frame = noisy[n_start : n_start + frame_length] * hamming_window
            noisy_frame_fft = np.fft.fft(noisy_frame, n=frame_length)
            noisy_psd = (np.abs(noisy_frame_fft) ** 2) / (frame_length * U)
            """========================VAD====================="""
            posterior_snr = noisy_psd / noise_psd
            posterior_snr_prime = posterior_snr - 1
            posterior_snr_prime = np.maximum(0, posterior_snr_prime)
            if j == 0:
                priori_snr = self.a_dd + (1 - self.a_dd) * posterior_snr_prime
            else:
                g_prev = g
                priori_snr = (
                    self.a_dd * (g_prev**2) * posterior_snr_prev
                    + (1 - self.a_dd) * posterior_snr_prime
                )

            posterior_snr_prev = posterior_snr
            g = np.real((priori_snr / (priori_snr + 1)) ** 0.5)
            """===================end of VAD ==================="""
            enhanced_frame = np.real(np.fft.ifft(noisy_frame_fft * g, n=frame_length))
            if j == 0:
                enhanced_signal[n_start : n_start + frame_length // 2] = enhanced_frame[
                    0 : frame_length // 2
                ]
                overlap = enhanced_frame[frame_length // 2 : frame_length]
            else:
                enhanced_signal[n_start : n_start + frame_length // 2] = (
                    overlap + enhanced_frame[0 : frame_length // 2]
                )
                overlap = enhanced_frame[frame_length // 2 : frame_length]
            n_start += frame_length // 2
        enhanced_signal[n_start : n_start + frame_length // 2] = overlap

        return enhanced_signal

    def frame_fbank(self, pow_frames, fs):
        nfft = self.nfft
        low_freq_mel = self.low_freq_mel
        high_freq_mel = 2595 * np.log10(1 + (fs / 2) / 700)
        nfilt = self.nfilt
        mel_points = np.linspace(
            low_freq_mel, high_freq_mel, nfilt + 2
        )  # 所有的mel中心点，为了方便后面计算mel滤波器组，左右两边各补一个中心点
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        fbank = np.zeros((nfilt, int(nfft / 2 + 1)))  # 各个mel滤波器在能量谱对应点的取值
        area_code = (hz_points / (fs / 2)) * (nfft / 2)  # 各个mel滤波器中心点对应FFT的区域编码，找到有值的位置
        for i in range(1, nfilt + 1):
            left = int(area_code[i - 1])
            center = int(area_code[i])
            right = int(area_code[i + 1])
            for j in range(left, center):
                fbank[i - 1, j + 1] = (j + 1 - area_code[i - 1]) / (
                    area_code[i] - area_code[i - 1]
                )
            for j in range(center, right):
                fbank[i - 1, j + 1] = (area_code[i + 1] - (j + 1)) / (
                    area_code[i + 1] - area_code[i]
                )

        # some trouble when using numpy.dot in some env, it is an auto choice function
        filter_banks = np.dot(pow_frames, fbank.T)

        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filter_banks = 20 * np.log10(filter_banks)  # dB

        return filter_banks

    def frame_mfcc(self, filter_banks):
        num_ceps = self.num_ceps
        mfcc_data = dct(filter_banks, type=2, axis=1, norm="ortho")[
            :, 1 : (num_ceps + 1)
        ]
        return mfcc_data

    def __feature_process_func(self, wave_data, fs):
        nfft = self.nfft

        sig = np.frombuffer(wave_data, dtype=np.int16)

        # 去除空白
        unblank_sig = blank_skip(sig)

        # 频谱
        frames, frame_length = frame_pad(unblank_sig, fs)
        hamming = np.hamming(frame_length)
        hamming_frames = frames * hamming
        wave_fft_data = np.absolute(np.fft.rfft(hamming_frames, nfft))

        # 降噪
        denoise_sig = self.wiener(unblank_sig, fs)

        emphasized_signal = pre_emphasis(denoise_sig)
        frames, frame_length = frame_pad(emphasized_signal, fs)
        hamming_frames = frames * hamming
        denoise_fft_data, pow_frames = frame_fft(hamming_frames, nfft)

        # mfcc
        filter_banks = self.frame_fbank(pow_frames, fs)
        mfcc_data = frame_mfcc(filter_banks)

        return {
            "original_data": (unblank_sig, np.log(wave_fft_data + 1)),
            "preprocessed_data": (emphasized_signal, np.log(denoise_fft_data + 1)),
            "mfcc_data": mfcc_data,
            "denoise_data": denoise_sig,
        }

    def get_version(self):
        return {"version": "1.2.0"}

    def process_and_save_features(self, taskid, raw_data):
        audio = io.BytesIO(raw_data)
        wf = wave.open(audio, "rb")
        channels, sampwidth, fs, frames = wf.getparams()[:4]
        wave_data = wf.readframes(frames)
        wf.close()
        audio.close()

        features = self.__feature_process_func(wave_data, fs)
        result = self.__upload_features(
            taskid, features, raw_data, channels, sampwidth, fs
        )
        return result

    def __upload_features(self, taskid, features, raw_data, channels, sampwidth, fs):
        save_result = {}

        # 保存图片
        for filename, data in {
            "origin_signal": features["original_data"][0],
            "preprocess_signal": features["preprocessed_data"][0],
        }.items():
            buf = io.BytesIO()
            plt.figure(0, figsize=(10, 1), dpi=100)
            plt.ioff()
            plt.axis("off")
            plt.plot(data)
            plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            # plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            plt.close()

            save_files = objectStorage.upload_files(
                f"asr/{taskid}/{filename}.png",
                [buf.getvalue()],
            )
            buf.close()
            save_result[filename] = save_files[0]

        for filename, data in {
            "origin_fft": features["original_data"][1],
            "preprocess_fft": features["preprocessed_data"][1],
            "mfcc": features["mfcc_data"],
        }.items():
            buf = io.BytesIO()
            plt.figure(0, figsize=(10, 1), dpi=100)
            plt.ioff()
            plt.axis("off")
            plt.pcolormesh(data.T, cmap="rainbow")
            # fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            plt.close()

            save_files = objectStorage.upload_files(
                f"asr/{taskid}/{filename}.png", [buf.getvalue()]
            )
            buf.close()
            save_result[filename] = save_files[0]

        # 保存音频
        for filename, sig in {
            "original": features["original_data"][0],
            "preprocess": features["preprocessed_data"][0],
            "denoise": features["denoise_data"],
        }.items():
            buf = io.BytesIO()
            wf = wave.open(buf, "wb")
            wf.setnchannels(channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(fs)
            wf.writeframes(sig.astype(np.short).tobytes())
            wf.close()
            save_files = objectStorage.upload_files(
                f"asr/{taskid}/{filename}.wav",
                [buf.getvalue()],
            )
            buf.close()
            save_result[filename] = save_files[0]
        log.info("save", save_result)
        buf.close()

        gc.collect()
        return save_result


def benchmark():
    import time
    import re

    p = FeatureProcessor()
    start_time = time.time()
    for i in range(100000):
        path = "../../data/C30S.wav"
        f = open(path, "rb")
        radio_data = f.read()
        f.close()
        rec_result = p.process_and_save_features("123456", radio_data)
        print(rec_result)

    print(time.time() - start_time)


if __name__ == "__main__":
    benchmark()
