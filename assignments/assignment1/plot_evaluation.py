import torch
import torchaudio
import matplotlib.pyplot as plt
from melbanks import LogMelFilterBanks

import scipy.io.wavfile as wav

# 1. Загрузка аудиофайла через scipy
wav_path = "10kHz_44100Hz_16bit_05sec.wav"  
sr, signal = wav.read(wav_path)

# Переводим в тензор и меняем размерность на (каналы, время)
signal = torch.tensor(signal, dtype=torch.float32)
if signal.ndim == 1:
    signal = signal.unsqueeze(0)
elif signal.ndim == 2:
    signal = signal.T 

# Если 2 канала - усредняем
if signal.shape[0] > 1:
    signal = signal.mean(dim=0, keepdim=True)

# Нормализуем значения к диапазону [-1.0, 1.0], как это делает torchaudio
if signal.abs().max() > 1.0:
    signal = signal / 32768.0

if sr != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    signal = resampler(signal)
    sr = 16000

# 2. Вычисление спектрограммы встроенным методом (native torchaudio)
melspec_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sr,
    n_fft=400,
    hop_length=160,
    n_mels=80,
    power=2.0
)
melspec = melspec_transform(signal)
log_melspec_torchaudio = torch.log(melspec + 1e-6)

# 3. Вычисление спектрограммы написанным классом 
logmel_module = LogMelFilterBanks(
    samplerate=sr,
    n_fft=400,
    hop_length=160,
    n_mels=80,
    power=2.0
)
log_melspec_custom = logmel_module(signal)

# 4. Проверка условия задания (Evaluation)
try:
    assert log_melspec_torchaudio.shape == log_melspec_custom.shape
    assert torch.allclose(log_melspec_torchaudio, log_melspec_custom, atol=1e-5)
    print("Проверка пройдена: Результаты torchaudio и реализация полностью совпадают")
except AssertionError:
    print("Ошибка: Результаты не совпадают!")

# 5. Построение сравнительных фигур для отчета
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(log_melspec_torchaudio[0].detach().numpy(), aspect='auto', origin='lower', cmap='viridis')
plt.title('Native (torchaudio.transforms.MelSpectrogram)')
plt.xlabel('Фреймы (Время)')
plt.ylabel('Мел-фильтры (Частота)')
plt.colorbar(format='%+2.0f dB')

plt.subplot(1, 2, 2)
plt.imshow(log_melspec_custom[0].detach().numpy(), aspect='auto', origin='lower', cmap='viridis')
plt.title('Custom (LogMelFilterBanks)')
plt.xlabel('Фреймы (Время)')
plt.ylabel('Мел-фильтры (Частота)')
plt.colorbar(format='%+2.0f dB')

plt.tight_layout()

plt.savefig("evaluation_plot.png", dpi=300, bbox_inches='tight')
print("График сохранен в файл 'evaluation_plot.png'!")
