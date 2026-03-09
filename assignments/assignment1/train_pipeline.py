import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS
from thop import profile
import matplotlib.pyplot as plt
from tqdm import tqdm

from melbanks import LogMelFilterBanks

import scipy.io.wavfile as wav

class BinarySpeechCommands(SPEECHCOMMANDS):
    def __init__(self, root="./data", subset="training"):
        os.makedirs(root, exist_ok=True)
        super().__init__(root=root, download=True, subset=subset)
        
        # Оставляем только 'yes' и 'no'
        new_walker = []
        for path in self._walker:
            label = os.path.basename(os.path.dirname(path))
            if label in ['yes', 'no']:
                new_walker.append(path)
        self._walker = new_walker
        
    def __getitem__(self, n):
        # Переопределяем метод загрузки напрямую через scipy, чтобы избежать ошибки torchaudio.load
        fileid = self._walker[n]
        sr, waveform = wav.read(fileid)
        
        # Приводим к виду (каналы=1, время)
        waveform = torch.tensor(waveform, dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 2:
            waveform = waveform.T
            
        # Усредняем и нормализуем (как делает torchaudio.load)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if waveform.abs().max() > 1.0:
            waveform = waveform / 32768.0
            
        # У нас sr всегда 16000 для этого датасета, но на всякий случай
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
            
        label = os.path.basename(os.path.dirname(fileid))
        label_idx = 1 if label == 'yes' else 0
        
        # Дополняем или обрезаем до 1 секунды (16000 сэмплов)
        if waveform.shape[1] < 16000:
            waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
        elif waveform.shape[1] > 16000:
            waveform = waveform[:, :16000]
            
        return waveform.squeeze(0), label_idx

class SpeechCNN(nn.Module):
    def __init__(self, in_channels, groups=1):
        super().__init__()
        c1, c2, c3 = 32, 64, 128
        
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=3, padding=1, groups=1),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(c1, c2, kernel_size=3, padding=1, groups=groups),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(c2, c3, kernel_size=3, padding=1, groups=groups),
            nn.BatchNorm1d(c3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Linear(c3, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class FullModel(nn.Module):
    def __init__(self, n_mels, groups=1):
        super().__init__()
        self.mel_extractor = LogMelFilterBanks(
            samplerate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=n_mels,
            power=2.0
        )
        self.cnn = SpeechCNN(in_channels=n_mels, groups=groups)
        
    def forward(self, x):
        features = self.mel_extractor(x)
        return self.cnn(features)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_flops(model_cnn, n_mels):
    dummy_input = torch.randn(1, n_mels, 101) 
    macs, _ = profile(model_cnn, inputs=(dummy_input, ), verbose=False)
    return macs * 2  

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for waveforms, labels in tqdm(loader, desc="Training", leave=False):
        waveforms, labels = waveforms.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(waveforms)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * waveforms.size(0)
    return running_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for waveforms, labels in tqdm(loader, desc="Evaluating", leave=False):
            waveforms, labels = waveforms.to(device), labels.to(device)
            outputs = model(waveforms)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    return correct / len(loader.dataset)

def run_experiment(n_mels, groups, train_loader, val_loader, test_loader, device, epochs=3):
    print(f"\n[{n_mels} mels | {groups} groups] Стартуем обучение...")
    model = FullModel(n_mels=n_mels, groups=groups).to(device)
    
    paramsCount = count_parameters(model.cnn)
    flopsCount = get_flops(model.cnn, n_mels)
    print(f"Параметры: {paramsCount:,} | FLOPs: {flopsCount:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    train_losses = []
    val_accs = []
    epoch_times = []
    
    for epoch in range(epochs):
        start_time = time.time()
        
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        
        epoch_time = time.time() - start_time
        
        train_losses.append(loss)
        val_accs.append(val_acc)
        epoch_times.append(epoch_time)
        
        print(f"Эпоха {epoch+1}/{epochs} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Время: {epoch_time:.2f} сек")
        
    test_acc = evaluate(model, test_loader, device)
    print(f"-> Test Accuracy: {test_acc:.4f}")
    
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    
    return {
        "train_losses": train_losses,
        "test_acc": test_acc,
        "avg_epoch_time": avg_epoch_time,
        "params": paramsCount,
        "flops": flopsCount
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Используемое устройство:", device)
    
    # 1. Загрузка данных
    print("Загрузка и подготовка SpeechCommands Dataset...")
    train_set = BinarySpeechCommands(subset="training")
    val_set = BinarySpeechCommands(subset="validation")
    test_set = BinarySpeechCommands(subset="testing")
    
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)
    
    print(f"Данные загружены: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
    
    # ==========================
    # Эксперимент 1: Разное количество n_mels
    # ==========================
    mels_list = [20, 40, 80]
    results_mels = {}
    
    for nm in mels_list:
        results_mels[nm] = run_experiment(nm, 1, train_loader, val_loader, test_loader, device, epochs=5)
        
    # График для Эксперимента 1
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for nm in mels_list:
        plt.plot(results_mels[nm]['train_losses'], label=f"n_mels={nm}")
    plt.title("Train Loss (Разные n_mels)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    test_accs = [results_mels[nm]['test_acc'] for nm in mels_list]
    plt.bar([str(nm) for nm in mels_list], test_accs, color='skyblue')
    plt.title("Test Accuracy vs n_mels")
    plt.xlabel("n_mels")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.1)
    
    plt.savefig("experiment1_mels.png", dpi=300)
    print("\nГрафик n_mels сохранен в experiment1_mels.png")
    
    # ==========================
    # Эксперимент 2: Групповые свертки (используем n_mels=80 как бейзлайн)
    # ==========================
    groups_list = [1, 2, 4, 8, 16] # 1 для сравнения
    results_groups = {}
    
    for gp in groups_list:
        results_groups[gp] = run_experiment(80, gp, train_loader, val_loader, test_loader, device, epochs=3)
        
    # Графики для Эксперимента 2
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    g_times = [results_groups[gp]['avg_epoch_time'] for gp in groups_list]
    plt.plot(groups_list, g_times, marker='o', color='orange')
    plt.title("Время эпохи vs Groups")
    plt.xlabel("Groups")
    plt.ylabel("Seconds")
    
    plt.subplot(1, 3, 2)
    g_params = [results_groups[gp]['params'] for gp in groups_list]
    plt.plot(groups_list, g_params, marker='o', color='green')
    plt.title("Кол-во параметров vs Groups")
    plt.xlabel("Groups")
    plt.ylabel("Params Count")
    
    plt.subplot(1, 3, 3)
    g_flops = [results_groups[gp]['flops'] / 1e6 for gp in groups_list]
    plt.plot(groups_list, g_flops, marker='o', color='purple')
    plt.title("FLOPs (Millions) vs Groups")
    plt.xlabel("Groups")
    plt.ylabel("FLOPs (M)")
    
    plt.tight_layout()
    plt.savefig("experiment2_groups.png", dpi=300)
    print("\nГрафик groups сохранен в experiment2_groups.png")
    print("ВЕСЬ ПАЙПЛАЙН ЗАВЕРШЕН!")

if __name__ == "__main__":
    main()
