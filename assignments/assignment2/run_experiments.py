import os
import torch
import torchaudio
import jiwer
from tqdm import tqdm
from wav2vec2decoder import Wav2Vec2Decoder

DATA_DIR_LIBRISPEECH = "data/librispeech_test_other"
DATA_DIR_EARNINGS22 = "data/earnings22_test"
CACHE_DIR = "cache"

def cache_logits(decoder, data_dir, cache_name):
    if not os.path.exists(data_dir):
        print(f"Dataset {data_dir} not found. Skipping.")
        return []
        
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}_logits.pt")
    
    if os.path.exists(cache_path):
        print(f"Loading cached logits from {cache_path}...")
        return torch.load(cache_path)
        
    print(f"Precomputing logits for {data_dir}...")
    dataset = [] 
    
    manifest_path = os.path.join(data_dir, "manifest.csv")
    if not os.path.exists(manifest_path):
        print(f"Manifest not found in {data_dir}. Skipping.")
        return []
        
    import csv
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(list(reader)):
            rel_path = row["path"]
            audio_path = os.path.join(".", rel_path)
            if not os.path.exists(audio_path):
                audio_path = os.path.join(data_dir, os.path.basename(rel_path))
            
            ref_text = row["text"].strip().lower()
            
            audio_input, sr = torchaudio.load(audio_path)
            assert sr == 16000
            
            inputs = decoder.processor(audio_input, return_tensors="pt", sampling_rate=16000)
            with torch.no_grad():
                logits = decoder.model(inputs.input_values.squeeze(0)).logits[0]
                
            dataset.append({
                "audio_path": audio_path,
                "ref": ref_text,
                "logits": logits.cpu()
            })
            
    torch.save(dataset, cache_path)
    return dataset

def evaluate_dataset(decoder, dataset, method="greedy"):
    references = []
    hypotheses = []
    
    for item in dataset:
        ref = item["ref"]
        logits = item["logits"]
        
        scaled_logits = logits / decoder.temperature
        
        if method == "greedy":
            hyp = decoder.greedy_decode(scaled_logits)
        elif method == "beam":
            hyp = decoder.beam_search_decode(scaled_logits)
        elif method == "beam_lm":
            hyp = decoder.beam_search_with_lm(scaled_logits)
        elif method == "beam_lm_rescore":
            beams = decoder.beam_search_decode(scaled_logits, return_beams=True)
            hyp = decoder.lm_rescore(beams)
        else:
            raise ValueError(f"Unknown method {method}")
            
        references.append(ref)
        hypotheses.append(hyp)
        
    wer = jiwer.wer(references, hypotheses)
    cer = jiwer.cer(references, hypotheses)
    return wer, cer

def run_all():
    print("--- ASSIGNMENT 2 EXPERIMENTS ---")
    decoder = Wav2Vec2Decoder(lm_model_path=None)
    
    ls_data = cache_logits(decoder, DATA_DIR_LIBRISPEECH, "librispeech")
    e22_data = cache_logits(decoder, DATA_DIR_EARNINGS22, "earnings22")
    
    if not ls_data:
        print("Librispeech data missing! Please run this script where `data/` exists.")
        return

    # Task 1: Greedy on Librispeech
    print("\nTask 1: Greedy Decoding")
    wer, cer = evaluate_dataset(decoder, ls_data, method="greedy")
    print(f"Librispeech Greedy -> WER: {wer:.2%}, CER: {cer:.2%}")
    
    # Task 2: Beam Search on Librispeech with different beam widths
    print("\nTask 2: Beam Search Decoding (Beam Width sweep)")
    for bw in [1, 3, 10, 50]:
        decoder.beam_width = bw
        wer, cer = evaluate_dataset(decoder, ls_data, method="beam")
        print(f"Librispeech Beam (w={bw}) -> WER: {wer:.2%}, CER: {cer:.2%}")
    decoder.beam_width = 3 # restore default
    
    # Task 3: Temperature on Greedy
    print("\nTask 3: Temperature scaling (Greedy)")
    for t in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        decoder.temperature = t
        wer, cer = evaluate_dataset(decoder, ls_data, method="greedy")
        print(f"Librispeech Greedy (T={t}) -> WER: {wer:.2%}, CER: {cer:.2%}")
    decoder.temperature = 1.0 # restore default
    
    # Load 3-gram for Remaining Tasks
    lm_3gram_path = "lm/3-gram.pruned.1e-7.arpa.gz"
    if not os.path.exists(lm_3gram_path):
        print(f"\nLanguage model {lm_3gram_path} missing. Skipping Tasks 4-9.")
        return
        
    print(f"\nLoading 3-gram LM from {lm_3gram_path}...")
    try:
        import kenlm
        decoder.lm_model = kenlm.Model(lm_3gram_path)
    except ImportError:
        print("KenLM is not installed. Skipping Language Model tasks.")
        return
    
    # Task 4: Shallow Fusion grid search
    alphas = [0.01, 0.1, 0.5, 1.0, 2.0]
    betas = [0.0, 0.5, 1.0]
    
    best_sf_wer = float('inf')
    best_sf_params = (1.0, 1.0)
    
    print("\nTask 4: Shallow Fusion Grid Search")
    for alpha in alphas:
        for beta in betas:
            decoder.alpha = alpha
            decoder.beta = beta
            wer, cer = evaluate_dataset(decoder, ls_data, method="beam_lm")
            print(f"  alpha={alpha}, beta={beta} -> WER: {wer:.2%}")
            if wer < best_sf_wer:
                best_sf_wer = wer
                best_sf_params = (alpha, beta)
                
    print(f"Best Shallow Fusion params: alpha={best_sf_params[0]}, beta={best_sf_params[1]}")
    
    # Task 6: Rescoring grid search
    print("\nTask 6: LM Rescoring Grid Search")
    best_rs_wer = float('inf')
    best_rs_params = (1.0, 1.0)
    for alpha in alphas:
        for beta in betas:
            decoder.alpha = alpha
            decoder.beta = beta
            wer, cer = evaluate_dataset(decoder, ls_data, method="beam_lm_rescore")
            print(f"  alpha={alpha}, beta={beta} -> WER: {wer:.2%}")
            if wer < best_rs_wer:
                best_rs_wer = wer
                best_rs_params = (alpha, beta)
                
    print(f"Best Rescoring params: alpha={best_rs_params[0]}, beta={best_rs_params[1]}")
    
    # Task 7: Earnings22 eval
    print("\nTask 7: Earnings22 (Out-of-Domain) Evaluation")
    if not e22_data:
        print("Earnings22 data missing! Skip.")
    else:
        wer, cer = evaluate_dataset(decoder, e22_data, method="greedy")
        print(f"Earnings22 Greedy -> WER: {wer:.2%}, CER: {cer:.2%}")
        
        wer, cer = evaluate_dataset(decoder, e22_data, method="beam")
        print(f"Earnings22 Beam -> WER: {wer:.2%}, CER: {cer:.2%}")
        
        decoder.alpha, decoder.beta = best_sf_params
        wer, cer = evaluate_dataset(decoder, e22_data, method="beam_lm")
        print(f"Earnings22 Beam+LM (SF) -> WER: {wer:.2%}, CER: {cer:.2%}")
        
        decoder.alpha, decoder.beta = best_rs_params
        wer, cer = evaluate_dataset(decoder, e22_data, method="beam_lm_rescore")
        print(f"Earnings22 Beam+LM (RS) -> WER: {wer:.2%}, CER: {cer:.2%}")
        
    print("\nAll experiments finished. Please copy the outputs to your report!")

if __name__ == "__main__":
    run_all()
