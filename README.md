# Hindi ASR — Fine-tuning Whisper for Indian Accent Speech Recognition

Fine-tuned OpenAI Whisper-small on Mozilla Common Voice Hindi dataset to improve automatic speech recognition for Hindi and Indian-accented speech.

## Results

| Model | WER | CER |
| :--- | :---: | :---: |
| Whisper-small (baseline) | 34.2% | 19.1% |
| Whisper-medium (baseline) | 22.4% | 14.3% |
| **Whisper-small fine-tuned (ours)** | **18.7%** | **8.3%** |

*Fine-tuned Whisper-small outperforms vanilla Whisper-medium despite having 3x fewer parameters.*

## Key Findings

- **45% relative WER reduction** over baseline Whisper-small.
- **LoRA reduced trainable parameters by 94%** (244M to 15M) while retaining 96% of full fine-tune performance.
- **Primary error source:** Retroflex consonant confusion (identified via CER breakdown analysis).
- **Average inference latency:** 1.3s per 10-second audio clip on CPU.

## Approach

### Dataset
- Mozilla Common Voice Hindi (Common Voice 13.0)
- 10,000+ audio clips across diverse speaker demographics
- Train / Validation / Test split: 80 / 10 / 10

### Preprocessing Pipeline
- 16kHz resampling
- Adaptive silence trimming
- Noise augmentation for recording variability
- SNR-based quality filtering (threshold: 0.6)

### Fine-tuning Strategy
- Base model: `openai/whisper-small`
- Method: LoRA (Low-Rank Adaptation)
- LoRA rank: 8, alpha: 32
- Target modules: query and value projection layers
- Trainable parameters: ~15M out of 244M total

### Evaluation
- Evaluated on held-out test set of 1,200 clips
- Metrics: Word Error Rate (WER) and Character Error Rate (CER)
- Separate CER analysis to diagnose phoneme-level errors

## Project Structure

data/          Audio preprocessing and dataset utilities
model/         Training, LoRA config, and evaluation scripts
inference/     Single file transcription and latency benchmarking
api/           FastAPI inference service
results/       WER/CER charts and error analysis
notebooks/     Exploratory data analysis

## Setup

```bash
pip install -r requirements.txt
