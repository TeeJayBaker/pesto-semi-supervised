import argparse
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import torch
import torchaudio
from src.models.pesto import PESTO
from src.models.networks.resnet1d import Resnet1d
from src.data.hcqt import HarmonicCQT as Preprocessor
from src.data.transforms import ToLogMagnitude
from src.data.audio_datamodule import mid_to_hz


def plot_csv(csv_path: str):
    # Read the CSV file without pandas
    times = []
    frequencies = []
    confidences = []

    with open(csv_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        # Assertions to ensure necessary columns exist
        fieldnames = reader.fieldnames
        assert "time" in fieldnames, "CSV must contain 'time' column."
        assert "frequency" in fieldnames, "CSV must contain 'frequency' column."
        assert "confidence" in fieldnames, "CSV must contain 'confidence' column."

        for row in reader:
            times.append(float(row["time"]))
            frequencies.append(float(row["frequency"]))
            confidences.append(float(row["confidence"]))

    # Convert lists to numpy arrays
    times = np.array(times)
    frequencies = np.array(frequencies)
    confidences = np.array(confidences)

    # Determine output PNG path from CSV path
    base, _ = os.path.splitext(csv_path)
    output_png_path = base + "_plot.png"

    # Plotting
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(times / 1000, frequencies, c=confidences, cmap="viridis_r", s=5)
    plt.colorbar(sc, label="Confidence")

    # Set logarithmic scale on the y-axis between 32 and 4096
    plt.yscale("log")
    plt.ylim(32, 4096)
    plt.xlim(times[0] / 1000, times[-1] / 1000)

    yticks = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    plt.yticks(yticks, [str(y) for y in yticks])

    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Pitch Predictions")
    plt.savefig(output_png_path)
    plt.close()
    print(f"Plot saved as {output_png_path}")


def run_inference(model_ckpt_path: str, audio_path: str, hop_size: float = 10.0):
    # Placeholder for model loading and inference
    # In practice, you'd load your model and run inference on the audio
    # For this example, we'll simulate some data
    assert os.path.exists(model_ckpt_path), "Model checkpoint path does not exist."
    assert os.path.exists(audio_path), "Audio path does not exist."

    try:
        x, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    checkpoint = torch.load(model_ckpt_path, map_location=torch.device("cpu"))

    hparams = checkpoint["hparams"]
    hcqt_params = checkpoint["hcqt_params"]
    state_dict = checkpoint["state_dict"]

    # instantiate preprocessors
    hop_length = int(hop_size * sr / 1000 + 0.5)
    preprocessor = Preprocessor(hop_length=hop_length, sr=sr, **hcqt_params)
    transform = ToLogMagnitude()

    # instantiate PESTO encoder
    encoder = Resnet1d(**hparams["encoder"])

    # Convert to mono and compute HCQT
    x = x.mean(dim=0)
    hcqt_kernels = preprocessor(x).squeeze(0).permute(2, 0, 1, 3)
    # (time, harmonics, freq_bins, 2)
    hcqt_kernels = transform(hcqt_kernels)  # Complex to log magnitude

    # instantiate main PESTO module and load its weights
    model = PESTO(
        encoder,
        torch.optim.Adam,  # dummy optimiser for now
        pitch_shift_kwargs=hparams["pitch_shift"],
        reduction=hparams["reduction"],
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Confidences are all 1 for now as the confidence head is linear
    pred, _ = model(hcqt_kernels)
    pred = mid_to_hz(pred)
    timesteps = torch.arange(pred.size(-1)) * hop_size

    # Calculate confidence naively from CQT energy
    confidence = hcqt_kernels.mean(dim=-2).max(dim=-1).values
    conf_min, conf_max = (
        confidence.min(dim=-1, keepdim=True).values,
        confidence.max(dim=-1, keepdim=True).values,
    )
    confidence = (confidence - conf_min) / (conf_max - conf_min)
    # Detach tensors

    pred = pred.detach().numpy()
    conf = confidence.detach().numpy()
    timesteps = timesteps.numpy()

    # Determine output CSV path from audio path
    base, _ = os.path.splitext(audio_path)
    csv_output_path = base + "_inference.csv"

    # Create DataFrame and save as CSV
    with open(csv_output_path, "w", newline="") as csvfile:
        fieldnames = ["time", "frequency", "confidence"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for t, f, c in zip(timesteps, pred, conf.squeeze()):
            writer.writerow({"time": t, "frequency": f, "confidence": c})

    print(f"Inference results saved as {csv_output_path}")
    return csv_output_path


def generate_spectrogram(audio_path: str):
    assert os.path.exists(audio_path), "Audio path does not exist."

    y, sr = librosa.load(audio_path)
    S = librosa.cqt(y, sr=sr)
    S_dB = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    # Determine output PNG path from audio path
    base, _ = os.path.splitext(audio_path)
    output_png_path = base + "_spectrogram.png"

    plt.figure(figsize=(10, 6))
    librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="cqt_hz")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    plt.savefig(output_png_path)
    plt.close()
    print(f"Spectrogram saved as {output_png_path}")


def main():
    parser = argparse.ArgumentParser(description="Process and plot audio data.")

    # Add mutually exclusive group for modes
    group = parser.add_argument_group("modes")
    group.add_argument(
        "-i", "--inference", action="store_true", help="Run model inference on audio."
    )
    group.add_argument(
        "-p", "--plot", action="store_true", help="Plot melody line from CSV."
    )
    group.add_argument(
        "-s", "--spec", action="store_true", help="Generate spectrogram from audio."
    )

    # Common arguments
    parser.add_argument("--csv", type=str, help="Path to CSV file.")
    parser.add_argument("--audio", type=str, help="Path to audio file.")
    parser.add_argument("--ckpt", type=str, help="Path to model checkpoint.")
    parser.add_argument("--hop", type=float, default=10.0, help="Hop size in ms.")

    args = parser.parse_args()
    if args.inference:
        assert args.ckpt, "Model checkpoint path must be provided for inference."
        assert args.audio, "Audio path must be provided for inference."
        assert args.csv is None, "Csv will be generated during inference."
        csv_path = run_inference(args.ckpt, args.audio)

    if args.plot:
        if not args.inference:
            assert args.csv, "CSV path must be provided for plotting."
            assert os.path.exists(args.csv), f"CSV file does not exist at {args.csv}"
            plot_csv(args.csv)
        else:
            plot_csv(csv_path)

    if args.spec:
        assert args.audio, "Audio path must be provided for spectrogram."
        generate_spectrogram(args.audio)


if __name__ == "__main__":
    main()
