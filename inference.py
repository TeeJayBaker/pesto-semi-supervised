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


def plot_csv(csv_path):
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
    sc = plt.scatter(times / 1000, frequencies, c=confidences, cmap="viridis", s=5)
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


# TODO fix the simple inference function for new models
def run_inference(model_ckpt_path, audio_path):
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
    import ipdb

    ipdb.set_trace()
    hparams = checkpoint["hparams"]
    hcqt_params = checkpoint["hcqt_params"]
    state_dict = checkpoint["state_dict"]
    hop_size = 10.0

    # instantiate preprocessor
    hop_length = int(hop_size * sr / 1000 + 0.5)
    preprocessor = Preprocessor(hop_length=hop_length, sr=sr, **hcqt_params)

    # instantiate PESTO encoder
    encoder = Resnet1d(**hparams["encoder"])

    # Conver to mono
    x = x.mean(dim=0)

    hcqt_kernels = preprocessor(x)

    # instantiate main PESTO module and load its weights
    model = PESTO(
        encoder,
        crop_kwargs=hparams["pitch_shift"],
        reduction=hparams["reduction"],
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    pred, conf = model(hcqt_kernels, sr=sr, return_activations=False)
    timesteps = torch.arange(pred.size(-1)) * hop_size

    # Determine output CSV path from audio path
    base, _ = os.path.splitext(audio_path)
    csv_output_path = base + "_inference.csv"

    # Create DataFrame and save as CSV
    with open(csv_output_path, "w", newline="") as csvfile:
        fieldnames = ["time", "frequency", "confidence"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for t, f, c in zip(timesteps, pred, conf):
            writer.writerow({"time": t, "frequency": f, "confidence": c})

    print(f"Inference results saved as {csv_output_path}")
    return csv_output_path


def generate_spectrogram(audio_path):
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

    args = parser.parse_args()
    if args.inference:
        assert args.ckpt, "Model checkpoint path must be provided for inference."
        assert args.audio, "Audio path must be provided for inference."
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
