import os
import penn
import numpy as np
import torch
import torchaudio
import itertools
from tqdm import tqdm
from pathlib import Path
from src.data.audio_datamodule import hz_to_mid
from src.callbacks.mir_eval import MIREvalCallback
from penn.core import resample, expected_frames


# Let's get all test files from the MIR-1K dataset
data_path = Path("data/MIR-1K/cache/mir-1k/test")
audio_files = [
    Path(f"{data_path}/{f}") for f in os.listdir(data_path) if f.endswith(".wav")
]
output_prefixes = [Path("penn/mir-1k") / file.stem for file in audio_files]

audio, sample_rate = torchaudio.load("data/MIR-1K/Vocals/abjones_1_01.wav")
# Hop size of 10ms
hopsize = 0.01
fmin = 30.0
fmax = 1000.0
gpu = "0"
batch_size = 64

# Select a checkpoint to use for inference. Selecting None will
# download and use FCNF0++ pretrained on MDB-stem-synth and PTDB
checkpoint = None

# Centers frames at hopsize / 2, 3 * hopsize / 2, 5 * hopsize / 2, ...
center = "half-hop"

# (Optional) Linearly interpolate unvoiced regions below periodicity threshold
interp_unvoiced_at = 0.065

penn.from_files_to_files(
    audio_files[:5],
    output_prefixes[:5],
    hopsize=hopsize,
    fmin=fmin,
    fmax=fmax,
    checkpoint=checkpoint,
    batch_size=batch_size,
    center=center,
    interp_unvoiced_at=interp_unvoiced_at,
    gpu=gpu,
)


def eval_penn(
    ds_name,
    hop_duration,
    skip_last=False,
    crop_audio=False,
    fmin=30.0,
    fmax=1000.0,
    gpu="0",
    batch_size=64,
    checkpoint=None,
    center="zero",
):
    hopsize_sec = hop_duration / 1000

    # download and use FCNF0++ pretrained on MDB-stem-synth and PTDB
    checkpoint = None

    data_dir = Path("data")

    audio_path = data_dir / Path(f"metadata/{ds_name}_test.csv")
    annot_path = data_dir / Path(f"metadata/{ds_name}_test_annot.csv")

    with audio_path.open("r") as f:
        audio_files = f.readlines()

    if annot_path is not None:
        with annot_path.open("r") as f:
            annot_files = f.readlines()
        annot_list = []

    pbar = tqdm(
        itertools.zip_longest(audio_files, annot_files, fillvalue=None),
        total=len(audio_files),
    )

    print(audio_files)
    preds = []
    for fname, annot in pbar:
        fname = fname.strip()
        pbar.set_description(fname)

        audio, sample_rate = torchaudio.load(data_dir / fname)

        # We just pad the audio with many samples
        if crop_audio:
            audio = torch.nn.functional.pad(
                audio, (0, int(50 * sample_rate * hopsize_sec)), mode="constant"
            )
        # print(expected_frames(audio.shape[-1], sample_rate, hopsize, center=center))
        out, conf = penn.from_audio(
            audio,
            sample_rate,
            hopsize=hopsize_sec,
            fmin=fmin,
            fmax=fmax,
            checkpoint=checkpoint,
            batch_size=batch_size,
            center=center,
            # interp_unvoiced_at=interp_unvoiced_at,
            gpu=gpu,
        )

        out = out[0]

        if annot is not None:
            annot = annot.strip()
            timesteps, freqs = np.loadtxt(
                data_dir / annot, delimiter=",", dtype=np.float32
            ).T
            annot_hop_duration = 1000 * (timesteps[1] - timesteps[0])

            if len(out) > len(freqs) and len(out) <= len(freqs) + 1:
                print(
                    f"There are {len(out) - len(freqs)} annoying frame in file {fname}. Cropping..."
                )
                out = out[: len(freqs)]

            # double-check for each file that hop sizes and lengths do match.
            # Since hop sizes are floats we put a tolerance of 1e-6 in the equality
            assert abs(annot_hop_duration - hop_duration) < 1e-6, (
                f"Inconsistency between {fname} and {annot}:\n"
                f"the resolution of the annotations ({len(freqs):d}) "
                f"does not match the number of CQT frames ({len(out):d}). "
                f"The hop duration between CQT frames should be identical "
                f"but got {annot_hop_duration:.1f} ms vs {hop_duration:.1f} ms. "
                f"Please either adjust the hop duration of the CQT or resample the annotations."
            )
            if crop_audio:
                out = out[: len(freqs)]
            if skip_last:
                # print(len(freqs))
                if len(freqs) > len(out):
                    freqs = freqs[: len(out)]
                # print(len(freqs))
            assert len(out) == len(freqs), (
                f"Inconsistency between {fname} and {annot}:"
                f"the resolution of the annotations ({len(freqs):d}) "
                f"does not match the number of CQT frames ({len(out):d}) "
                f"despite hop durations match. "
                f"Please check that your annotations are correct."
            )
            annot_list.extend(hz_to_mid(freqs))

        preds.extend(hz_to_mid(out.cpu()))

    return np.array(preds), np.array(annot_list)


penn.DECODER = "local_expected_value"

# preds, annot_list = eval_penn('mir-1k', 20, skip_last=False, crop_audio=True, center='half-hop')
# preds, annot_list = eval_penn('mdb', 2.902, skip_last=False, crop_audio=True, center='half-hop')
preds, annot_list = eval_penn(
    "ptdb", 10, skip_last=False, crop_audio=True, center="half-window"
)

mir_eval_cb = MIREvalCallback()
all_metrics = []
filter_unvoiced = True
pred = preds
annot = annot_list
if filter_unvoiced:
    voiced = np.array([_annot > 0 for _annot in annot_list])
    pred = preds[voiced]
    annot = annot_list[voiced]


metrics, optimal_shift = mir_eval_cb.evaluate(
    pred, annot, confidences=np.ones_like(pred), plot_cdf=False, print_metrics=True
)

print(metrics)
print(optimal_shift)
