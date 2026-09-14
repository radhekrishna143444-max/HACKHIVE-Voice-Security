import os
import subprocess
import tempfile

import numpy as np
import librosa
import imageio_ffmpeg


def extract_features(data):
    input_path = None
    wav_path = None

    try:
        # Save uploaded audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as f:
            f.write(data)
            input_path = f.name

        wav_path = input_path + ".wav"

        # Get bundled FFmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

        # Convert any supported audio to WAV
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                input_path,
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "wav",
                wav_path,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )

        # Load converted WAV
        y, sr = librosa.load(
            wav_path,
            sr=16000,
            mono=True
        )

        if len(y) == 0:
            raise ValueError("Audio file is empty.")

        # Normalize audio
        y = y / (np.max(np.abs(y)) + 1e-9)

        # MFCC
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=20
        )

        # Mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=40
        )

        # Spectral centroid
        sc = librosa.feature.spectral_centroid(
            y=y,
            sr=sr
        )

        # Zero crossing rate
        z = librosa.feature.zero_crossing_rate(y)

        # Combine features
        f = np.concatenate(
            [
                np.mean(mfcc, axis=1),
                np.std(mfcc, axis=1),
                np.mean(librosa.power_to_db(mel), axis=1),
                [
                    np.mean(sc),
                    np.std(sc),
                    np.mean(z),
                    np.std(z)
                ]
            ]
        )

        return f.astype(np.float32), sr

    except subprocess.CalledProcessError as e:
        error = e.stderr.decode(
            "utf-8",
            errors="ignore"
        )[-1000:]

        raise ValueError(
            f"Audio decoding failed: {error}"
        )

    except Exception as e:
        raise ValueError(
            f"Audio processing failed: {e}"
        )

    finally:
        # Delete temporary files
        if input_path and os.path.exists(input_path):
            os.remove(input_path)

        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)