import os
import requests
import time
import subprocess
from pathlib import Path
from textwrap import shorten

# Environment variables provided by GitHub Actions secrets
API_SECRET = os.environ["API_SECRET"]
BASE_URL = os.environ["APPSCRIPT_URL"]

# Base directory of the repository
BASE_DIR = Path(__file__).parent.resolve()

# Paths to compiled binaries
WHISPER_BIN = (BASE_DIR / "whisper.cpp/build/bin/whisper-cli").resolve()
LLAMA_BIN = (BASE_DIR / "llama.cpp/build/bin/llama-cli").resolve()

# Paths to model files
WHISPER_MODEL = (BASE_DIR / "models/ggml-medium.bin").resolve()
LLAMA_MODEL = (BASE_DIR / "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf").resolve()

# Directory for temporary audio downloads
DOWNLOAD_DIR = (BASE_DIR / "downloads")
DOWNLOAD_DIR.mkdir(exist_ok=True)


def get_next_job():
    """Fetch the next transcription job from the App Script backend."""
    url = f"{BASE_URL}?key={API_SECRET}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    job = r.json()
    print("Fetched job payload:", job)   # <-- add this
    return job if job and "row" in job else None


def extract_drive_file_id(url: str) -> str:
    """Extract the Google Drive file ID from a URL."""
    if "id=" in url:
        return url.split("id=")[1].split("&")[0]
    parts = url.split("/")
    if "d" in parts:
        idx = parts.index("d")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return parts[-2]


import gdown

def download_audio(drive_url: str, row: int) -> Path:
    """Download audio from Google Drive and convert to 16kHz mono WAV."""
    file_id = extract_drive_file_id(drive_url)

    raw_path = DOWNLOAD_DIR / f"meeting_{row}.input"
    print(f"Downloading audio for row {row} via gdown")
    gdown.download(id=file_id, output=str(raw_path), quiet=False)

    # Convert to 16kHz mono PCM WAV for whisper.cpp
    wav_path = DOWNLOAD_DIR / f"meeting_{row}.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(raw_path),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(wav_path)
    ]
    subprocess.run(cmd, check=True)

    raw_path.unlink(missing_ok=True)
    return wav_path


def transcribe_with_whisper(audio_path: Path) -> str:
    """Run whisper.cpp to transcribe the audio file into text (file output only)."""
    print(f"Transcribing {audio_path.name} with whisper.cpp")

    cmd = [
        str(WHISPER_BIN),
        "-m", str(WHISPER_MODEL),
        str(audio_path.resolve()),
        "-l", "da",
        "-otxt"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Whisper failed: {result.stderr}")

    txt_path = Path(str(audio_path) + ".txt")  # meeting_X.wav.txt
    if not txt_path.exists() or txt_path.stat().st_size == 0:
        raise RuntimeError(f"Transcript file missing or empty: {txt_path}")

    transcript = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
    print("Transcript preview:", transcript[:200], "...")
    print("Transcript length:", len(transcript))
    return transcript


def clean_llama_output(raw: str) -> str:
    """
    Clean llama-cli output so only the summary text remains.
    - Remove banners and metadata
    - Keep only the text after the last 'Resumé:' (or '(truncated)' fallback)
    - Remove trailing 'Exiting...'
    """
    text = raw

    # Cut before 'Exiting...'
    if "Exiting..." in text:
        text = text.split("Exiting...", 1)[0]

    # Prefer the last 'Resumé:' marker
    if "Resumé:" in text:
        idx = text.rfind("Resumé:")
        text = text[idx + len("Resumé:"):]
    elif "(truncated)" in text:
        idx = text.rfind("(truncated)")
        text = text[idx + len("(truncated)"):]
    else:
        # fallback: drop everything before the first 'Transskription:' if present
        if "Transskription:" in text:
            idx = text.find("Transskription:")
            text = text[idx + len("Transskription:"):]

    return text.strip()


def summarize_with_llama(transcript: str, row: int) -> str:
    """Run llama.cpp to generate an English narrative summary grouped by themes, forbidding invention."""
    print("Summarizing transcript with llama.cpp")

    truncated = shorten(transcript, width=6000, placeholder="... [truncated]")
    prompt = (
        "You are an assistant that writes clear discussion summaries.\n\n"
        "Task:\n"
        "- Write a narrative summary in English.\n"
        "- Group content by themes (topics, ideas, problems).\n"
        "- Use only information from the transcript.\n"
        "- Do not invent new details or examples.\n"
        "- Keep proper names and technical terms exactly as they appear.\n\n"
        f"Transcript:\n{truncated}\n\nSummary (in English):\n"
    )

    result = subprocess.run(
        [
            str(LLAMA_BIN),
            "-m", str(LLAMA_MODEL),
            "-p", prompt,
            "-n", "512",
            "-c", "4096",
            "--single-turn",
            "--simple-io",
            "--no-display-prompt",
            "--no-show-timings",
            "--log-disable",
            "--log-colors", "off"
        ],
        capture_output=True,
        text=True,
        timeout=600
    )

    if result.returncode != 0:
        raise RuntimeError(f"Llama failed: {result.stderr}")

    cleaned = clean_llama_output(result.stdout)
    if not cleaned:
        raise RuntimeError("Llama produced no summary output")

    print("Summary preview:", cleaned[:200], "...")
    return cleaned


def submit_results(row: int, transcript: str, summary: str):
    """Send transcript and summary back to the App Script backend."""
    url = f"{BASE_URL}?key={API_SECRET}"
    payload = {"row": row, "transcript": transcript, "summary": summary}
    print(f"Submitting results for row {row}")
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()


def process_all_jobs():
    """Main loop: fetch jobs, process audio, transcribe, summarize, submit results."""
    while True:
        print("Requesting next job...")
        job = get_next_job()
        if not job:
            print("No more jobs. Exiting.")
            break

        row = job["row"]
        audio_url = job["sourcefile"]
        print(f"Processing row {row}")

        try:
            audio_path = download_audio(audio_url, row)
            transcript = transcribe_with_whisper(audio_path)
            summary = summarize_with_llama(transcript, row)
            submit_results(row, transcript, summary)
            print(f"Completed row {row}")
        except Exception as e:
            print(f"Error processing row {row}: {e}")

        time.sleep(2)


if __name__ == "__main__":
    process_all_jobs()
