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
WHISPER_MODEL = (BASE_DIR / "models/ggml-base.bin").resolve()
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

    # Probe the file with ffmpeg to log its format
    probe_cmd = ["ffmpeg", "-i", str(raw_path)]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
    print(probe_result.stderr.strip())

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
    """Run whisper.cpp to transcribe the audio file into text."""
    print(f"Transcribing {audio_path.name} with whisper.cpp")

    # Initial prompt to bias Whisper toward preserving English terms
    initial_prompt = (
        "Behold engelske artikelnavne, bogtitler, projektnavne og tekniske termer uændret. "
        "Eksempler: Bayesian Inference, Deep Learning, Nature, Science, Perl, Conarro, Tsugu, "
        "LExO, causality, Actor–network theory, exponential organizations, problem tree, "
        "solution tree, problem solution tree."
    )

    cmd = [
        str(WHISPER_BIN),
        "-m", str(WHISPER_MODEL),
        str(audio_path.resolve()),
        "-l", "da",   # specify Danish language
        "-otxt",
        "--initial-prompt", initial_prompt
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Whisper failed: {result.stderr}")

    # Whisper writes e.g. meeting_2.txt (not meeting_2.wav.txt)
    txt_path = audio_path.with_suffix(".txt")

    if not txt_path.exists():
        raise RuntimeError(f"Transcript file not found: {txt_path}")

    transcript = txt_path.read_text(encoding="utf-8", errors="ignore").strip()

    print("Transcript preview:", transcript[:200], "...")
    print("Transcript length:", len(transcript))
    return transcript



def clean_llama_output(raw: str, prompt: str) -> str:
    """
    Extract only the summary text from llama-cli output.
    - Cut everything up to and including '(truncated)' OR any 'Resumé...' line
    - Cut everything from '[ Prompt:' onward
    - Remove prompt echo if present
    """
    text = raw
    print("RAW LLAMA OUTPUT PREVIEW:", text[:500])  # safe debug print


    # Remove prompt echo if present
    if prompt in text:
        text = text.replace(prompt, "")

    # Step 1: cut before/including '(truncated)' OR any 'Resumé...' marker
    start_marker1 = "(truncated)"
    # Match only the stable prefix, not the full examples list
    start_prefixes = [
        "Resumé (skriv kun på dansk",
        "Resumé (skriv kun på dansk,"
    ]

    if start_marker1 in text:
        idx = text.find(start_marker1)
        text = text[idx + len(start_marker1):]
    else:
        for prefix in start_prefixes:
            if prefix in text:
                idx = text.find(prefix)
                # cut after the prefix line
                text = text[idx + len(prefix):]
                break

    # Step 2: cut after '[ Prompt:'
    end_marker = "[ Prompt:"
    if end_marker in text:
        idx = text.find(end_marker)
        text = text[:idx]

    return text.strip()


def summarize_with_llama(transcript: str) -> str:
    """Run llama.cpp to generate a Danish summary of the transcript."""
    print("Summarizing transcript with llama.cpp")

    truncated = shorten(transcript, width=6000, placeholder="... [truncated]")

    prompt = (
        "Du er en assistent, der skriver klare mødereferater.\n\n"
        "Givet følgende mødetransskription, lav et resumé.\n\n"
        f"Transskription:\n{truncated}\n\n"
        "Resumé (skriv kun på dansk, men behold engelske artikelnavne, bogtitler, projektnavne og tekniske termer uændret. "
        "Eksempler: Bayesian Inference, Deep Learning, Nature, Science, Perl, Conarro, Tsugu, LExO, causality, "
        "Actor–network theory, exponential organizations, problem tree, solution tree, problem solution tree):\n"
    )

    cmd = [
        str(LLAMA_BIN),
        "-m", str(LLAMA_MODEL),
        "-p", prompt,
        "-n", "512",
        "-c", "4096",
        "-t", "4",
        "-b", "512",
        "--temp", "0.7",
        "--top-k", "40",
        "--top-p", "0.95",
        "--repeat-penalty", "1.1",
        "--single-turn",
        "--simple-io",          # suppresses banners and metadata
        "--no-display-prompt"   # suppresses prompt echo
    ]

    print("LLAMA COMMAND:", " ".join(cmd))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        raise RuntimeError("Llama summarization timed out after 10 minutes")

    if result.returncode != 0:
        raise RuntimeError(f"Llama failed: {result.stderr}")

    cleaned = clean_llama_output(result.stdout, prompt)
    return cleaned


def submit_results(row: int, transcript: str, summary: str):
    """Send transcript and summary back to the App Script backend."""
    url = f"{BASE_URL}?key={API_SECRET}"
    payload = {
        "row": row,
        "transcript": transcript,
        "summary": summary
    }
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
            summary = summarize_with_llama(transcript)
            submit_results(row, transcript, summary)
            print(f"Completed row {row}")
        except Exception as e:
            print(f"Error processing row {row}: {e}")

        time.sleep(2)


if __name__ == "__main__":
    process_all_jobs()
