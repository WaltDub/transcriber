import os
import requests
import time
import subprocess
from pathlib import Path
from textwrap import shorten

API_SECRET = os.environ["API_SECRET"]
BASE_URL = os.environ["APPSCRIPT_URL"]

BASE_DIR = Path(__file__).parent.resolve()

WHISPER_BIN = (BASE_DIR / "whisper.cpp/build/bin/whisper-cli").resolve()
WHISPER_MODEL = (BASE_DIR / "models/ggml-base.en.bin").resolve()

LLAMA_BIN = (BASE_DIR / "llama.cpp/build/bin/llama-cli").resolve()
LLAMA_MODEL = (BASE_DIR / "models/llama-2-7b-chat.Q4_K_M.gguf").resolve()

DOWNLOAD_DIR = (BASE_DIR / "downloads")
DOWNLOAD_DIR.mkdir(exist_ok=True)



def get_next_job():
    url = f"{BASE_URL}?key={API_SECRET}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    job = r.json()
    return job if job and "row" in job else None


def extract_drive_file_id(url: str) -> str:
    if "id=" in url:
        return url.split("id=")[1].split("&")[0]
    parts = url.split("/")
    if "d" in parts:
        idx = parts.index("d")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return parts[-2]


def download_audio(drive_url: str, row: int) -> Path:
    file_id = extract_drive_file_id(drive_url)
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    out_path = DOWNLOAD_DIR / f"meeting_{row}.wav"
    print(f"  → Downloading audio for row {row}")
    r = requests.get(download_url, timeout=300)
    r.raise_for_status()

    with open(out_path, "wb") as f:
        f.write(r.content)

    return out_path


def transcribe_with_whisper(audio_path: Path) -> str:
    print(f"  → Transcribing {audio_path.name} with whisper.cpp")

    cmd = [
        str(WHISPER_BIN),
        "-m", str(WHISPER_MODEL),
        str(audio_path.resolve()),
        "-l", "auto",   # autodetect language
        "-otxt"
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Whisper failed: {result.stderr}")

    # Whisper writes <audio>.wav.txt by default
    txt_path = Path(str(audio_path) + ".txt")

    if txt_path.exists():
        transcript = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
    else:
        # Fall back to stdout if no file was created
        transcript = result.stdout.strip()

    # Debug logging
    print("  → Whisper stdout:", result.stdout[:200], "...")
    print("  → Whisper stderr:", result.stderr.strip())
    print("  → Transcript preview:", transcript[:200], "...")
    print("  → Transcript length:", len(transcript))

    return transcript



import re
import subprocess
from textwrap import shorten

def clean_llama_output(raw: str) -> str:
    """
    Cleans the raw output from llama-cli by removing:
    - ANSI escape codes
    - spinner characters
    - backspaces and erased characters
    - carriage returns
    - 'Loading model...' noise
    - prompt echo (optional)
    """

    # Remove ANSI escape sequences (colors, cursor moves, etc.)
    ansi_pattern = r"""\x1b\[[0-9;]*[A-Za-z]"""
    raw = re.sub(ansi_pattern, '', raw)

    # Remove backspaces and the characters they erase
    raw = re.sub(r'.\x08', '', raw)

    # Remove carriage returns
    raw = raw.replace('\r', '')

    # Remove spinner characters (| / - \)
    raw = re.sub(r'[|/\\\-]', '', raw)

    # Remove "Loading model..." lines
    raw = re.sub(r'Loading model.*', '', raw)

    # Remove repeated spaces
    raw = re.sub(r' +', ' ', raw)

    # Strip leading/trailing whitespace
    cleaned = raw.strip()

    return cleaned


def summarize_with_llama(transcript: str) -> str:
    """
    Runs llama-cli in non-conversation mode to generate a clean summary.
    Ensures:
    - No chat mode
    - No prompt echo in final output
    - No spinner or control characters
    - Timeout protection
    """

    print("  → Summarizing transcript with llama.cpp")

    # Keep transcript manageable for context window
    truncated = shorten(transcript, width=6000, placeholder="... [truncated]")

    # The actual prompt we want the model to complete
    prompt = (
        "You are an assistant that writes clear meeting summaries.\n\n"
        "Given the following meeting transcript, provide:\n"
        "1. A concise summary (5–10 sentences)\n"
        "2. A bullet list of key decisions\n"
        "3. A bullet list of action items with owners if mentioned\n\n"
        "Transcript:\n"
        f"{truncated}\n\n"
        "Now provide the summary and lists in plain text:\n"
    )

    # llama-cli command using ONLY flags supported by your version
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
        "--single-turn"   # ensures non-interactive summarization
    ]




    print("LLAMA COMMAND:", " ".join(cmd))

    try:
        # Run llama-cli with timeout protection
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Llama summarization timed out after 10 minutes")

    # If llama-cli itself failed
    if result.returncode != 0:
        raise RuntimeError(f"Llama failed: {result.stderr}")

    # Clean the raw output
    cleaned = clean_llama_output(result.stdout)

    # Remove the prompt echo if llama-cli printed it
    if prompt in cleaned:
        cleaned = cleaned.replace(prompt, "").strip()

    return cleaned



def submit_results(row: int, transcript: str, summary: str):
    url = f"{BASE_URL}?key={API_SECRET}"
    payload = {
        "row": row,
        "transcript": transcript,
        "summary": summary
    }
    print(f"  → Submitting results for row {row}")
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()


def process_all_jobs():
    while True:
        print("🔎 Requesting next job...")
        job = get_next_job()

        if not job:
            print("✅ No more jobs. Exiting.")
            break

        row = job["row"]
        audio_url = job["sourcefile"]
        print(f"🎧 Processing row {row}")

        try:
            audio_path = download_audio(audio_url, row)
            transcript = transcribe_with_whisper(audio_path)
            summary = summarize_with_llama(transcript)
            submit_results(row, transcript, summary)
            print(f"✅ Completed row {row}")

        except Exception as e:
            print(f"❌ Error processing row {row}: {e}")

        time.sleep(2)


if __name__ == "__main__":
    process_all_jobs()
