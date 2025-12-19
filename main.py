import os
import requests
import time
import subprocess
from pathlib import Path
from textwrap import shorten

API_SECRET = os.environ["API_SECRET"]
BASE_URL = os.environ["APPSCRIPT_URL"]

WHISPER_BIN = Path("./whisper.cpp/main")
WHISPER_MODEL = Path("./models/ggml-base.en.bin")

LLAMA_BIN = Path("./llama.cpp/build/bin/llama")
LLAMA_MODEL = Path("./models/llama-3.1-8b.gguf")

DOWNLOAD_DIR = Path("./downloads")
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
        "-f", str(audio_path),
        "-l", "en",
        "-otxt"
    ]

    result = subprocess.run(
        cmd,
        cwd=audio_path.parent,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Whisper failed: {result.stderr}")

    txt_path = audio_path.with_suffix(".txt")
    if not txt_path.exists():
        return result.stdout.strip()

    return txt_path.read_text(encoding="utf-8", errors="ignore").strip()


def summarize_with_llama(transcript: str) -> str:
    print("  → Summarizing transcript with llama.cpp")

    truncated = shorten(transcript, width=6000, placeholder="... [truncated]")

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

    cmd = [
        str(LLAMA_BIN),
        "-m", str(LLAMA_MODEL),
        "-p", prompt,
        "-n", "512",
        "--temp", "0.7"
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Llama failed: {result.stderr}")

    return result.stdout.strip()


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
