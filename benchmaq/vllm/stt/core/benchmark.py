import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
import wave
from typing import Any, Dict, List, Optional


def _get_audio_duration(audio_file: str) -> float:
    """Get audio duration in seconds.

    Uses stdlib wave module for WAV files, falls back to ffprobe for others.
    """
    ext = os.path.splitext(audio_file)[1].lower()

    if ext == ".wav":
        with wave.open(audio_file, "rb") as w:
            return w.getnframes() / w.getframerate()

    # Fallback: ffprobe for non-WAV formats
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                audio_file,
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    raise RuntimeError(
        f"Cannot determine audio duration for '{audio_file}'. "
        "Use a .wav file or install ffprobe."
    )


def _get_content_type(audio_file: str) -> str:
    """Get MIME content type from file extension."""
    ext = os.path.splitext(audio_file)[1].lower()
    types = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
        ".webm": "audio/webm",
    }
    return types.get(ext, "application/octet-stream")


async def _send_request(
    url: str,
    audio_bytes: bytes,
    filename: str,
    content_type: str,
    model: str,
    semaphore: asyncio.Semaphore,
    extra_data: dict,
) -> Dict[str, Any]:
    """Send a single transcription request and measure timing."""
    import requests

    async with semaphore:
        start = time.perf_counter()
        try:
            result = await asyncio.to_thread(
                _post_transcription,
                url, audio_bytes, filename, content_type, model, extra_data,
            )
            elapsed = time.perf_counter() - start
            return {
                "success": True,
                "elapsed": elapsed,
                "text": result.get("text", ""),
                "response": result,
            }
        except Exception as e:
            elapsed = time.perf_counter() - start
            return {
                "success": False,
                "elapsed": elapsed,
                "text": "",
                "error": str(e),
            }


def _post_transcription(
    url: str,
    audio_bytes: bytes,
    filename: str,
    content_type: str,
    model: str,
    extra_data: dict,
) -> dict:
    """Blocking POST request for audio transcription."""
    import requests

    files = {"file": (filename, audio_bytes, content_type)}
    data = {"model": model}
    data.update(extra_data)

    resp = requests.post(url, files=files, data=data, timeout=300)
    resp.raise_for_status()
    return resp.json()


async def _run_async(
    url: str,
    audio_bytes: bytes,
    filename: str,
    content_type: str,
    model: str,
    num_requests: int,
    max_concurrency: int,
    request_rate: float,
    extra_data: dict,
) -> List[Dict[str, Any]]:
    """Run concurrent transcription requests with rate limiting."""
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = []

    interval = 0.0 if request_rate == float("inf") else 1.0 / request_rate

    for i in range(num_requests):
        task = asyncio.create_task(
            _send_request(
                url, audio_bytes, filename, content_type,
                model, semaphore, extra_data,
            )
        )
        tasks.append(task)
        if interval > 0 and i < num_requests - 1:
            await asyncio.sleep(interval)

    return await asyncio.gather(*tasks)


def _format_results(
    results: List[Dict[str, Any]],
    audio_file: str,
    audio_duration: float,
    total_time: float,
    num_requests: int,
    max_concurrency: int,
) -> str:
    """Format benchmark results into human-readable output."""
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    success_times = [r["elapsed"] for r in successes]

    lines = []
    lines.append(f"Testing: {max_concurrency} max concurrent requests")
    lines.append("=" * 60)

    # Results summary
    lines.append("Results:")
    lines.append(
        f"  Total: {num_requests} | Success: {len(successes)} | Failed: {len(failures)}"
    )
    success_rate = (len(successes) / num_requests * 100) if num_requests > 0 else 0
    lines.append(f"  Success Rate: {success_rate:.1f}%")

    # Timing
    lines.append("Timing:")
    lines.append(f"  Total Time: {total_time:.2f}s")
    throughput = num_requests / total_time if total_time > 0 else 0
    lines.append(f"  Throughput: {throughput:.2f} req/s")

    # Per-request processing time
    if success_times:
        lines.append("Processing Time per Request:")
        lines.append(
            f"  Min: {min(success_times):.2f}s | Max: {max(success_times):.2f}s"
        )
        mean_time = statistics.mean(success_times)
        median_time = statistics.median(success_times)
        lines.append(f"  Mean: {mean_time:.2f}s | Median: {median_time:.2f}s")

        # RTF
        lines.append("RTF (Real-Time Factor):")
        lines.append(f"  Audio file: {audio_file}")
        lines.append(f"  Audio duration: {audio_duration:.2f}s")
        lines.append(f"  Mean processing time: {mean_time:.2f}s")
        mean_rtf = mean_time / audio_duration if audio_duration > 0 else float("inf")
        lines.append(f"  Mean RTF: {mean_rtf:.3f}")
        min_rtf = min(success_times) / audio_duration if audio_duration > 0 else float("inf")
        max_rtf = max(success_times) / audio_duration if audio_duration > 0 else float("inf")
        lines.append(f"  Min RTF: {min_rtf:.3f} | Max RTF: {max_rtf:.3f}")
    else:
        lines.append("Processing Time per Request:")
        lines.append("  No successful requests")

    # Sample audio output
    if successes:
        lines.append("Audio Output:")
        lines.append(f"  {successes[0]['text']}")

    # Errors
    if failures:
        lines.append("Errors:")
        unique_errors = list({r.get("error", "unknown") for r in failures})
        for err in unique_errors[:5]:
            lines.append(f"  - {err}")

    return "\n".join(lines)


def _build_json_result(
    result_name: str,
    model: str,
    audio_file: str,
    audio_duration: float,
    num_requests: int,
    max_concurrency: int,
    request_rate: float,
    total_time: float,
    results: List[Dict[str, Any]],
) -> dict:
    """Build structured JSON result for saving."""
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    success_times = [r["elapsed"] for r in successes]

    output = {
        "benchmark_name": result_name,
        "model": model,
        "audio_file": audio_file,
        "audio_duration_s": audio_duration,
        "num_requests": num_requests,
        "max_concurrency": max_concurrency,
        "request_rate": request_rate,
    }

    result_data = {
        "total": num_requests,
        "success": len(successes),
        "failed": len(failures),
        "success_rate": (len(successes) / num_requests * 100) if num_requests > 0 else 0,
        "total_time_s": round(total_time, 3),
        "throughput_rps": round(num_requests / total_time, 3) if total_time > 0 else 0,
    }

    if success_times:
        mean_time = statistics.mean(success_times)
        result_data["processing_time"] = {
            "min_s": round(min(success_times), 3),
            "max_s": round(max(success_times), 3),
            "mean_s": round(mean_time, 3),
            "median_s": round(statistics.median(success_times), 3),
        }
        result_data["rtf"] = {
            "mean": round(mean_time / audio_duration, 3) if audio_duration > 0 else None,
            "min": round(min(success_times) / audio_duration, 3) if audio_duration > 0 else None,
            "max": round(max(success_times) / audio_duration, 3) if audio_duration > 0 else None,
        }

    output["results"] = result_data

    if successes:
        output["sample_output"] = successes[0]["text"]
    if failures:
        output["errors"] = list({r.get("error", "unknown") for r in failures})[:10]

    return output


def run_benchmark(
    model: str,
    port: int,
    result_name: str,
    results_config: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """Run STT benchmark with concurrent audio transcription requests.

    Args:
        model: Model name (sent in API request body)
        port: vLLM server port
        result_name: Name for this benchmark run
        results_config: Dict with save_result, result_dir
        **kwargs: Benchmark parameters:
            audio_file: Path to audio file (required)
            num_requests: Number of requests to send (default: 10)
            max_concurrency: Max concurrent requests (default: 10)
            request_rate: Requests per second, float or "inf" (default: inf)
            endpoint: API endpoint (default: /v1/audio/transcriptions)
            language: Language hint (optional)
            response_format: Response format (optional)
    """
    print()
    print("=" * 64)
    print(f"STT BENCHMARK: {result_name}")
    print("=" * 64)
    sys.stdout.flush()

    # Extract benchmark params
    audio_file = kwargs.pop("audio_file", None)
    if not audio_file:
        raise ValueError("'audio_file' is required in bench config")
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    num_requests = int(kwargs.pop("num_requests", 10))
    max_concurrency = int(kwargs.pop("max_concurrency", 10))
    request_rate_raw = kwargs.pop("request_rate", "inf")
    request_rate = float(request_rate_raw)
    endpoint = kwargs.pop("endpoint", "/v1/audio/transcriptions")

    # Remaining kwargs become extra form data (language, response_format, etc.)
    extra_data = {}
    for key in list(kwargs.keys()):
        extra_data[key] = kwargs.pop(key)

    # Get audio info
    audio_duration = _get_audio_duration(audio_file)
    filename = os.path.basename(audio_file)
    content_type = _get_content_type(audio_file)

    with open(audio_file, "rb") as f:
        audio_bytes = f.read()

    url = f"http://localhost:{port}{endpoint}"

    print(f"Audio file: {audio_file}")
    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"Requests: {num_requests} | Concurrency: {max_concurrency} | Rate: {request_rate_raw} rps")
    print(f"Endpoint: {url}")
    print()

    # Run benchmark
    wall_start = time.perf_counter()
    results = asyncio.run(
        _run_async(
            url, audio_bytes, filename, content_type, model,
            num_requests, max_concurrency, request_rate, extra_data,
        )
    )
    total_time = time.perf_counter() - wall_start

    # Format and print results
    formatted = _format_results(
        results, audio_file, audio_duration, total_time,
        num_requests, max_concurrency,
    )
    print(formatted)
    sys.stdout.flush()

    # Save results
    results_config = results_config or {}
    save_result = results_config.get("save_result", False)
    result_dir = results_config.get("result_dir", "./stt_benchmark_results")

    if save_result:
        os.makedirs(result_dir, exist_ok=True)

        # Save JSON
        json_data = _build_json_result(
            result_name, model, audio_file, audio_duration,
            num_requests, max_concurrency, request_rate,
            total_time, results,
        )
        json_path = os.path.join(result_dir, f"{result_name}.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"\nResults saved: {json_path}")

        # Save TXT log
        txt_path = os.path.join(result_dir, f"{result_name}.txt")
        with open(txt_path, "w") as f:
            f.write(f"STT BENCHMARK: {result_name}\n")
            f.write("=" * 64 + "\n")
            f.write(formatted + "\n")
        print(f"Log saved: {txt_path}")

    return 0
