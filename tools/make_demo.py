"""Build a narrated walkthrough from actual captured browser screenshots (macOS)."""
import json
from pathlib import Path
import subprocess


def main():
    root = Path("artifacts/demo").resolve()
    shots = json.loads((root / "storyboard.json").read_text())
    timeline, subtitles, elapsed = [], [], 0.0
    def stamp(seconds):
        ms = round(seconds*1000)
        return f"{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}"
    for i, shot in enumerate(shots):
        source = root / shot["image"]
        if not source.exists():
            raise FileNotFoundError(source)
        audio = root / f"shot-{i:02}.aiff"
        video = root / f"shot-{i:02}.mp4"
        subprocess.run(["say", "-v", "Tingting", "-r", "185", "-o", str(audio), shot["narration"]], check=True)
        duration = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(audio)], text=True))+1
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-loop", "1", "-i", str(source),
                        "-i", str(audio), "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1",
                        "-af", "apad=pad_dur=1", "-t", str(duration), "-r", "24", "-c:v", "libx264", "-preset", "fast",
                        "-crf", "22", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000", str(video)], check=True)
        timeline.append(f"file '{video.name}'")
        subtitles.append(f"{i+1}\n{stamp(elapsed)} --> {stamp(elapsed+duration)}\n{shot['narration']}\n")
        elapsed += duration
    (root / "concat.txt").write_text("\n".join(timeline)+"\n")
    (root / "captions.srt").write_text("\n".join(subtitles))
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(root / "concat.txt"), "-i", str(root / "captions.srt"), "-c", "copy", "-c:s", "mov_text",
                    "-disposition:s:0", "default", "-movflags", "+faststart", str(root / "walkthrough.mp4")], check=True)
    print(f"Created narrated walkthrough: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
