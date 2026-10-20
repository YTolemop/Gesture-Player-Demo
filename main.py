import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent


def positive(value):
    number = float(value)
    if not 0 < number < float("inf"):
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def parser():
    p = argparse.ArgumentParser(description="Gesture Player Demo — three-channel recognition")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="download and verify the landmark model")
    sub.add_parser("check", help="check dependencies and landmark model; no camera access")
    for name in ("camera", "record", "run", "recognize"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--camera", type=int, default=0)
        cmd.add_argument("--seconds", type=positive, help="otherwise stop with Ctrl+C")
        cmd.add_argument("--preview", action="store_true",
                         help="show camera and raw landmarks; Q/Esc stops")
        if name == "record":
            cmd.add_argument("--label", required=True, help="your own gesture class name")
            cmd.add_argument("--channel", choices=("left", "right", "both"), required=True)
            cmd.add_argument("--output", type=Path, default=ROOT / "data")
            cmd.add_argument("--purpose", choices=("train", "test"), default="train")
        if name == "recognize":
            cmd.add_argument("--model", type=Path, default=ROOT / "models/model.npz")
        if name == "run":
            cmd.add_argument("--server", required=True)
            cmd.add_argument("--room", default="")
            cmd.add_argument("--pair-code", required=True)
            cmd.add_argument("--model", type=Path, default=ROOT / "models/model.npz")
    train = sub.add_parser("train")
    train.add_argument("--data", type=Path, default=ROOT / "data")
    train.add_argument("--model", type=Path, default=ROOT / "models/model.npz")
    test = sub.add_parser("test", help="evaluate independent held-out recordings")
    test.add_argument("--data", type=Path, default=ROOT / "data")
    test.add_argument("--model", type=Path, default=ROOT / "models/model.npz")
    test.add_argument("--output", type=Path, default=ROOT / "models/evaluation-three-channel.json")
    return p


def capture(args):
    import cv2
    from core.camera import Camera
    from core.landmarks import Landmarks
    from network.commands import Commands

    pipeline = client = None
    if args.command in ("run", "recognize"):
        from student.pipeline import Pipeline
        from network.arena_client import ArenaClient
        pipeline = Pipeline(args.model)
        flags = pipeline.algorithms
        if not (flags.get("normalization") and flags.get("knn") and
                (not flags.get("filterEnabled") or flags.get("filter"))):
            raise RuntimeError("Student pipeline is unfinished; implement it and set algorithms availability")

    with ExitStack() as stack:
        detector = Landmarks()
        stack.callback(detector.close)
        camera = Camera(args.camera)
        stack.callback(camera.close)
        if args.command == "run":
            client = ArenaClient()
            stack.callback(client.close)
            result = client.connect(args.server, args.room, args.pair_code, pipeline.algorithms)
            print(f"Paired: room={result['roomCode']}; ready/start in arena", flush=True)
        if args.preview:
            from core.preview import show_preview
            stack.callback(cv2.destroyAllWindows)
        video = samples = None
        if args.command == "record":
            import uuid
            args.output.mkdir(parents=True, exist_ok=True)
            stem = args.output / f"capture-{time.time_ns()}-{uuid.uuid4().hex[:8]}"
            samples = stack.enter_context(stem.with_suffix(".jsonl").open("x", encoding="utf-8"))
            print(f"Recording: {stem}.mp4 + .jsonl", flush=True)
        start = last_frame = last_report = time.monotonic()
        previous_timestamp = None
        last_send = 0
        revision = -1
        count = 0
        print("Camera running. Ctrl+C stops and closes files.", flush=True)
        try:
            while args.seconds is None or time.monotonic() - start < args.seconds:
                frame, timestamp = camera.latest()
                now = time.monotonic()
                if frame is None or timestamp == previous_timestamp:
                    if now - last_frame > 2:
                        raise RuntimeError("Camera stopped delivering frames")
                    time.sleep(.01)
                    continue
                previous_timestamp = timestamp
                last_frame = now
                sample = detector.detect(frame, timestamp, camera.actual)
                count += 1
                if samples is not None:
                    if video is None:
                        video = cv2.VideoWriter(str(stem.with_suffix(".mp4")),
                                                cv2.VideoWriter_fourcc(*"mp4v"), 30,
                                                (frame.shape[1], frame.shape[0]))
                        stack.callback(video.release)
                        if not video.isOpened():
                            raise RuntimeError("Cannot open MP4 writer")
                    video.write(frame)
                    samples.write(json.dumps({**asdict(sample), "label": args.label,
                                              "channel": args.channel, "purpose": args.purpose}) + "\n")
                if pipeline:
                    current = client.revision if client else 0
                    if current != revision:
                        pipeline.reset()
                        revision = current
                    # Students own handedness, missing-hand behavior and filtering.
                    commands = pipeline.predict(sample)
                    if not isinstance(commands, Commands):
                        raise TypeError("Pipeline.predict must return Commands")
                    if client and now - last_send >= .1:
                        client.send(commands, revision)
                        last_send = now
                    if client:
                        client.send_preview(frame, sample, revision)
                if args.preview and not show_preview(frame, sample):
                    break
                if now - last_report >= 1:
                    status = f"frames={count} hands={len(sample.hands)}"
                    if pipeline:
                        status += f" commands={commands.payload()}"
                    print(status, flush=True)
                    last_report = now
        except KeyboardInterrupt:
            print("Stopped.")
        finally:
            print(f"Captured {count} frames.", flush=True)


def main():
    args = parser().parse_args()
    if args.command == "prepare":
        from core.resources import prepare
        print(json.dumps(prepare()))
    elif args.command == "check":
        import cv2, numpy, mediapipe, socketio
        from core.landmarks import Landmarks
        detector = Landmarks()
        detector.close()
        print("Dependencies and landmark model OK. Camera and student pipeline not tested.")
    elif args.command == "test":
        from student.pipeline import evaluate
        evaluate(args.data, args.model, args.output)
    elif args.command == "train":
        from student.pipeline import train
        train(args.data, args.model)
    else:
        capture(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
