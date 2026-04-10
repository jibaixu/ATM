import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_INPUT_PATH = Path(
    "/data_jbx/Datasets/Realbot/4_4_four_tasks_wan/meta/episodes_train_test.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split each JSONL record with list-valued video/track fields into "
            "multiple records with string-valued video/track fields."
        )
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Source JSONL file containing list-valued video and track fields.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        type=Path,
        default=None,
        help=(
            "Output JSONL path. Defaults to the source file name with "
            "'_single_video' appended before the suffix."
        ),
    )
    return parser.parse_args()


def resolve_output_path(input_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    return input_path.with_name(f"{input_path.stem}_single_video{input_path.suffix}")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSON on line {line_number}: {exc}") from exc
    return entries


def write_jsonl(path: Path, entries: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in entries:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")


def expand_record(entry: Dict[str, Any], line_number: int) -> List[Dict[str, Any]]:
    video_field = entry.get("video")
    track_field = entry.get("track")

    if not isinstance(video_field, list):
        raise TypeError(
            f"Line {line_number}: expected 'video' to be a list, got {type(video_field).__name__}"
        )
    if not isinstance(track_field, list):
        raise TypeError(
            f"Line {line_number}: expected 'track' to be a list, got {type(track_field).__name__}"
        )
    if len(video_field) != len(track_field):
        raise ValueError(
            f"Line {line_number}: 'video' and 'track' lengths do not match "
            f"({len(video_field)} != {len(track_field)})"
        )

    expanded_entries: List[Dict[str, Any]] = []
    for pair_index, (video_path, track_path) in enumerate(zip(video_field, track_field)):
        if not isinstance(video_path, str):
            raise TypeError(
                f"Line {line_number}, pair {pair_index}: expected video item to be str, "
                f"got {type(video_path).__name__}"
            )
        if not isinstance(track_path, str):
            raise TypeError(
                f"Line {line_number}, pair {pair_index}: expected track item to be str, "
                f"got {type(track_path).__name__}"
            )

        single_entry = dict(entry)
        single_entry["video"] = video_path
        single_entry["track"] = track_path
        expanded_entries.append(single_entry)

    return expanded_entries


def build_single_video_records(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    single_video_entries: List[Dict[str, Any]] = []
    for line_number, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise TypeError(
                f"Line {line_number}: expected each JSONL record to be an object, "
                f"got {type(entry).__name__}"
            )
        single_video_entries.extend(expand_record(entry, line_number))
    return single_video_entries


def main() -> None:
    args = parse_args()
    input_path = args.input_path
    output_path = resolve_output_path(input_path, args.output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL file does not exist: {input_path}")

    entries = read_jsonl(input_path)
    single_video_entries = build_single_video_records(entries)
    write_jsonl(output_path, single_video_entries)

    print(f"Source records: {len(entries)}")
    print(f"Generated records: {len(single_video_entries)}")
    print(f"Output path: {output_path}")


if __name__ == "__main__":
    main()
