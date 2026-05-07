import json
import os
from datetime import datetime


def ensure_serializable(value):
    if isinstance(value, dict):
        return {str(k): ensure_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [ensure_serializable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def format_kv_lines(data, prefix=""):
    lines = []
    for key, value in data.items():
        label = f"{prefix}{key}"
        if isinstance(value, dict):
            lines.append(f"{label}:")
            lines.extend(format_kv_lines(value, prefix=prefix + "  "))
        else:
            lines.append(f"{label}: {value}")
    return lines


def write_summary_files(summary_dir, base_name, summary):
    os.makedirs(summary_dir, exist_ok=True)
    summary = ensure_serializable(summary)
    timestamp = datetime.now().isoformat(timespec="seconds")
    summary["saved_at"] = timestamp

    json_path = os.path.join(summary_dir, f"{base_name}.json")
    txt_path = os.path.join(summary_dir, f"{base_name}.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    title = summary.get("title", base_name)
    lines = [title, "=" * len(title)]
    for key, value in summary.items():
        if key == "title":
            continue
        if isinstance(value, dict):
            lines.append("")
            lines.append(f"{key}:")
            lines.extend(format_kv_lines(value, prefix="  "))
        else:
            lines.append(f"{key}: {value}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return json_path, txt_path
