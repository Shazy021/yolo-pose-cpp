#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import shutil

from ultralytics import YOLO


def parse_args():
    """Parse command-line arguments for YOLO pose model export."""
    parser = argparse.ArgumentParser(
        description="Export Ultralytics YOLO pose model (.pt) to ONNX"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to YOLO pose .pt model or model name (e.g. yolov8n-pose.pt, yolo11n-pose.pt)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=[640],  # H W
        help="Image size (h w) for export, default: 640, 640",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save ONNX model (default: models)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Export with dynamic shapes (batch/imgsz)",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Run ONNX simplifier after export",
    )
    return parser.parse_args()


def main():
    """Export YOLO pose model to ONNX format."""
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model (any Ultralytics YOLO pose model)
    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)

    # Determine output filename
    model_name = Path(args.model).stem  # e.g. yolov8n-pose.pt → yolov8n-pose
    onnx_path = Path(args.output_dir) / f"{model_name}.onnx"

    # Handle imgsz: Ultralytics accepts int or [h, w] list
    if len(args.imgsz) == 1:
        imgsz = args.imgsz[0]
    else:
        imgsz = args.imgsz  # [h, w]

    print(f"[INFO] Exporting to ONNX: {onnx_path}")
    print(f"[INFO] imgsz={imgsz}, dynamic={args.dynamic}, simplify={args.simplify}, opset={args.opset}")

    # Export to ONNX format
    model.export(
        format="onnx",
        imgsz=imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=args.simplify,
        device="cpu",  # Always CPU (Docker runs on CPU anyway)
    )

    # Ultralytics saves .onnx next to .pt by default → move to output dir.
    default_onnx = Path(f"{model_name}.onnx")
    if default_onnx.exists() and default_onnx.resolve() != onnx_path.resolve():
        shutil.move(str(default_onnx), str(onnx_path))

    print(f"[INFO] ONNX model saved to: {onnx_path}")


if __name__ == "__main__":
    main()