import argparse
import os

try:
    from openvino.tools import mo
except ImportError:
    from openvino import convert_model as mo_convert_model
    mo = None

import openvino as ov
from openvino.runtime import serialize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx-path",
        type=str,
        default=r"thor_model_segm\thor_segm_20251031.onnx",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"thor\openvino_ir",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="thor_segm_model_50",
    )
    parser.add_argument(
        "--encrypt",
        action="store_true",
        help="Enable model encryption with password",
    )
    parser.add_argument(
        "--password",
        type=str,
        default="",
        help="Password for model encryption (if --encrypt is set)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    onnx_path = os.path.abspath(args.onnx_path)

    print(f"Converting ONNX model: {onnx_path}")
    
    if mo is not None:
        ov_model = mo.convert_model(onnx_path)
    else:
        ov_model = mo_convert_model(onnx_path)

    xml_path = os.path.join(args.output_dir, args.model_name + ".xml")
    bin_path = os.path.join(args.output_dir, args.model_name + ".bin")
    
    # 加密模型
    if args.encrypt:
        if not args.password:
            print("Error: --password is required when --encrypt is set")
            return
        
        print(f"Encrypting model with password...")
        serialize(ov_model, xml_path, bin_path, save_weights_encrypted=True, model_password=args.password)
        print("OpenVINO IR saved (encrypted) to:")
    else:
        serialize(ov_model, xml_path, bin_path)
        print("OpenVINO IR saved (unencrypted) to:")
    
    print(f"  XML: {xml_path}")
    print(f"  BIN: {bin_path}")


if __name__ == "__main__":
    main()
