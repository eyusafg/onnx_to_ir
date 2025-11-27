import argparse
import os
import sys
import time

sys.path.insert(0, "")
sys.path.insert(0, "../../../")
sys.path.append("bisebetv2_cloth_segm")

import cv2
import numpy as np
import openvino as ov

from configs.segmentation import set_cfg_from_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        dest="config",
        type=str,
        default=r"bisebetv2_cloth_segm\configs\segmentation\bisenetv2_syt_segm_edge_thor_1203.py",
    )
    parser.add_argument(
        "--xml-path",
        dest="xml_path",
        type=str,
        default=r"thor_model_segm\openvino_ir\thor_segm_model_50.xml",
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        default="GPU",
    )
    parser.add_argument(
        "--img-dir",
        dest="img_dir",
        type=str,
        default=r"Dataset\thor\20251112",
    )
    parser.add_argument(
        "--save-dir",
        dest="save_dir",
        type=str,
        default="openvino_out",
    )
    parser.add_argument(
        "--model-password",
        dest="model_password",
        type=str,
        default="",
        help="Password for encrypted model (if model is encrypted)",
    )
    return parser.parse_args()


def preprocess_image(img_path, target_size):
    im_bgr = cv2.imread(img_path)
    if im_bgr is None:
        raise RuntimeError(f"failed to read image: {img_path}")

    origin_bgr = im_bgr.copy()
    im_bgr = cv2.resize(im_bgr, (target_size[0], target_size[1]))
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    im_rgb = im_rgb.astype(np.float32)
    im_chw = im_rgb.transpose(2, 0, 1)
    im_chw = np.expand_dims(im_chw, axis=0)
    return origin_bgr, im_chw


def main():
    args = parse_args()

    cfg = set_cfg_from_file(args.config)
    cfg_dict = dict(cfg.__dict__)
    target_size = (384, 384)
    if "target_size" in cfg_dict:
        target_size = cfg.target_size

    os.makedirs(args.save_dir, exist_ok=True)

    core = ov.Core()
    
    # 检查设备是否可用
    available_devices = core.available_devices
    print(f"Available devices: {available_devices}")
    
    # 编译模型
    print(f"Compiling model on device: {args.device}")
    
    # 如果模型加密，需要提供密码
    if args.model_password:
        print("Loading encrypted model with password...")
        compiled_model = core.compile_model(
            args.xml_path, 
            args.device,
            {"MODEL_ENCRYPTION_KEY": args.model_password}
        )
    else:
        compiled_model = core.compile_model(args.xml_path, args.device)
    
    output_port = compiled_model.output(0)
    
    # 获取实际使用的设备
    actual_device = compiled_model.get_property("EXECUTION_DEVICES")
    print(f"Model compiled on: {actual_device}")

    img_list = [
        f
        for f in os.listdir(args.img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    total_infer_time = 0.0
    for img_name in img_list:
        img_path = os.path.join(args.img_dir, img_name)
        print(f"infer image: {img_path}")

        origin_bgr, input_data = preprocess_image(img_path, target_size)

        # 推理时间统计
        infer_start = time.time()
        result = compiled_model([input_data])[output_port]
        infer_end = time.time()
        infer_time = infer_end - infer_start
        total_infer_time += infer_time

        if result.ndim == 4:
            if result.shape[1] > 1:
                pred = np.argmax(result, axis=1)
            else:
                pred = result[:, 0]
        elif result.ndim == 3:
            pred = result
        else:
            raise RuntimeError(f"unexpected output shape: {result.shape}")

        pred = pred[0]
        pred = pred.astype(np.uint8)
        mask = np.where(pred > 0, 255, 0).astype(np.uint8)

        mask_resized = cv2.resize(
            mask,
            (origin_bgr.shape[1], origin_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        mask_bgr = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
        dst = cv2.addWeighted(origin_bgr, 0.8, mask_bgr, 0.5, 0)

        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, img_name)
        cv2.imwrite(save_path, dst)
        print(f"saved result to: {save_path}")
        print(f"inference time: {infer_time:.4f}s\n")
    
    # 打印统计信息
    print("=" * 60)
    print(f"Total images processed: {len(img_list)}")
    print(f"Total inference time: {total_infer_time:.4f}s")
    if len(img_list) > 0:
        print(f"Average inference time per image: {total_infer_time / len(img_list):.4f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
