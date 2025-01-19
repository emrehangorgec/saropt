import torch
import numpy as np
import h5py
import cv2
from model import MODEL_MM, MODEL_SAR, MODEL_OPT


def load_model(model_path, mode, device, sar_pretrain=None, opt_pretrain=None):
    """
    Loads the trained model and prepares it for inference.
    """
    if mode == "sar":
        model = MODEL_SAR(sar_pretrain)
    elif mode == "opt":
        model = MODEL_OPT(opt_pretrain)
    elif mode == "all":
        model = MODEL_MM(sar_pretrain, opt_pretrain)
    else:
        raise ValueError("Invalid mode. Choose from 'sar', 'opt', or 'all'.")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(opt_path, optftp_path, input_size=(256, 256), device="cpu"):
    """
    Preprocesses the optical and footprint images for inference.
    """
    # Load and preprocess optical image
    with h5py.File(opt_path, "r") as f_opt:
        opt = np.array(f_opt["x3"])
    if opt.shape[0] != 3:
        raise ValueError(
            f"Optical image must have 3 channels (RGB), but got {opt.shape[0]} channels."
        )
    opt = cv2.resize(opt.transpose(1, 2, 0), input_size).transpose(2, 0, 1)
    opt = opt.astype(np.float32) / 255.0

    # Load and preprocess optical footprint
    with h5py.File(optftp_path, "r") as f_optftp:
        optftp = np.array(f_optftp["x4"])
    if len(optftp.shape) == 2:  # If single-channel (grayscale)
        optftp = np.expand_dims(optftp, axis=0)  # Add channel dimension (1, H, W)
    optftp = cv2.resize(optftp[0], input_size)  # Resize single channel
    optftp = np.repeat(optftp[np.newaxis, :, :], 3, axis=0)  # (H, W) -> (3, H, W)
    optftp = optftp.astype(np.float32)  # Already normalized to [0, 1]

    # Convert to PyTorch tensors
    opt_tensor = torch.tensor(opt, dtype=torch.float32).unsqueeze(0).to(device)
    optftp_tensor = torch.tensor(optftp, dtype=torch.float32).unsqueeze(0).to(device)

    return opt_tensor, optftp_tensor


def predict(opt_tensor, optftp_tensor, model):
    """
    Performs a prediction using the loaded model.
    """
    with torch.no_grad():
        outputs = model(opt_tensor, optftp_tensor)
        probabilities = torch.sigmoid(outputs).squeeze().item()

    # Determine class based on threshold
    prediction = "Yıkılmış" if probabilities > 0.5 else "Sağlam"
    return prediction, probabilities


if __name__ == "__main__":
    import argparse

    # Argument parser
    parser = argparse.ArgumentParser(
        description="Test a trained model on a single image."
    )
    parser.add_argument(
        "--opt", required=True, type=str, help="Path to the optical image file."
    )
    parser.add_argument(
        "--optftp", required=True, type=str, help="Path to the optical footprint file."
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Path to the trained model (.pth file).",
    )
    parser.add_argument(
        "--mode",
        default="opt",
        type=str,
        choices=["sar", "opt", "all"],
        help="Model mode.",
    )
    parser.add_argument(
        "--input_format", default="mat", type=str, choices=["mat"], help="Input format."
    )
    parser.add_argument(
        "--sar_pretrain",
        default=None,
        type=str,
        help="Path to SAR pretrain weights (if applicable).",
    )
    parser.add_argument(
        "--opt_pretrain",
        default=None,
        type=str,
        help="Path to OPT pretrain weights (if applicable).",
    )
    args = parser.parse_args()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(
        args.model, args.mode, device, args.sar_pretrain, args.opt_pretrain
    )

    # Preprocess images
    opt_tensor, optftp_tensor = preprocess_image(
        args.opt, args.optftp, input_size=(256, 256), device=device
    )

    # Make prediction
    prediction, probabilities = predict(opt_tensor, optftp_tensor, model)

    print(f"Prediction: {prediction} (Probability: {probabilities:.4f})")
