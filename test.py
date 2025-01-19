import torch
import numpy as np
import h5py
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


def preprocess_image(
    opt_path, optftp_path, input_format, device, opt_key="x3", optftp_key="x4"
):
    if input_format == "mat":
        # Read .mat files using h5py
        with h5py.File(opt_path, "r") as f_opt:
            opt = np.array(f_opt[opt_key])  # (95, 3, 130) gibi
        with h5py.File(optftp_path, "r") as f_optftp:
            optftp = np.array(f_optftp[optftp_key])  # (130, 95) gibi
    else:
        raise ValueError("Only .mat format is supported for this test script.")

    # Normalize the inputs
    opt = opt.astype(np.float32) / 255.0
    optftp = optftp.astype(np.float32) / 255.0

    # Adjust opt dimensions (C, H, W)
    if opt.ndim == 3 and opt.shape[0] != 3:  # (95, 3, 130) gibi yanlış boyut
        opt = np.transpose(opt, (1, 2, 0))  # (95, 3, 130) -> (3, 130, 95)

    # Adjust optftp dimensions to (C, H, W)
    if optftp.ndim == 2:  # Gri tonlamalı (H, W)
        optftp = np.expand_dims(optftp, axis=0)  # (H, W) -> (1, H, W)
        optftp = np.repeat(optftp, 3, axis=0)  # (1, H, W) -> (3, H, W)
    elif optftp.ndim == 3 and optftp.shape[0] == 1:  # (1, H, W)
        optftp = np.repeat(optftp, 3, axis=0)  # (1, H, W) -> (3, H, W)

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
        print("Model raw outputs:", outputs)
        print(
            "Optical tensor stats:",
            opt_tensor.min(),
            opt_tensor.max(),
            opt_tensor.mean(),
        )
        print(
            "Fine-tuned tensor stats:",
            optftp_tensor.min(),
            optftp_tensor.max(),
            optftp_tensor.mean(),
        )

    prediction = "Yıkılmış" if probabilities > 0.5 else "Sağlam"
    return prediction, probabilities


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test a trained model on optical images."
    )
    parser.add_argument(
        "--opt", required=True, type=str, help="Path to the optical (opt) image file."
    )
    parser.add_argument(
        "--optftp",
        required=True,
        type=str,
        help="Path to the optical fine-tuned (optftp) image file.",
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
        "--input_format",
        default="mat",
        type=str,
        choices=["mat"],
        help="Input format (mat only).",
    )
    args = parser.parse_args()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(args.model, args.mode, device)

    # Preprocess images
    opt_tensor, optftp_tensor = preprocess_image(
        args.opt, args.optftp, args.input_format, device
    )

    # Make prediction
    prediction, probabilities = predict(opt_tensor, optftp_tensor, model)

    print(f"Prediction: {prediction} (Probability: {probabilities:.4f})")
