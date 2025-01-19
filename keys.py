import h5py
import numpy as np

opt_path = "C:/Users/Asus/Desktop/1138536582_opt.mat"
opt_ftp_path = "C:/Users/Asus/Desktop/1138536582_optftp.mat"
with h5py.File(opt_path, "r") as f:
    print("Keys in the opt file:", list(f.keys()))
with h5py.File(opt_ftp_path, "r") as f:
    print("Keys in the opt_ftp file:", list(f.keys()))
with h5py.File(opt_path, "r") as f:
    opt = np.array(f["x3"])
    print(f"opt shape: {opt.shape}")
    print(f"opt min: {opt.min()}, opt max: {opt.max()}")
with h5py.File(opt_ftp_path, "r") as f:
    opt_ftp = np.array(f["x4"])
    print(f"opt_ftp shape: {opt_ftp.shape}")
    print(f"opt min: {opt_ftp.min()}, opt max: {opt_ftp.max()}")
