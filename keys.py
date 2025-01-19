import h5py

file_path = "C:/Users/Asus/Desktop/1138536582_optftp.mat"

with h5py.File(file_path, "r") as f:
    print("Keys in the file:", list(f.keys()))
