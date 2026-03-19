import sys
import subprocess
import os

def install_packages(required_packages):
    if not required_packages:
        return
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"Failed to install packages with uv: {e}")
        sys.exit(1)


def main():
    required_packages = ["scipy"]

    if required_packages:
        install_packages(required_packages)

    import scipy.io
    import os

    file_path = "./data/VORTALL.mat"

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return

    # Get file size in GB
    file_size_bytes = os.path.getsize(file_path)
    file_size_gb = file_size_bytes / (1024 ** 3)
    print(f"File size: {file_size_gb:.6f} GB")

    try:
        # Load the .mat file
        data = scipy.io.loadmat(file_path)
        
        # Print shapes of all variables in the mat file
        for key in data.keys():
            if not key.startswith('__'):
                var = data[key]
                print(f"Shape of variable '{key}': {var.shape}")
    except Exception as e:
        print(f"Error loading .mat file: {e}")


if __name__ == "__main__":
    main()