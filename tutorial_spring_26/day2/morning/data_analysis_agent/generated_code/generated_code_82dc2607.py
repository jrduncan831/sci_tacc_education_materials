import sys
import subprocess
import os

def install_packages(required_packages):
    if not required_packages:
        return
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-t", os.environ["TMPDIR"]] + required_packages, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"Failed to install packages with uv: {e}")
        sys.exit(1)


def main():
    required_packages = ["scipy", "numpy", "matplotlib", "scikit-learn"]  # Required packages

    if required_packages:
        install_packages(required_packages)

    # Import after installing required packages
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.io import loadmat
    from sklearn.decomposition import PCA
    
    # Load data from VORTALL.mat file
    data_file = './data/VORTALL.mat'
    mat_data = loadmat(data_file)
    
    # Assuming the data variable is stored in the mat file under key 'VORTALL' (common name for such files)
    # If it has a different key, that should be adjusted
    X = mat_data['VORTALL']
    
    # Perform PCA
    pca = PCA()
    pca.fit(X)
    
    # Compute cumulative variance explained
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Plot first 10 modes
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, min(11, len(cumulative_variance) + 1)), cumulative_variance[:10], marker='o')
    plt.xlabel('Number of Modes')
    plt.ylabel('Cumulative Variance Explained')
    plt.title('Cumulative Variance for First 10 PCA Modes')
    plt.grid(True)
    
    # Ensure directory exists
    output_dir = './generated_images/pca'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save plot
    plot_path = os.path.join(output_dir, 'cumulative_var.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()