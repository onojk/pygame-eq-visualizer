import subprocess
import sys

# List of required libraries
required_libraries = [
    "numpy",
    "pyaudio",
    "pygame",
    "scipy"
]

def install_libraries():
    for library in required_libraries:
        try:
            # Try importing the library to check if it's already installed
            __import__(library)
            print(f"{library} is already installed.")
        except ImportError:
            # If the library is not installed, attempt to install it
            print(f"{library} not found. Installing...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", library])
                print(f"Successfully installed {library}.")
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {library}. Please install it manually.")
                print(f"For manual installation, check the requirements.txt file.")

def check_requirements():
    print("Checking required libraries...")
    install_libraries()
    print("\nIf any issues occurred, refer to the 'requirements.txt' for manual installation instructions.")

if __name__ == "__main__":
    check_requirements()

