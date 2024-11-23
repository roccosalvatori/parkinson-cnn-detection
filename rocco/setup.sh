#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting setup process...${NC}"

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo -e "${RED}Homebrew is not installed. Installing Homebrew...${NC}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install Python 3.11
echo -e "${YELLOW}Installing Python 3.11...${NC}"
brew install python@3.11

# Remove existing virtual environment if it exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}Removing existing virtual environment...${NC}"
    rm -rf venv
fi

# Create new virtual environment
echo -e "${YELLOW}Creating new virtual environment...${NC}"
python3.11 -m venv venv

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install PyTorch and torchaudio with MPS support
echo -e "${YELLOW}Installing PyTorch and torchaudio...${NC}"
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# Create and install other requirements
echo -e "${YELLOW}Creating requirements.txt and installing other dependencies...${NC}"
cat > requirements.txt << EOL
numpy>=1.24.3
librosa>=0.10.1
scikit-learn>=1.3.0
matplotlib>=3.7.2
seaborn>=0.12.2
tabulate>=0.9.0
timm>=0.9.2
tqdm>=4.65.0
soundfile
EOL

# Install other requirements
pip install -r requirements.txt

# Create directories if they don't exist
echo -e "${YELLOW}Creating project directories...${NC}"
mkdir -p sick healthy

# Create a test script to verify PyTorch installation
cat > test_torch.py << EOL
import torch
import torchaudio
import soundfile as sf

print("PyTorch version:", torch.__version__)
print("TorchAudio version:", torchaudio.__version__)
print("MPS (Metal) available:", torch.backends.mps.is_available())
print("CUDA available:", torch.cuda.is_available())

# Test tensor creation
x = torch.randn(2, 3)
print("\nTest tensor:")
print(x)

if torch.backends.mps.is_available():
    x_mps = x.to('mps')
    print("\nMPS tensor:")
    print(x_mps)
EOL

echo -e "${YELLOW}Testing PyTorch installation...${NC}"
python test_torch.py

echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "${GREEN}Virtual environment is now activated.${NC}"
echo -e "${GREEN}Place your audio files in the following directories:${NC}"
echo -e "${YELLOW}- sick/ for Parkinson's patients' audio files${NC}"
echo -e "${YELLOW}- healthy/ for control group audio files${NC}"

# Create activation script
cat > activate_venv.sh << EOL
#!/bin/bash
source venv/bin/activate
EOL

chmod +x activate_venv.sh

echo -e "${GREEN}You veirtual environment has been activated with the command:${NC}"
echo -e "${YELLOW}source venv/bin/activate${NC}"

source venv/bin/activate

# Execute the script in the current shell
exec $SHELL 