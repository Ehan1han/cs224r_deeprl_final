#!/bin/bash
set -e

# Function to check if miniconda is already installed
function check_miniconda {
    if [ -d "$HOME/miniconda3" ]; then
        echo "Miniconda is already installed."
        return 0
    else
        return 1
    fi
}

# Function to install miniconda
function install_miniconda {
    echo "Installing Miniconda..."
    
    # Download the miniconda installer
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    
    # Install miniconda
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    
    # Clean up
    rm /tmp/miniconda.sh
    
    # Add conda to path for current session
    export PATH="$HOME/miniconda3/bin:$PATH"
    
    # Initialize conda for bash shell
    $HOME/miniconda3/bin/conda init bash
    
    echo "Miniconda installed successfully."
}

# Check if miniconda is installed, if not install it
if ! check_miniconda; then
    install_miniconda
fi

# Source conda
source $HOME/miniconda3/etc/profile.d/conda.sh

# Create and activate conda environment
echo "Creating conda environment 'rl_llm'..."
conda create -y -n rl_llm python=3.10

echo "Activating conda environment 'rl_llm'..."
conda activate rl_llm

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install requirements from requirements.txt
echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

# Set up Weights & Biases
echo "Setting up Weights & Biases..."
wandb login 55dde296d8a0abdfc29716d67d95ed465b6e40d4

# Create necessary directories
mkdir -p outputs/sft
mkdir -p outputs/dpo
mkdir -p outputs/rloo
mkdir -p outputs/reward_model_full outputs/reward_model_100

echo "Environment setup complete!"
echo "To activate the environment, run: conda activate rl_llm" 