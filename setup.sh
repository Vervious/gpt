#!/bin/bash

# Run using ./setup.sh, not using sh

# Define the virtual environment directory
VENV_DIR="../.venv"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install Python3 first."
    exit 1
fi

# Check if the virtual environment already exists
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists. Activating it..."
else
    # Create the virtual environment if it doesn't exist
    echo "Creating a Python virtual environment..."
    python3 -m venv $VENV_DIR
fi

# Activate the virtual environment
echo "Activating the virtual environment..."
source $VENV_DIR/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing packages from requirements.txt..."
    pip install -r requirements.txt
else
    echo "No requirements.txt found. Installing predefined packages..."
    # Define a list of packages to install
    packages=(
        numpy
    )
    
    for package in "${packages[@]}"; do
        echo "Installing $package..."
        pip install "$package"
    done
fi

mkdir log

echo "Environment setup complete!"

# Optional: Deactivate the virtual environment
echo "You can deactivate the virtual environment by running 'deactivate'."

echo "Setting up git parameters"

git config --global user.email "ben@vervious.com"
git config --global user.name "Benjamin Chan"

cd gpt

echo "Setup complete!"
