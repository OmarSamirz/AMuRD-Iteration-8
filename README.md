# Prodify


conda create -n iteration8 python=3.12

## Setup for Development

### Conda Environment
Create a new Conda environment:
```bash
conda create -n iteration8 python=3.12
```

Install pip and project dependencies:
```bash
pip install -r requirements.txt
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

### GPU Configuration
1. **Download and install the Nvidia driver** appropriate for your GPU

2. **Install the CUDA toolkit**:
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Follow the installation instructions

3. **Install CUDA deep learning package (cuDNN)**:
   - Download from: https://developer.nvidia.com/cudnn-downloads
   - Extract and follow installation instructions

4. **Set up PyTorch with CUDA support**:
   ```bash
   # In your Conda environment
   pip uninstall torch torchvision torchaudio -y
   pip install torch --index-url https://download.pytorch.org/whl/cu126
   ```

5. **Verify CUDA installation**:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA device count: {torch.cuda.device_count()}")
   print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
   ```

## Database Setup

1. **Create a Teradata account** on the Clearscape Analytics platform: https://clearscape.teradata.com/

2. **Configure database credentials** in `.env`:
   ```env
   TD_HOST=your-teradata-host.com
   TD_NAME=your-database-name
   TD_USER=your-username
   TD_PASSWORD=your-password
   TD_PORT=1025
   TD_AUTH_TOKEN=your-auth-token
   ```

## How to Run

### Run the app
```python
streamlit run ./src/app.py
```

### Run pipeline
```python
conda run --live-stream --name iteration8 python ./src/main.py
```

This runs the complete classification pipeline:
1. **Data Insertion**: Load products and classes into Teradata
2. **Text Cleaning**: Clean and normalize product/class names
3. **Translation**: Translate Arabic products to English
4. **Embedding Generation**: Create vector embeddings for products and classes
5. **Classification**: Perform similarity-based classification using Teradata vector functions
6. **Evaluation**: Calculate F1-score and display results

