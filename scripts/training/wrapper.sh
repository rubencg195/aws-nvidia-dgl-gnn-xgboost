#!/bin/bash
set -e
set -x

echo "=== Starting wrapper script ==="

# Verify GPU is accessible
echo "=== Checking GPU availability ==="
nvidia-smi || echo "WARNING: nvidia-smi failed"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); print(f'CUDA device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Fix pyg-lib compatibility
echo "=== Fixing pyg-lib compatibility ==="
pip uninstall -y pyg-lib
pip install --no-cache-dir pyg-lib -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

# Verify installation
echo "=== Verifying pyg-lib installation ==="
python -c "import pyg_lib; print(f'pyg-lib version: {pyg_lib.__version__}')"

# Set up environment
echo "=== Setting up environment ==="
export PYTHONPATH=/opt/nim:${PYTHONPATH}

# Warm up CUDA and verify GPU operations work
echo "=== Warming up CUDA ==="
python << 'WARMUP'
import torch
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    x = torch.randn(100, 100, device=device)
    y = torch.randn(100, 100, device=device)
    z = torch.matmul(x, y)
    print(f"CUDA warmup successful. Device: {device}, Result shape: {z.shape}")
    torch.cuda.synchronize()
else:
    print("WARNING: CUDA not available!")
WARMUP

echo "=== Inspecting /opt/ml recursively ==="
ls -R /opt/ml/

# Create training launcher
echo "=== Creating training launcher ==="
cat > /tmp/launch_training.py << 'EOF'
import json
import logging
import sys
import torch
import os
import shutil
import subprocess
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logging.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logging.info(f"CUDA version: {torch.version.cuda}")
    torch.cuda.init()
    logging.info("CUDA context initialized")

sys.path.insert(0, '/opt/nim/lib/financial_fraud_training')

config_path = '/opt/ml/input/data/config/config.json'
logging.info(f"Loading config from: {config_path}")

with open(config_path, 'r') as f:
    config_dict = json.load(f)

logging.info("Config loaded successfully")
training_start_time = datetime.now()

try:
    from src.validate_and_launch import validate_config_and_run_training
    
    logging.info("=" * 60)
    logging.info("STARTING TRAINING")
    logging.info("=" * 60)
    
    validate_config_and_run_training(config_dict)
    
    training_end_time = datetime.now()
    training_duration = training_end_time - training_start_time
    training_success = True
    
    logging.info("=" * 60)
    logging.info("TRAINING COMPLETED SUCCESSFULLY")
    logging.info("=" * 60)
    
except Exception as e:
    training_end_time = datetime.now()
    training_duration = training_end_time - training_start_time
    training_success = False
    
    logging.error("=" * 60)
    logging.error("TRAINING FAILED")
    logging.error("=" * 60)
    logging.error(f"Error: {e}", exc_info=True)
    
finally:
    logging.info("=" * 60)
    logging.info("CREATING TRAINING SNAPSHOT")
    logging.info("=" * 60)
    
    snapshot_dir = '/opt/ml/model/training_snapshot'
    os.makedirs(snapshot_dir, exist_ok=True)
    
    logging.info("Saving configuration...")
    shutil.copy(config_path, os.path.join(snapshot_dir, 'config.json'))
    
    logging.info("Saving input data snapshot...")
    input_data_dir = os.path.join(snapshot_dir, 'input_data')
    os.makedirs(input_data_dir, exist_ok=True)
    
    sagemaker_data_dir = '/opt/ml/input/data'
    if os.path.exists(sagemaker_data_dir):
        for item in os.listdir(sagemaker_data_dir):
            if item.endswith('-manifest'):
                continue
            
            src_path = os.path.join(sagemaker_data_dir, item)
            dest_path = os.path.join(input_data_dir, item)
            
            try:
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
                    file_count = sum([len(files) for r, d, files in os.walk(dest_path)])
                    dir_size = sum([os.path.getsize(os.path.join(r, f)) for r, d, files in os.walk(dest_path) for f in files])
                    logging.info(f"Copied input channel '{item}': {file_count} files, {dir_size / 1e6:.2f} MB")
                elif os.path.isfile(src_path):
                    shutil.copy2(src_path, dest_path)
                    file_size = os.path.getsize(dest_path)
                    logging.info(f"Copied input file '{item}': {file_size / 1e6:.2f} MB")
            except Exception as e:
                logging.warning(f"Could not copy input data '{item}': {e}")
    
    logging.info("Saving training metadata...")
    metadata = {
        'training_start': training_start_time.isoformat(),
        'training_end': training_end_time.isoformat(),
        'training_duration_seconds': training_duration.total_seconds(),
        'training_success': training_success,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        metadata['cuda_version'] = torch.version.cuda
        metadata['gpu_name'] = torch.cuda.get_device_name(0)
        metadata['gpu_memory_total_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        metadata['gpu_memory_allocated_gb'] = torch.cuda.memory_allocated(0) / 1e9
        metadata['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved(0) / 1e9
    
    with open(os.path.join(snapshot_dir, 'training_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info("Saving environment info...")
    result = subprocess.run(['pip', 'list', '--format=freeze'], capture_output=True, text=True)
    with open(os.path.join(snapshot_dir, 'requirements.txt'), 'w') as f:
        f.write(result.stdout)
    
    logging.info("Saving system info...")
    with open(os.path.join(snapshot_dir, 'system_info.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Training Status: {'SUCCESS' if training_success else 'FAILED'}\n")
        f.write(f"Training Duration: {training_duration}\n")
        f.write(f"Start Time: {training_start_time}\n")
        f.write(f"End Time: {training_end_time}\n")
        f.write(f"\n{'=' * 40}\n")
        f.write(f"PyTorch version: {torch.__version__}\n")
        f.write(f"CUDA available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"CUDA version: {torch.version.cuda}\n")
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
            f.write(f"GPU memory total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
            f.write(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB\n")
            f.write(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB\n")
    
    logging.info("Saving source code...")
    code_snapshot_dir = os.path.join(snapshot_dir, 'code')
    os.makedirs(code_snapshot_dir, exist_ok=True)
    
    source_dirs = [
        '/opt/nim/lib/financial_fraud_training/src',
        '/opt/nim/lib/financial_fraud_training'
    ]
    
    for src_dir in source_dirs:
        if os.path.exists(src_dir):
            dest_name = os.path.basename(src_dir)
            dest_dir = os.path.join(code_snapshot_dir, dest_name)
            try:
                shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)
                logging.info(f"Copied {src_dir} to snapshot")
            except Exception as e:
                logging.warning(f"Could not copy {src_dir}: {e}")
    
    logging.info("Cataloging model artifacts...")
    model_files = []
    if os.path.exists('/opt/ml/model'):
        for root, dirs, files in os.walk('/opt/ml/model'):
            if 'training_snapshot' in root:
                continue
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                relative_path = os.path.relpath(file_path, '/opt/ml/model')
                model_files.append({
                    'path': relative_path,
                    'size_bytes': file_size,
                    'size_mb': file_size / 1e6
                })
    
    with open(os.path.join(snapshot_dir, 'model_artifacts.json'), 'w') as f:
        json.dump({'artifacts': model_files, 'total_files': len(model_files)}, f, indent=2)
    
    logging.info("Cataloging input data...")
    input_files = []
    if os.path.exists(input_data_dir):
        for root, dirs, files in os.walk(input_data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                relative_path = os.path.relpath(file_path, input_data_dir)
                input_files.append({
                    'path': relative_path,
                    'size_bytes': file_size,
                    'size_mb': file_size / 1e6
                })
    
    with open(os.path.join(snapshot_dir, 'input_data_catalog.json'), 'w') as f:
        json.dump({
            'input_files': input_files, 
            'total_files': len(input_files),
            'total_size_mb': sum(f['size_mb'] for f in input_files)
        }, f, indent=2)
    
    logging.info(f"Training snapshot saved to {snapshot_dir}")
    logging.info(f"Total model artifacts: {len(model_files)}")
    logging.info(f"Total input files: {len(input_files)}")
    
    with open(os.path.join(snapshot_dir, 'SUMMARY.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("TRAINING SNAPSHOT SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Status: {'[SUCCESS]' if training_success else '[FAILED]'}\n")
        f.write(f"Duration: {training_duration}\n")
        f.write(f"Start: {training_start_time}\n")
        f.write(f"End: {training_end_time}\n\n")
        f.write(f"Files in snapshot:\n")
        f.write(f"  - config.json (training configuration)\n")
        f.write(f"  - training_metadata.json (detailed metadata)\n")
        f.write(f"  - system_info.txt (system and GPU info)\n")
        f.write(f"  - requirements.txt (Python packages)\n")
        f.write(f"  - model_artifacts.json (list of model files)\n")
        f.write(f"  - input_data_catalog.json (list of input files)\n")
        f.write(f"  - code/ (source code snapshot)\n")
        f.write(f"  - input_data/ (all input data channels)\n")
        f.write(f"\nTotal model artifacts: {len(model_files)}\n")
        f.write(f"Total input files: {len(input_files)}\n")
        if input_files:
            f.write(f"Total input data size: {sum(f['size_mb'] for f in input_files):.2f} MB\n")
    
    logging.info("Snapshot creation completed")
    
    if not training_success:
        raise

EOF

echo "=== Inspecting config file ==="
cat /opt/ml/input/data/config/config.json

echo "=== Starting training ==="
torchrun --standalone --nproc_per_node=1 /tmp/launch_training.py

echo "=== Final model directory contents ==="
ls -lah /opt/ml/model/
echo ""
echo "=== Snapshot contents ==="
ls -lah /opt/ml/model/training_snapshot/
echo ""
echo "=== Snapshot summary ==="
cat /opt/ml/model/training_snapshot/SUMMARY.txt 2>/dev/null || echo "No summary file found"

echo "=== Wrapper script completed ==="
