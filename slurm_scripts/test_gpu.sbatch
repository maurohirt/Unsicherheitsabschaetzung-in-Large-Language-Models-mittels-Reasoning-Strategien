#!/bin/bash

#====== SLURM Optionen ======
#SBATCH --job-name=gpu_test
#SBATCH --output=slurm-gpu-%j.out
#SBATCH --error=slurm-gpu-%j.err
#SBATCH --partition=performance   # Anpassen an deine GPU-Partition
#SBATCH --gpus=1                  # 1 GPU anfordern
#SBATCH --cpus-per-task=2         # 2 CPUs sollten reichen
#SBATCH --mem=4G                  # 4GB RAM
#SBATCH --time=00:10:00           # 10 Minuten sollten genügen

echo "GPU-Test Job gestartet am: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"

# GPU-Info anzeigen
echo "GPU Information:"
nvidia-smi

# Pfade definieren
HOST_PROJECT_DIR=$(pwd)
CONTAINER_PATH="./containers/cot_uq_latest.sif"  # Passe an, wenn anderer Name verwendet wurde
CONTAINER_PROJECT_DIR="/app/CoT-UQ"

# Prüfen, ob Container existiert
if [ ! -f "${CONTAINER_PATH}" ]; then
    echo "ERROR: Container nicht gefunden: ${CONTAINER_PATH}"
    echo "Bitte zuerst pull_test_container.sbatch ausführen"
    exit 1
fi

# GPU-Test-Skript erstellen
cat > gpu_test.py << 'EOL'
import torch
import time

def test_gpu():
    print("\n" + "=" * 50)
    print("PyTorch GPU Test")
    print("=" * 50)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU only.")
        return
    
    # Print GPU info
    device_count = torch.cuda.device_count()
    print(f"Found {device_count} CUDA device(s)")
    
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        device_cap = torch.cuda.get_device_capability(i)
        device_props = torch.cuda.get_device_properties(i)
        
        print(f"\nDevice {i}: {device_name}")
        print(f"  Compute Capability: {device_cap[0]}.{device_cap[1]}")
        print(f"  Total Memory: {device_props.total_memory / (1024**3):.2f} GB")
    
    # Simple GPU computation test
    print("\nRunning simple matrix multiplication test...")
    
    # Create random matrices
    size = 5000
    a = torch.randn(size, size, device="cuda")
    b = torch.randn(size, size, device="cuda")
    
    # Warmup
    torch.matmul(a, b)
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"Matrix multiplication of size {size}x{size} completed in {end_time - start_time:.4f} seconds")
    print("Test result shape:", c.shape)
    print("GPU memory allocated:", torch.cuda.memory_allocated() / (1024**3), "GB")
    print("\nGPU test completed successfully!")

if __name__ == "__main__":
    test_gpu()
EOL

# GPU-Test mit Container ausführen
echo "Starte Singularity Container mit GPU-Unterstützung..."
singularity exec --nv \
    --bind "${HOST_PROJECT_DIR}":"${CONTAINER_PROJECT_DIR}" \
    "${CONTAINER_PATH}" \
    python "${CONTAINER_PROJECT_DIR}/gpu_test.py"

EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "GPU-Test erfolgreich abgeschlossen."
else
    echo "GPU-Test fehlgeschlagen mit Exit-Code ${EXIT_CODE}"
fi

# Cleanup
rm -f gpu_test.py

echo "Job beendet am: $(date)"
exit ${EXIT_CODE}