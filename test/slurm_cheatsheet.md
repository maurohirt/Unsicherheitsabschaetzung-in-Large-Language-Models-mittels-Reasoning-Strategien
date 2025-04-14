## Slurm Cheatsheet (Erweitert)

### Grundlegende Kommandos
```bash
# Interaktiven Job starten (z. B. zum Debugging)
srun -p performance -t 02:00:00 --gpus=1 --job-name=MyTraining python train.py

# Batch-Job starten
sbatch slurm_train.sh

# Ressourcen reservieren
salloc --gpus=1 --cpus=2 --mem=10G --time=01:00:00

# Cluster-Infos anzeigen
sinfo

# Jobs in der Queue anzeigen
squeue -u <username>

# Job abbrechen
scancel <JOBID>

# Erweiterte Infos
scontrol show job <JOBID>

# Diagnosedaten
sdiag <JOBID>

# Jobstatus
sstat <JOBID>
```

### Slurm Begriffe
- **Node**: Physischer Rechner für Jobs
- **Partition**: Slurm-Queue mit bestimmten Limits
- **Resource**: CPU, RAM, GPU, Zeit etc.
- **Allocation**: Reservierte Ressourcen
- **Job**: Das auszuführende Programm/Skript
- **Reservation**: Admin-seitige Reservierung (z. B. Wartung)

### Beispiel Batch-Skript
```bash
#!/bin/bash -l
#SBATCH -p performance
#SBATCH -t 02:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --job-name=MyTraining
#SBATCH --output=out.log
#SBATCH --error=err.log

python train.py
```

### Singularity mit Slurm
```bash
# Docker-Image in SIF umwandeln
singularity pull myTF.sif docker://tensorflow/tensorflow:latest-gpu

# Ausfuehren mit GPU, ohne HOME-Mount\ssingularity exec --nv --no-home --bind ./data:/app myTF.sif python /app/train.py
```

### Interaktives JupyterLab starten
```bash
# Start-Skript aufrufen
/cluster/common/jupyter/start-jupyter.sh --sif=myContainer.sif --gpus=1 --time=2:0:0

# Port-Forwarding (lokal ausführen)
ssh -N -L 8888:localhost:8888 <user>@slurmlogin.cs.technik.fhnw.ch
```

### Job Arrays
```bash
# 20 Jobs mit unterschiedlichen Parametern starten
sbatch --array=0-19 myScript.sh

# Nur 5 gleichzeitig ausführen
sbatch --array=0-19%5 myScript.sh

# Output-Dateien differenzieren
#SBATCH --output=output_%A_%a.log
```

### Job-Dependencies
```bash
# Warte bis anderer Job erfolgreich war
sbatch --dependency=afterok:<JOBID> myScript.sh
```

### Python mit simple-slurm
```python
from simple_slurm import Slurm
slurm = Slurm(gpus=1, partition='performance', time='1-0:0:0', job_name='Job', cpus_per_task='16')
slurm.sbatch("python3 train.py")
```

### Datenübertragung
```bash
# Lokale Dateien hochladen
scp -i ~/.ssh/key -r ./data <user>@slurmlogin.cs.technik.fhnw.ch:~/remoteDir
```

### Tipps & Tricks
- Kein Heavy Lifting auf dem Login-Node (z. B. keine Singularity-Builds)
- Ressourcen mit --exclude einschränken statt --nodelist nutzen
- Für Requeue: `scontrol requeue <jobid>`
- --chdir statt `cd` im Script verwenden

### Ressourcen
- **70 TB**: /mnt/nas05/data01 (Primärer Projektdatenspeicher)
- **77 TB**: /mnt/nas05/data02 (Sekundärer Projektdatenspeicher)
- **40 TB**: /mnt/nas05/astrodata01 (Astrodaten)
- **38 TB**: /mnt/nas05/clusterdata01 (Cluster-spezifisch)
- **Home**: /mnt/nas05/clusterdata01/home2/<user>
- **Gruppenverzeichnis**: /mnt/nas05/clusterdata01/group

### Ansprechpartner
- **Manuel Stutz**: Slurm & Cluster (manuel.stutz@fhnw.ch)
- **Jackie Schindler**: Infrastruktur-Fragen (via Teams)
- Teams-Channel: **W-HSI_HPC_Infrastructure_M365 → User Exchange**