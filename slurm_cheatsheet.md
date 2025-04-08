### Slurm Cheatsheet

#### Grundlegende Kommandos
- **`srun`**: Führt einen Job interaktiv aus.
  ```bash
  srun -p performance -t 02:00:00 --gpus=1 --job-name=MyTraining python train.py
  ```

- **`sbatch`**: Job im Batch-Modus ausführen (Hintergrundausführung).
  ```bash
  sbatch slurm_train.sh
  ```

- **`salloc`**: Ressourcen reservieren.
  ```bash
  salloc --gpus=1 --cpus=2 --mem=10G --time=01:00:00
  ```

- **`sinfo`**: Basisinformationen über den Cluster anzeigen.
  ```bash
  sinfo
  ```

- **`squeue`**: Warteschlange der Jobs anzeigen.
  ```bash
  squeue -u username
  ```

- **`scancel`**: Einen Job abbrechen.
  ```bash
  scancel JOBID
  ```

- **`scontrol`**: Erweiterte Informationen über Jobs und Nodes abrufen.
  ```bash
  scontrol show job JOBID
  ```

- **`sdiah`**: Diagnosedaten für Jobs anzeigen.
  ```bash
  sdiah JOBID
  ```

- **`sstat`**: Statusdetails eines Jobs oder Steps anzeigen.
  ```bash
  sstat JOBID
  ```

#### Batch-Skript Vorlage (`sbatch`)
```bash
#!/bin/bash -l
#SBATCH -p performance
#SBATCH -t 02:00:00
#SBATCH --gpus=1
#SBATCH --job-name=MyTraining

conda activate test-env
python train.py
```

#### Singularity mit Slurm
- Container-Job starten:
  ```bash
  srun singularity exec --nv myContainer.sif python train.py
  ```

- Container mit Overlay oder Mounts:
  ```bash
  singularity exec --nv --bind /host/path:/container/path myContainer.sif python train.py
  ```

#### Interaktiven Jupyter-Server starten
- Starten:
  ```bash
  /cluster/common/jupyter/start-jupyter.sh --sif=myContainer.sif --gpus=1 --time=2:0:0
  ```
- Portweiterleitung:
  ```bash
  ssh -N -L 8888:localhost:8888 user@clusterIP
  ```

#### Datenübertragung auf Cluster
- Dateien hochladen:
  ```bash
  scp -i .ssh/key -r lokaleDateien username@slurmlogin.cs.technik.fhnw.ch:~/remoteOrdner
  ```

#### Tipps & Hinweise
- Ressourcen immer angemessen reservieren
- Job frühzeitig submitten und in der Queue warten lassen
- Interaktive Sessions nur für Debugging oder kurze Jobs verwenden
- Kondensierte Job-Logs sind standardmäßig unter `slurm-JOBID.out` gespeichert
- Bei Problemen jederzeit an Ansprechpartner wenden

#### Ansprechpartner
- **Manuel Stutz**: manuel.stutz@fhnw.ch (bei Fragen zu Slurm und Clusterproblemen)

