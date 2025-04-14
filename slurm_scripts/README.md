# SLURM Workflow für CoT-UQ

Diese Anleitung erklärt, wie du das CoT-UQ Projekt auf dem SLURM-Cluster ausführst.

## Vorbereitung

1. **Repository klonen**
   ```bash
   git clone https://github.com/yourusername/CoT-UQ.git
   cd CoT-UQ
   ```

2. **Verzeichnisstruktur vorbereiten**
   ```bash
   mkdir -p containers slurm-logs
   ```

## Container bauen

1. **Container bauen**
   ```bash
   sbatch slurm_scripts/build_container.sbatch
   ```

   Dieser Job baut einen Singularity-Container basierend auf PyTorch und installiert alle benötigten Abhängigkeiten.

2. **Status prüfen**
   ```bash
   squeue -u $USER
   ```

   Warte bis der Job beendet ist. Die Ausgabe findest du in `slurm-build-JOBID.out`.

## Test-Job ausführen

1. **Basic Test** 
   ```bash
   sbatch slurm_scripts/simple_cpu_test.sbatch
   ```

   Dieser Job testet, ob der Container korrekt funktioniert und Python ausgeführt werden kann.

## Experiment ausführen

1. **Hauptexperiment starten**
   ```bash
   sbatch slurm_scripts/run_experiment.sbatch
   ```

   Dieser Job führt den LLaMA Pipeline-Prozess im Container mit GPU-Unterstützung aus.

## Anmerkungen

- Die Jobs werden vom aktuellen Verzeichnis aus ausgeführt. Alle Pfade sind relativ dazu.
- Der Container wird im Verzeichnis `./containers/` gespeichert
- Output-Dateien werden als `slurm-NAME-JOBID.out/err` gespeichert
- Die Experimente im Container verwenden die GPU mit dem `--nv` Flag

## Tipps bei Problemen

- Überprüfe die Fehlerprotokolle in den `.err`-Dateien
- Stelle sicher, dass die GPU-Partition verfügbar ist (`sinfo`)
- Bei Container-Problemen kann ein direkter Docker-Pull hilfreich sein:
  ```bash
  singularity pull containers/cot_uq_env.sif docker://pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
  ```