# SLURM Workflow für CoT-UQ mit GitLab CI/CD

Diese Anleitung erklärt, wie das CoT-UQ Projekt mit GitLab CI/CD und auf dem SLURM-Cluster ausgeführt wird.

## Übersicht des Workflows

1. **Vorbereitung**: Dockerfile und CI/CD im Repository konfigurieren
2. **GitLab CI**: Automatischer Build und Push des Docker-Images in die Registry
3. **SLURM**: Container aus Registry ziehen und für Tests/Experimente verwenden

## Für Entwickler: GitLab Setup

1. **Starte Pipeline im GitLab**
   - Pushe Code inkl. Dockerfile und .gitlab-ci.yml auf dein GitLab-Repository
   - Warte bis die CI/CD Pipeline den Container gebaut und in die Registry gepusht hat

## Für Cluster-Nutzung: Testlauf auf SLURM

1. **Repository klonen**
   ```bash
   git clone https://github.com/yourusername/CoT-UQ.git
   cd CoT-UQ
   ```

2. **Container aus Registry ziehen**
   ```bash
   # Container-Verzeichnis erstellen
   mkdir -p containers
   
   # SLURM-Job starten, der das Image aus der Registry zieht
   sbatch slurm_scripts/pull_test_container.sbatch
   ```
   
   **WICHTIG**: Passe vorher in `pull_test_container.sbatch` die Registry-URL und Image-Pfade an
   ```
   REGISTRY_URL="registry.gitlab.com"
   IMAGE_PATH="your-group/your-project/cot-uq"  # Anpassen!
   ```

3. **CPU-Tests ausführen**
   ```bash
   sbatch slurm_scripts/test_simple.sbatch
   ```
   
   Dieser Job testet das Python-Setup und die Imports im Container.

4. **GPU-Tests ausführen (falls GPU verfügbar)**
   ```bash
   sbatch slurm_scripts/test_gpu.sbatch
   ```
   
   Dieser Job testet die GPU-Verfügbarkeit und -Leistung mit PyTorch.

5. **Experiment ausführen**
   ```bash
   sbatch slurm_scripts/run_experiment_registry.sbatch
   ```
   
   **WICHTIG**: Passe vorher in `run_experiment_registry.sbatch` die Registry-URL und Image-Pfade an!

## Status und Protokolle prüfen

```bash
# Aktive Jobs anzeigen
squeue -u $USER

# Ausgabe des Pull-Jobs prüfen
cat slurm-pull-*.out

# Test-Ausgabe prüfen
cat slurm-test-*.out

# GPU-Test-Ausgabe prüfen
cat slurm-gpu-*.out
```

## Datei-Erklärungen

- **Dockerfile**: Definiert die Container-Umgebung
- **.gitlab-ci.yml**: Konfiguriert automatische Builds
- **pull_test_container.sbatch**: Lädt Container aus der Registry
- **test_simple.sbatch**: Führt CPU-basierte Tests aus
- **test_gpu.sbatch**: Führt GPU-basierte Tests aus
- **run_experiment_registry.sbatch**: Führt das Hauptexperiment aus

## Anmerkungen

- Die Jobs werden vom aktuellen Verzeichnis aus ausgeführt
- Container werden in `./containers/` gespeichert
- Für GPU-Tests und -Experimente wird das `--nv` Flag benötigt
- Für private GitLab-Registries werden Anmeldedaten benötigt

Siehe auch `README_GITLAB.md` für weitere Details zum GitLab CI/CD-Workflow.