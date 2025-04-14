# GitLab CI/CD und SLURM Workflow für CoT-UQ

Diese Anleitung erklärt, wie du das CoT-UQ Projekt mit GitLab CI/CD und SLURM Cluster ausführst.

## GitLab CI/CD Workflow

### Übersicht
1. Dockerfile im Repository definiert die Umgebung
2. GitLab CI baut das Docker-Image und pusht es in die Registry
3. SLURM-Skripte ziehen das Image und wandeln es in Singularity-Format um
4. Experimente werden auf dem Cluster mit GPU-Unterstützung ausgeführt

### Erforderliche GitLab-Einstellungen
1. Repository mit GitLab verbinden
2. CI/CD-Variablen in den Repository-Einstellungen konfigurieren:
   - `CI_REGISTRY_USER` und `CI_REGISTRY_PASSWORD` für Docker Registry-Zugriff

### Docker-Build manuell ausführen (optional)
```bash
docker build -t registry.gitlab.com/your-group/your-project/cot-uq:latest .
docker push registry.gitlab.com/your-group/your-project/cot-uq:latest
```

## SLURM Workflow

### Vorbereitung
1. Repository klonen
   ```bash
   git clone https://github.com/yourusername/CoT-UQ.git
   cd CoT-UQ
   ```

2. Verzeichnisstruktur vorbereiten
   ```bash
   mkdir -p containers
   ```

### Experiment mit Registry-Image ausführen

1. **SLURM-Skript anpassen**

   Öffne `slurm_scripts/run_experiment_registry.sbatch` und ändere:
   ```bash
   REGISTRY_URL="registry.gitlab.com"
   IMAGE_PATH="your-group/your-project/cot-uq"  # An dein GitLab-Repo anpassen
   IMAGE_TAG="latest"  # Oder spezifischer Tag
   ```

2. **Job starten**
   ```bash
   sbatch slurm_scripts/run_experiment_registry.sbatch
   ```

   Dieses Skript:
   - Lädt das Container-Image aus der GitLab-Registry
   - Konvertiert es zu Singularity-Format falls nötig
   - Bindet das lokale Repository in den Container ein
   - Führt das Experiment mit GPU-Unterstützung aus

### Status überwachen

1. **Job-Status prüfen**
   ```bash
   squeue -u $USER
   ```

2. **Output prüfen**
   ```bash
   cat slurm-run-JOBID.out
   ```

## Vorteile dieses Ansatzes

- **Reproduzierbarkeit**: Gleiche Container-Umgebung überall
- **CI/CD**: Automatische Image-Erstellung bei jedem Push
- **Ressourceneffizienz**: Keine Container-Builds auf dem Cluster nötig
- **Flexibilität**: Einfacher Wechsel zwischen verschiedenen Image-Versionen

## Troubleshooting

- Bei Problemen mit dem Registry-Login: Prüfe die Zugangsdaten oder erstelle ein Personal Access Token
- Falls kein Zugriff auf die GitLab Registry möglich ist, kannst du als Alternative auch öffentliche Container-Registries wie Docker Hub verwenden
- Bei GPU-Problemen: Stelle sicher, dass `--nv` Flag im Singularity-Befehl vorhanden ist