# test_basic_setup.py
import os
import sys
import platform
import time

# Versuche, einige Kernbibliotheken zu importieren, die im Container sein sollten
print("="*50)
print("Python Basic Info:")
print(f"  Executable: {sys.executable}")
print(f"  Version: {sys.version}")
print(f"  Platform: {platform.platform()}")
print("="*50)

print("\n" + "="*50)
print("Attempting to import key libraries...")
try:
    import torch
    print(f"  SUCCESS: Imported torch (Version: {torch.__version__})")
except ImportError as e:
    print(f"  FAILED: Could not import torch. Error: {e}")

try:
    import transformers
    print(f"  SUCCESS: Imported transformers (Version: {transformers.__version__})")
except ImportError as e:
    print(f"  FAILED: Could not import transformers. Error: {e}")

# Füge hier ggf. weitere wichtige Imports hinzu

print("="*50)


# Zeige das aktuelle Arbeitsverzeichnis IM CONTAINER an
print("\n" + "="*50)
try:
    cwd = os.getcwd()
    print(f"Current working directory inside container: {cwd}")
except Exception as e:
    print(f"Could not get current working directory. Error: {e}")
print("="*50)

# Zeige den Inhalt des gemounteten Projektverzeichnisses an
# Passe den Pfad an, je nachdem wohin du mountest (hier /project angenommen)
print("\n" + "="*50)
project_dir_in_container = "/project" # Oder /opt/CoT-UQ, wenn du das verwendest
print(f"Listing contents of project directory ({project_dir_in_container}) inside container:")
try:
    if os.path.exists(project_dir_in_container):
        for item in os.listdir(project_dir_in_container):
            print(f"  - {item}")
    else:
        print(f"  Directory '{project_dir_in_container}' does NOT exist inside the container!")
except Exception as e:
    print(f"Could not list directory '{project_dir_in_container}'. Error: {e}")
print("="*50)


print("\nBasic setup test finished successfully at:", time.strftime("%Y-%m-%d %H:%M:%S"))
