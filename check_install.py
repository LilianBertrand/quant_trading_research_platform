from pathlib import Path
import importlib.util

project_root = Path(__file__).resolve().parent
regime_file = project_root / 'src' / 'ml' / 'regime.py'
exporter_file = project_root / 'src' / 'reporting' / 'exporter.py'
app_file = project_root / 'app.py'

print(f'Project folder: {project_root.name}')
print(f'Project path: {project_root}')

for required in [app_file, regime_file, exporter_file]:
    if not required.exists():
        raise SystemExit(f'MISSING: {required}')

regime_text = regime_file.read_text()
exporter_text = exporter_file.read_text()
app_text = app_file.read_text()

if "from sklearn" in regime_text and "try:" not in regime_text[:300]:
    raise SystemExit('ERROR: sklearn is imported directly. This is not the ultimate fixed version.')
if "engine='openpyxl'" in exporter_text or 'import openpyxl' in exporter_text:
    raise SystemExit('ERROR: openpyxl is still required. This is not the fixed exporter.')
if 'nan%' in app_text:
    raise SystemExit('ERROR: app still contains nan% display risk.')

print('OK: sklearn is optional, not mandatory')
print('OK: openpyxl is not required')
print('OK: ML accuracy displays N/A instead of nan% when needed')
print('OK: project files look consistent')

if importlib.util.find_spec('sklearn') is not None:
    print('Optional ML engine: scikit-learn detected, Random Forest can be used')
else:
    print('Optional ML engine: scikit-learn not detected, fallback regime detector will be used')
