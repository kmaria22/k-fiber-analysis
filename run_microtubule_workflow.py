import subprocess

# Run data loading
result = subprocess.run(['python', 'load_data_TARDIS.py'], check=True)

# Only run pre-analysis if data loading succeeded
if result.returncode == 0:
    subprocess.run(['python', 'pre_analysis.py'], check=True)
