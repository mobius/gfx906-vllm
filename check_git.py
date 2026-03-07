import subprocess
import os

os.chdir(r'D:\VLLM\vllm-gfx906')

print("=== Git Status ===")
result = subprocess.run(['git', 'status'], capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

print("\n=== Git Log ===")
result = subprocess.run(['git', 'log', '--oneline', '-5'], capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

print("\n=== Git Config ===")
result = subprocess.run(['git', 'config', 'user.name'], capture_output=True, text=True)
print("user.name:", result.stdout.strip() or "not set")
result = subprocess.run(['git', 'config', 'user.email'], capture_output=True, text=True)
print("user.email:", result.stdout.strip() or "not set")
