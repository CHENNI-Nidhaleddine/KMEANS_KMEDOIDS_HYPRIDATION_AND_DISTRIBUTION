# launch_clients.py
import subprocess

num_clients = 10  # Set the number of clients you want to launch

for _ in range(num_clients):
    subprocess.Popen(["python", "client.py"])