import os
import subprocess
import tempfile

with open("tensors/frostt_tensors.txt", 'r') as frostt_file:
    for line in frostt_file:
        filename = line.split("/")[-1].split(".gz")[0]
        
        if os.exists(f"tensors/{filename}"):
            print(f"File {filename} already exists! Skipping download...")
        else:
            subprocess.run(f"curl -o tensors/{filename}.gz {line}", shell=True)
            subprocess.run(f"gunzip -D tensors/{filename}.gz", shell=True)

            stdout = tempfile.TemporaryFile() 
            stderr = tempfile.TemporaryFile() 
            subprocess.run(f"cat tensors/{filename} | wc -l", stdout=stdout, stderr=stderr)

            stdout.seek(0)
            stderr.seek(0)

        linecount = int(stdout.read().decode('UTF-8'))

        if os.exists(f"tensors/{filename}_converted.hdf5"):
            print(f"File {filename} HDF5 exists! Skipping conversion...")
        else:
            subprocess.run(f". tensor_io/build/process_frostt_tensor {filename} {linecount}", shell=True)
