import os
import subprocess
import tempfile

with open("../tensors/frostt_tensors.txt", 'r') as frostt_file:
    for line in frostt_file:
        if line.startswith("#"):
            continue
        filename = line.split("/")[-1].split(".gz")[0]
        
        if os.path.exists(f"../tensors/{filename}"):
            print(f"File {filename} already exists! Skipping download...")
        else:
            subprocess.run(f"curl -o ../tensors/{filename}.gz {line}", shell=True)
            subprocess.run(f"gunzip -d ../tensors/{filename}.gz", shell=True)

        stdout = tempfile.TemporaryFile() 
        stderr = tempfile.TemporaryFile() 
        subprocess.run(f"cat ../tensors/{filename} | wc -l", shell=True, stdout=stdout, stderr=stderr)

        stdout.seek(0)
        stderr.seek(0)
        linecount = int(stdout.read().decode('UTF-8'))
        print(f"{filename}: {linecount}")

        if os.path.exists(f"../tensors/{filename}_converted.hdf5"):
            print(f"File {filename} HDF5 exists! Skipping conversion...")
        else:
            subprocess.run(f"./build/process_frostt_tensor ../tensors/{filename} {linecount}", shell=True)

        print(f"Conversion of {filename} complete!")