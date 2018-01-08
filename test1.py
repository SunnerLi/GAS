import subprocess

record_file_name = "gpu_usage.txt"

def recordGPUUsage():
    """
        Dump the nvidia-smi information into file
        * Notice: the imformation will be flashed while you call this function again
    """
    global record_file_name
    nvidia_args = ['nvidia-smi', '-f', record_file_name]
    cmd = " ".join(nvidia_args)
    subprocess.call(cmd, shell=True)    

if __name__ == '__main__':
    recordGPUUsage()

