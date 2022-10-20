import subprocess
import tempfile

def run_shell_cmd(job_cmd):
    stdout, stderr = (tempfile.TemporaryFile(), tempfile.TemporaryFile())
    subprocess.run(job_cmd, shell=True, stdout=stdout, stderr=stderr)

    stdout.seek(0)
    stderr.seek(0)

    stdout_text = stdout.read().decode('UTF-8')
    stderr_text = stderr.read().decode('UTF-8')

    return stdout_text, stderr_text


def parse_nodelist(lst_string): 
    if '[' not in lst_string and ',' not in lst_string:
        return [lst_string]

    tokens = lst_string.split('[')
    prefix, suffix = tokens[0], tokens[1][:-1]
    noderanges = suffix.split(',')

    nodes = [] 
    for rng in noderanges:
        if '-' in rng:
            rsplit = rng.split('-')
            pad_len = len(rsplit[0])
            start, end = int(rsplit[0]), int(rsplit[1])
            nodes_to_add = [str(el) for el in range(start, end+1)]

            for node in nodes_to_add:
                while len(node) < pad_len:
                    node = '0' + node
                nodes.append(prefix + node)
        else:
            nodes.append(prefix + rng)
