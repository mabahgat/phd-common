import subprocess
import tensorflow as tf

# GPU_INDEX = 3

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices(physical_devices[GPU_INDEX:], 'GPU')

# 'Using', physical_devices[GPU_INDEX]


def set_gpu_to_next_available(min_memory_mb=20_000):
    physical_devices_lst = tf.config.list_physical_devices('GPU')
    if not physical_devices_lst:
        raise Exception('No GPUs found')

    memory_lst = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        check=True,
        stdout=subprocess.PIPE,
        universal_newlines=True
        ).stdout.splitlines()
    gpu_index = -1
    for idx, mem in enumerate(memory_lst):
        if int(mem) >= min_memory_mb:
            gpu_index = idx
            break
    if gpu_index == -1:
        raise Exception('Could not find a gpu with free memory of {} out of {} gpus'.format(min_memory_mb, len(memory_lst)))

    tf.config.set_visible_devices(physical_devices_lst[gpu_index:], 'GPU')
    return physical_devices_lst[gpu_index]


if __name__ == '__main__':
    print(set_gpu_to_next_available())
