import tensorflow as tf


def device_setup(config):
    if config.devices.get("log_device_placement", False):
        tf.debugging.set_log_device_placement(True)
    devices = tf.config.experimental.list_physical_devices("GPU")
    if devices:
        print("GPU devices found")
        if config.devices.get("memory_growth", False):
            print("Enabling memory growth")
            tf.config.experimental.set_memory_growth(devices[0], True)
    else:
        print("No GPU device found")
