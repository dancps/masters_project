from tensorflow.python.platform import build_info
import tensorflow as tf

def main():
    
    print("cudnn_version",build_info.build_info['cudnn_version'])
    print("cuda_version",build_info.build_info['cuda_version'])


    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print("We got a GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print("Sorry, no GPU for you...")

if(__name__=='__main__'):
    main()