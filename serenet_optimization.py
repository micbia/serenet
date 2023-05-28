import os, random, numpy as np, sys
import tensorflow as tf
from glob import glob

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import ELU, LeakyReLU, PReLU, ReLU

from config.net_config import NetworkConfig
from utils_network.networks import Unet, SERENEt, FullSERENEt
from utils_network.metrics import get_avail_metris
from utils_network.callbacks import HistoryCheckpoint, SaveModelCheckpoint, ReduceLR
from utils_network.data_generator import LightConeGenerator, LightConeGenerator_SERENEt, LightConeGenerator_FullSERENEt
from utils.other_utils import get_data, get_data_lc, config_paths, config_path_opti
from utils_plot.plotting import plot_loss

import optuna

# title
print("""
  _____              _    _ _   _      _   
 / ____|            | |  | | \ | |    | |  
| (___   ___  __ _  | |  | |  \| | ___| |_ 
 \___ \ / _ \/ _` | | |  | | . ` |/ _ \ __|
 ____) |  __/ (_| | | |__| | |\  |  __/ |_ 
|_____/ \___|\__, |  \____/|_| \_|\___|\__|
              __/ |                        
             |___/                         
""")
config_file = sys.argv[1]
conf = NetworkConfig(config_file)

# create output file structure
PATH_OUT = config_path_opti(conf=conf, path_scratch=conf.SCRATCH_PATH, prefix='')
# copy config file to output path
os.system('cp %s %s' %(config_file, PATH_OUT))

def objective(trial):

    # Hyperparemeters
    coarse_dim = trial.suggest_int('coarse_dim', 128, 512,log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.5,step=0.02)
    kernel_size = trial.suggest_int('kernel_size', 3, 11)
    activation = trial.suggest_categorical('activation', ['relu', 'elu'])
    final_activation = trial.suggest_categorical('final_activation', ['linear', 'sigmoid',None])
    depth = trial.suggest_int('depth', 2, 6)
    pooling_type = trial.suggest_categorical('pooling_type', ['max', 'average'])

    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int('batch_size', 8, 64, log=True)

    wandbConfig = dict(trial.params)
    wandbConfig["trialNumber"] = trial.number

    # --------------------- NETWORK & RESUME OPTIONS ---------------------
    TYPE_NET = conf.AUGMENT
    RANDOM_SEED = 2022
    BATCH_SIZE = batch_size
    METRICS = [get_avail_metris(m) for m in conf.METRICS]

    if(isinstance(conf.LOSS, list)):
        LOSS = [get_avail_metris(loss) for loss in conf.LOSS]
        LOSS = {"out_imgSeg": LOSS[0], "out_imgRec": LOSS[1]}
        LOSS_WEIGHTS = {"out_imgSeg": 0.5, "out_imgRec": 0.5}
    else:
        LOSS = get_avail_metris(conf.LOSS)
    OPTIMIZER = Adam(lr=learning_rate)
    if isinstance(conf.DATASET_PATH, list):
        PATH_TRAIN = conf.IO_PATH+'inputs/'+conf.DATASET_PATH[0]
        PATH_VALID = conf.IO_PATH+'inputs/'+conf.DATASET_PATH[1]
    else:
        PATH_TRAIN = conf.IO_PATH+'inputs/'+conf.DATASET_PATH
        PATH_VALID = PATH_TRAIN
    ZIPFILE = (0 < len(glob(PATH_TRAIN+'data/*tar.gz')) and 0 < len(glob(PATH_VALID+'data/*tar.gz')))
    # TODO: if you want to restart from the previous best model set conf.RESUME_EPOCH = conf.BEST_EPOCH and loss need to be cut accordingly
    # -------------------------------------------------------------------
    random.seed(RANDOM_SEED)

    # Define GPU distribution strategy
    strategy = tf.distribute.MirroredStrategy()
    NR_GPUS = strategy.num_replicas_in_sync
    print ('Number of GPU devices: %d' %NR_GPUS)
    BATCH_SIZE *= NR_GPUS

    # Load data
    #size_train_dataset, size_valid_dataset = 10000*552, 1500*552
    size_train_dataset, size_valid_dataset = 10000//50, 1500//15


    train_idx = np.arange(0, size_train_dataset, dtype=int)
    valid_idx = np.arange(0, size_valid_dataset, dtype=int)

    # Create data generator from tensorflow.keras.utils.Sequence
    if(TYPE_NET == 'full_serenet'):
        train_generator = LightConeGenerator_FullSERENEt(path=PATH_TRAIN, data_temp=train_idx, data_shape=conf.IM_SHAPE, batch_size=BATCH_SIZE, shuffle=True)
        valid_generator = LightConeGenerator_FullSERENEt(path=PATH_VALID, data_temp=valid_idx, data_shape=conf.IM_SHAPE, batch_size=BATCH_SIZE, shuffle=True)

        # Define generator functional
        def generator_train():
            multi_enqueuer = tf.keras.utils.OrderedEnqueuer(train_generator, use_multiprocessing=False)
            multi_enqueuer.start(workers=10, max_queue_size=10)
            while True:
                batch_xs, batch_ys1, batch_ys2 = next(multi_enqueuer.get()) 
                yield ({'Image': batch_xs}, {'rec_out_img':batch_ys1, 'seg_out_img':batch_ys2})
                
        def generator_valid():
            multi_enqueuer = tf.keras.utils.OrderedEnqueuer(valid_generator, use_multiprocessing=False)
            multi_enqueuer.start(workers=10, max_queue_size=10)
            while True:
                batch_xs, batch_ys1, batch_ys2 = next(multi_enqueuer.get()) 
                yield ({'Image': batch_xs}, {'rec_out_img':batch_ys1, 'seg_out_img':batch_ys2})

        # Create dataset from data generator
        train_dataset = tf.data.Dataset.from_generator(generator_train, output_types=({'Image': tf.float32}, {'rec_out_img': tf.float32, 'seg_out_img': tf.float32}))
        valid_dataset = tf.data.Dataset.from_generator(generator_valid, output_types=({'Image': tf.float32}, {'rec_out_img': tf.float32, 'seg_out_img': tf.float32}))
    elif(TYPE_NET == 'serenet'):
        train_generator = LightConeGenerator_SERENEt(path=PATH_TRAIN, data_temp=train_idx, data_shape=conf.IM_SHAPE, batch_size=BATCH_SIZE, shuffle=True)
        valid_generator = LightConeGenerator_SERENEt(path=PATH_VALID, data_temp=valid_idx, data_shape=conf.IM_SHAPE, batch_size=BATCH_SIZE, shuffle=True)

        # Define generator functional
        def generator_train():
            multi_enqueuer = tf.keras.utils.OrderedEnqueuer(train_generator, use_multiprocessing=False)
            multi_enqueuer.start(workers=10, max_queue_size=10)
            while True:
                batch_xs1, batch_xs2, batch_ys = next(multi_enqueuer.get()) 
                yield ({'Image1': batch_xs1, 'Image2': batch_xs2}, {'out_img': batch_ys})
                
        def generator_valid():
            multi_enqueuer = tf.keras.utils.OrderedEnqueuer(valid_generator, use_multiprocessing=False)
            multi_enqueuer.start(workers=10, max_queue_size=10)
            while True:
                batch_xs1, batch_xs2, batch_ys = next(multi_enqueuer.get()) 
                yield ({'Image1': batch_xs1, 'Image2': batch_xs2}, {'out_img': batch_ys})
                
        # Create dataset from data generator
        train_dataset = tf.data.Dataset.from_generator(generator_train, output_types=({'Image1': tf.float32, 'Image2': tf.float32}, {'out_img': tf.float32}))
        valid_dataset = tf.data.Dataset.from_generator(generator_valid, output_types=({'Image1': tf.float32, 'Image2': tf.float32}, {'out_img': tf.float32}))
    elif(TYPE_NET == 'segunet' or TYPE_NET == 'recunet'):
        if(TYPE_NET == 'segunet'):
            DATA_TYPE = 'xH'
        elif(TYPE_NET == 'recunet'):
            DATA_TYPE = 'dT2'
        
        train_generator = LightConeGenerator(path=PATH_TRAIN, data_temp=train_idx, data_shape=conf.IM_SHAPE, zipf=ZIPFILE, batch_size=BATCH_SIZE, data_type=['dT4pca4', DATA_TYPE], shuffle=True)
        valid_generator = LightConeGenerator(path=PATH_VALID, data_temp=valid_idx, data_shape=conf.IM_SHAPE, zipf=ZIPFILE, batch_size=BATCH_SIZE, data_type=['dT4pca4', DATA_TYPE], shuffle=True)

        # Define generator functional
        def generator_train():
            multi_enqueuer = tf.keras.utils.OrderedEnqueuer(train_generator, use_multiprocessing=False)
            multi_enqueuer.start(workers=10, max_queue_size=10)
            while True:
                batch_xs, batch_ys = next(multi_enqueuer.get()) 
                yield batch_xs, batch_ys

        def generator_valid():
            multi_enqueuer = tf.keras.utils.OrderedEnqueuer(valid_generator, use_multiprocessing=False)
            multi_enqueuer.start(workers=10, max_queue_size=10)
            while True:
                batch_xs, batch_ys = next(multi_enqueuer.get()) 
                yield batch_xs, batch_ys

        # Create dataset from data generator
        train_dataset = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32, tf.float32), output_shapes=(tf.TensorShape([None]*(len(conf.IM_SHAPE)+2)), tf.TensorShape([None]*(len(conf.IM_SHAPE)+2))))
        valid_dataset = tf.data.Dataset.from_generator(generator_valid, output_types=(tf.float32, tf.float32), output_shapes=(tf.TensorShape([None]*(len(conf.IM_SHAPE)+2)), tf.TensorShape([None]*(len(conf.IM_SHAPE)+2))))

    # Distribute the dataset to the devices
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    valid_dist_dataset = strategy.experimental_distribute_dataset(valid_dataset)

    # Set the sharding policy to DATA
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_dataset.with_options(options)
    valid_dataset.with_options(options)

    print('\nModel on %d GPU\n' %NR_GPUS)
    hyperpar = {'coarse_dim': coarse_dim,
                'dropout': dropout,
                'kernel_size': kernel_size,
                'activation': activation,
                'final_activation': final_activation,
                'depth': depth,
                'pooling_type': pooling_type}

    # for Regression image + astropars
    if(TYPE_NET == 'full_serenet'):
        model = FullSERENEt(img_shape=np.append(conf.IM_SHAPE, 1), params=hyperpar, path=PATH_OUT)
        model.compile(optimizer=OPTIMIZER, loss=[LOSS, LOSS], loss_weights=LOSS_WEIGHTS, metrics=[METRICS, METRICS])
    elif(TYPE_NET == 'serenet'):
        model = SERENEt(img_shape1=np.append(conf.IM_SHAPE, 1), img_shape2=np.append(conf.IM_SHAPE, 1), params=hyperpar, path=PATH_OUT)
        model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)
    elif(TYPE_NET == 'segunet' or TYPE_NET == 'recunet'):
        model = Unet(img_shape=np.append(conf.IM_SHAPE, 1), params=hyperpar, path=PATH_OUT)
        model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)

    # define callbacks
    callbacks = [EarlyStopping(patience=100, verbose=1),
                ReduceLR(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7, verbose=1)]#,
                #SaveModelCheckpoint(PATH_OUT+'checkpoints/model-sem21cm_ep{epoch:d}.tf', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, best=RESUME_LOSS),
                #HistoryCheckpoint(filepath=PATH_OUT+'outputs/', verbose=0, save_freq=1, in_epoch=conf.RESUME_EPOCH)]


    # model fit
    results = model.fit(x=train_dist_dataset,
                        batch_size=BATCH_SIZE, 
                        epochs=conf.EPOCHS,
                        steps_per_epoch=size_train_dataset//BATCH_SIZE,
                        initial_epoch=conf.RESUME_EPOCH,
                        callbacks=callbacks, 
                        validation_data=valid_dist_dataset,
                        validation_steps=size_valid_dataset//BATCH_SIZE,
                        shuffle=True)


    wandbConfig["best_val_loss"] = np.min(results.history['val_loss'])

    # save optimization parameters and results for later use in wandb
    with open(PATH_OUT+'outputs/optimization.txt', 'a') as f:
        f.write(str(wandbConfig)+"\n")
    
    
    return wandbConfig["best_val_loss"]


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=2)

print("************* finished *****************")
print(study.best_params)
