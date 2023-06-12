import wandb
import sys
import pandas as pd
import time


OPTI_FILE_PATH = sys.argv[1]

with open(OPTI_FILE_PATH, 'r') as file:
    data = file.read()

def get_data(data):
    # create a list
    data = data.split('\n')

    # remove empty string
    data = list(filter(None, data))

    # string to dictionary
    data = [eval(i) for i in data]
    return data

data = get_data(data)

df = pd.DataFrame(data)


wandb.login(key="dbed837f90017d357c1c1c3e4b6e212abfdc0fa8") # API key of the serenet project on Massimo's account

metrics = ['val_loss', 'iou', 'matthews_coef']

# keep only certain columns
df_metrics = df[metrics]

# create a copy where there is all but the metrics columns 
df_hyperparams = df.drop(columns=metrics)


# simulation of a optimization run
for i in range(len(df_hyperparams)):

    # create a dictionary with the hyperparameters
    wandbConfig = dict(df_hyperparams.iloc[i])

    # create a dictionary with the metrics
    wandbMetrics = dict(df_metrics.iloc[i])

    wandb_kwargs = {
    "project":"segunet2",
    "config":wandbConfig,
    "reinit":True
    } 

    run = wandb.init(**wandb_kwargs)

    # log the metrics
    run.log(wandbMetrics)

    # end the run
    run.finish()

    # sleep for 2 second to not overload the server
    time.sleep(2)
    # sleep for 2 second to not overload the server
    time.sleep(2)