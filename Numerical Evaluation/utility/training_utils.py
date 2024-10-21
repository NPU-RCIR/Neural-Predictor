'''
@Filename       : training_utils.py
@Description    : 
@Create_Time    : 2024/07/16 09:30:14
@Author         : Ao Jin
@Email          : jinao.nwpu@outlook.com
'''

from torch.utils.tensorboard import SummaryWriter

def create_writer(folder: str,
                  experiment_name: str, 
                  model_name: str, 
                  extra: str=None,
                  timestamp=False):
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """
    from datetime import datetime
    import os

    # Get timestamp of current date (all experiments on certain day live in same folder)
    date = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

    if extra:
        if timestamp:
            log_dir = os.path.join(folder, date, experiment_name, model_name, extra)
        else:
            log_dir = os.path.join(folder, experiment_name, model_name, extra)
    else:
        if timestamp:
            log_dir = os.path.join(folder, date, experiment_name, model_name)
        else:
            log_dir = os.path.join(folder, experiment_name, model_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir), log_dir