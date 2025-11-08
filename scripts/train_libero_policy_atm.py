import os
import argparse


# environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# default track transformer path
DEFAULT_TRACK_TRANSFORMERS = {
    "libero_spatial": "./results/track_transformer/libero_track_transformer_libero-spatial/",
    "libero_object": "./results/track_transformer/libero_track_transformer_libero-object/",
    "libero_goal": "./results/track_transformer/libero_track_transformer_libero-goal/",
    "libero_10": "./results/track_transformer/libero_track_transformer_libero-100/",
}

# input parameters
parser = argparse.ArgumentParser()
parser.add_argument("--suite", default="libero_goal", choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"], 
                    help="The name of the desired suite, where libero_10 is the alias of libero_long.")
parser.add_argument("-tt", "--track-transformer", default=None, help="Then path to the trained track transformer.")
parser.add_argument("--use_traj_gen", action="store_true", help="Whether to use trajectory generation.")
parser.add_argument("--load_path", default=None, help="path to trained policy.")
parser.add_argument("--traj_gen_path", default=None, help="path to trained traj_gen.")
parser.add_argument("--ae_dir", default=None, help="path to trained autoencoder for traj_gen.")
parser.add_argument("--nfe", default=50, type=int, help="Number of forward evaluations.")
args = parser.parse_args()

print(f"args: {args}", flush=True)

# training configs
CONFIG_NAME = "libero_vilt"

train_gpu_ids =  [0, 1, 2, 3]
NUM_DEMOS = 10

root_dir = "./data/atm_libero/"
suite_name = args.suite
task_dir_list = os.listdir(os.path.join(root_dir, suite_name))
task_dir_list.sort()

# dataset
train_path_list = [f"{root_dir}/{suite_name}/{task_dir}/bc_train_{NUM_DEMOS}" for task_dir in task_dir_list]
val_path_list = [f"{root_dir}/{suite_name}/{task_dir}/val" for task_dir in task_dir_list]

track_fn = args.track_transformer or DEFAULT_TRACK_TRANSFORMERS[suite_name]

for seed in range(3):
    commond = (f'python -m engine.train_bc --config-name={CONFIG_NAME} train_gpus="{train_gpu_ids}" '
                f'experiment=atm-policy_{suite_name.replace("_", "-")}_demo{NUM_DEMOS} '
                f'train_dataset="{train_path_list}" val_dataset="{val_path_list}" '
                f'model_cfg.track_cfg.track_fn={track_fn} '
                f'model_cfg.track_cfg.use_zero_track=False '
                f'model_cfg.spatial_transformer_cfg.use_language_token=False '
                f'model_cfg.temporal_transformer_cfg.use_language_token=False '
                f'seed={seed} '
                f'model_cfg.load_path={args.load_path} '
                f'+model_cfg.traj_gen_path={args.traj_gen_path} '
                f'+model_cfg.ae_dir={args.ae_dir} '
                f'+model_cfg.use_traj_gen={args.use_traj_gen} ' 
                f'+model_cfg.nfe={args.nfe}')

    os.system(commond)
