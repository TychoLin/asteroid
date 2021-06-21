import argparse
import os

import musdb
import museval
import numpy as np
import soundfile as sf
import yaml
from asteroid.data import MUSDB18Dataset
from asteroid.dsp.normalization import normalize_estimates
from asteroid.losses import PITLossWrapper, pairwise_mse
from asteroid.models import ConvTasNet
from asteroid.utils import tensors_to_device
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--use_gpu", type=int, default=0, help="Whether to use the GPU for model execution"
)
parser.add_argument("--root", type=str, help="The path to the MUSDB18 dataset")
parser.add_argument("--exp_dir", default="exp/tmp", help="Experiment root")


def main(conf):
    model_path = os.path.join(conf["exp_dir"], "best_model.pth")
    model = ConvTasNet.from_pretrained(model_path)
    # Handle device placement
    if conf["use_gpu"]:
        model.cuda()
    model_device = next(model.parameters()).device
    sources = conf["train_conf"]["data"]["sources"]
    test_set = MUSDB18Dataset(
        root=conf["root"],
        split="minitest",
        sources=sources,
        targets=sources,
    )
    mus = musdb.DB(root=conf["root"], subsets="minitest")
    # Used to reorder sources only
    loss_func = PITLossWrapper(pairwise_mse, pit_from="pw_mtx")

    ex_save_dir = os.path.join(conf["exp_dir"], "EvaluateResults_musdb18_testdata/")

    for idx in tqdm(range(len(test_set))):
        mix, targets = tensors_to_device(test_set[idx], device=model_device)
        pred = model(mix.unsqueeze(0))
        loss, reordered_estimates = loss_func(pred, targets.unsqueeze(0), return_est=True)
        mix_np = mix.cpu().data.numpy()
        estimates_np = reordered_estimates.squeeze(0).cpu().data.numpy()
        estimates_np_normalized = normalize_estimates(estimates_np, mix_np)

        track_name = os.path.basename(test_set.tracks[idx]["path"])
        track_dir = os.path.join(ex_save_dir, track_name + "/")
        os.makedirs(track_dir, exist_ok=True)

        estimates = {}

        for i, estimate in enumerate(estimates_np_normalized):
            instrument_name = conf["train_conf"]["data"]["sources"][i]
            sf.write(track_dir + "{}.wav".format(instrument_name), estimate.T, conf["sample_rate"])
            estimates[instrument_name] = estimate.T

        scores = museval.eval_mus_track(mus[idx], estimates, output_dir=track_dir)

        print(scores)


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    # Load training config
    conf_path = os.path.join(args.exp_dir, "conf.yml")
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)
    arg_dic["sample_rate"] = train_conf["data"]["sample_rate"]
    arg_dic["train_conf"] = train_conf

    main(arg_dic)
