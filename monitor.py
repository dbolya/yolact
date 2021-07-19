#! /usr/bin/python3

from argparse import ArgumentParser
import json
from pathlib import Path
from torch._C import default_generator
import visdom
import torch
import time
import tqdm


class RoughTimer:
    def __init__(self, period: float):
        self.period = period
        self.record = time.time()

    def sleep(self):
        now = time.time()
        remaining = self.record + self.period - now
        if remaining > 0:
            time.sleep(remaining)
        self.record = time.time()


def monitor():
    parser = ArgumentParser("get loss from log")
    parser.add_argument("--path", type=str, default="logs/yolact_resnet50.log",
                        help="path to the log file")
    parser.add_argument("--env", type=str, default="monitor",
                        help="visdom environment")
    parser.add_argument("--period", type=float, default=5.0,
                        help="update period")
    parser.add_argument("--session", type=int, default=0,
                        help="only show this session. show all by default")
    args = parser.parse_args()

    # visdom instance
    vis = visdom.Visdom(server="10.201.159.235", port=8097, env=args.env)

    # visdom window cache
    windows = {}
    pth = Path(args.path)
    for attr in dir(args):
        if attr.startswith('_'):
            continue
        print("{}: {}".format(attr, getattr(args, attr)))
    # print("Update period : {} sec".format(args.period))
    timer = RoughTimer(args.period)

    try:
        # tqdm with inifinite iterator to see how many updates are occured.
        for x in tqdm.tqdm(iter(int, 1)):
            with pth.open('r') as f:
                lines = f.readlines()

                logs = {"train": {}, "val": {}}
                legends = {}

                for line in lines:
                    parsed = json.loads(line)
                    s_type = parsed["type"]
                    session = str(parsed["session"])
                    if (args.session != 0) and (args.session != int(session)):
                        continue
                    if s_type == "train":
                        iteration = parsed["data"]["iter"]
                        epoch = parsed["data"]["epoch"]

                        loss = parsed["data"]["loss"]
                        if session not in logs["train"].keys():
                            logs["train"][session] = {
                                "epoch": [], "iter": [], "loss": []}
                            legends["train"] = list(loss.keys())

                        logs["train"][session]["epoch"].append(epoch)
                        logs["train"][session]["iter"].append(iteration)
                        logs["train"][session]["loss"].append(
                            list(loss.values()))

                    if s_type == "val":
                        iteration = parsed["data"]["iter"]
                        epoch = parsed["data"]["epoch"]

                        box = parsed["data"]["box"]
                        mask = parsed["data"]["mask"]
                        if session not in logs["val"].keys():
                            logs["val"][session] = {
                                "epoch": [], "iter": [], "box": [], "mask": []}
                            legends["val"] = list(mask.keys())

                        logs["val"][session]["epoch"].append(epoch)
                        logs["val"][session]["iter"].append(iteration)
                        logs["val"][session]["box"].append(list(box.values()))
                        logs["val"][session]["mask"].append(
                            list(mask.values()))

                for s in logs["train"]:
                    x = torch.Tensor(logs["train"][s]['iter'])
                    y = torch.Tensor(logs["train"][s]["loss"])
                    opts = {
                        "legend": legends["train"],
                        "title": "session "+str(s) + " (train)"
                    }
                    win_key = "train_{}".format(s)
                    if win_key in windows.keys():
                        vis.line(Y=y, X=x, opts=opts, update="replace",
                                 win=windows[win_key])
                    else:
                        windows[win_key] = vis.line(Y=y, X=x, opts=opts)

                for s in logs["val"]:
                    x = torch.Tensor(logs["val"][s]['iter'])
                    y = torch.Tensor(logs["val"][s]["mask"])
                    opts = {
                        "legend": legends["val"],
                        "title": "session "+str(s) + " (val)"
                    }
                    win_key = "val_{}".format(s)
                    if win_key in windows.keys():
                        vis.line(Y=y, X=x, opts=opts, update="replace",
                                 win=windows[win_key])
                    else:
                        windows[win_key] = vis.line(Y=y, X=x, opts=opts)

            timer.sleep()

    except KeyboardInterrupt:
        for k, v in windows.items():
            print("close {}: {}".format(k, v))
            vis.close(win=v, env=args.env)
        print("close all windwos. bye bye!")


if __name__ == "__main__":
    monitor()
