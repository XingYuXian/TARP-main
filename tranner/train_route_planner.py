import os
import pickle
import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader, random_split
from itertools import cycle
from os.path import join

from imputation.history_avg_at_time_interval import history_avg_at_time_interval
from planner.route_planner import TransPlanner
from data_preprocess import count_od_frequency


class RouteDataset(Dataset):
    def __init__(self, data_dict: dict):
        self.route = data_dict["route"]
        self.travel_time_cost = data_dict["travel_time_cost_imputation"]
        self.start_time = data_dict["start_time"]
        self.route_length = data_dict["route_length"]
        self.route_padding = data_dict["route_padding"]

    def __len__(self):
        return len(self.route)

    def __getitem__(self, idx):
        return {
            "route": torch.tensor(self.route[idx], dtype=torch.long),
            "travel_time_cost_imputation": torch.tensor(self.travel_time_cost[idx], dtype=torch.float),
            "start_time": torch.tensor(self.start_time[idx], dtype=torch.long),
            "route_length": torch.tensor(self.route_length[idx], dtype=torch.long),
            "route_padding": torch.tensor(self.route_padding[idx], dtype=torch.long),
        }


def route_collate_fn(batch):
    routes = [item["route"] for item in batch]
    times = [item["travel_time_cost_imputation"] for item in batch]
    paddings = [item["route_padding"] for item in batch]
    lengths = torch.stack([item["route_length"] for item in batch])
    starts = torch.stack([item["start_time"] for item in batch])

    return {
        "route": routes,
        "travel_time_cost": times,
        "route_padding": paddings,
        "route_length": lengths,
        "start_time": starts
    }


class Trainer:
    def __init__(self,
                 model: TransPlanner,
                 train_data: Dataset,
                 test_data: Dataset,
                 device: torch.device,
                 model_path: str,
                 ):
        self.model = model
        self.train_dataset = train_data
        self.test_dataset = test_data
        self.device = device
        self.model_path = model_path

    def train(self, n_epoch, batch_size, lr, scaler=None):
        optimizer = torch.optim.Adam(self.model.parameters(), lr)
        # split train test
        trainloader = DataLoader(
            self.train_dataset,
            batch_size,
            collate_fn=route_collate_fn
        )
        testloader = DataLoader(
            self.test_dataset,
            batch_size,
            collate_fn=route_collate_fn
        )
        self.model.train()

        iter, train_loss_avg = 0, 0
        # scaler = GradScaler()
        try:
            for epoch in range(n_epoch):
                for xs in trainloader:
                    data = xs['route']
                    start_time = xs['start_time']
                    optimizer.zero_grad()
                    #with autocast(device_type='cuda'):
                    loss = self.model(data, start_time)
                    assert not torch.any(torch.isnan(loss)), f"在iter =  {iter} 时，损失值中存在NaN"
                    train_loss_avg += loss.item()

                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    # scaler.scale(loss).backward()
                    # TODO: clip norm
                    # scaler.unscale_(optimizer)
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    # scaler.step(optimizer)
                    # scaler.update()
                    iter += 1
                    if iter % 20 == 0 or iter == 1:
                        denom = 1 if iter == 1 else 20
                        # eval test
                        test_loss = next(self.eval_test(testloader))
                        print(
                            f"e: {epoch}, i: {iter}, train loss: {train_loss_avg / denom: .4f}, test loss: {test_loss: .4f}")
                        # for name, parms in self.model.named_parameters():
                        #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad)
                        train_loss_avg = 0.

        except KeyboardInterrupt as E:
            print("Training interruptted, begin saving...")
            self.model.eval()
            model_name = f"tmp_iter_{iter}.pth"
        # save
        self.model.eval()
        model_name = f"finished_{iter}.pth"
        model_dict = self.model.state_dict()
        torch.save(model_dict, join(self.model_path, model_name))
        print("save finished!")

    def eval_test(self, test_loader):
        with torch.no_grad():
            for txs in cycle(test_loader):
                test_loss = self.model(txs['route'], txs['start_time'])
                yield test_loss.item()


@hydra.main(config_path="../config", config_name="train_planner_config", version_base="1.3")
def main(cfg: DictConfig):
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    device = torch.device(cfg.device if cfg.device != "default" else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # 加载图数据
    G = pickle.load(open(join(cfg.data_dir, "xian_G.pkl"), "rb"))
    A = torch.load(join(cfg.data_dir, "xian_A.ts"))
    print("Graph loaded.")

    # time_data_avg_all = np.load(cfg.time_pickle_path, allow_pickle=True).item()
    HA = r"D:\Work\PostGraduate\DI_RP\wandbTest\imputation\traffic_matrix.npz"
    ha_data = np.load(HA, allow_pickle=True)
    ha_data = dict(ha_data.items())
    time_data_avg_all = ha_data['traffic_matrix']
    segment_mapping = pd.read_pickle(cfg.seg_map_path)

    train_data = pd.read_pickle(join(cfg.data_dir, "train_data.pkl"))
    test_data = pd.read_pickle(join(cfg.data_dir, "test_data.pkl"))
    val_data = pd.read_pickle(join(cfg.data_dir, "val_data.pkl"))

    train_dataset = RouteDataset(train_data)
    val_dataset = RouteDataset(val_data)
    test_dataset = RouteDataset(test_data)
    # avg_len = sum(len(r) for r in train_dataset.route) / len(train_dataset.route)
    # print("train_dataset平均路径序列长度:", avg_len)
    # print(count_od_frequency(train_dataset.route))
    # avg_len = sum(len(r) for r in val_dataset.route) / len(val_dataset.route)
    # print("val_dataset平均路径序列长度:", avg_len)
    # print(count_od_frequency(val_dataset.route))
    # avg_len = sum(len(r) for r in test_dataset.route) / len(test_dataset.route)
    # print("test_dataset平均路径序列长度:", avg_len)
    # print(count_od_frequency(test_dataset.route))
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=route_collate_fn)

    # 模型与训练器
    model = TransPlanner(
        G, A, device, x_emb_dim=cfg.x_emb_dim,
        segment_mapping=segment_mapping,
        time_dist_seg=time_data_avg_all,
        pretrain_path=None
    )
    trainer = Trainer(model, train_dataset, val_dataset, device, cfg.model_path)

    # 保存配置
    os.makedirs(cfg.model_path, exist_ok=True)
    with open(join(cfg.model_path, f"{cfg.model_name}.info"), "w") as f:
        f.write(str(cfg))

    trainer.train(cfg.n_epoch, cfg.bs, cfg.lr)
    print("Training completed.")


if __name__ == "__main__":
    main()
