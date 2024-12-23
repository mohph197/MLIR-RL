import os

class CSVLogger:
    instance_folder: str
    sample_csv: str
    train_csv: str
    train_folder: str
    eval_folder: str

    def __init__(self, logs_folder: str, tags: list[str]):
        if not os.path.exists(logs_folder):
            self.instance_folder = f'{logs_folder}/0'
        else:
            dirs_indices: list[int] = []
            for folder in os.listdir(logs_folder):
                try:
                    dirs_indices.append(int(folder))
                except ValueError:
                    pass
            if len(dirs_indices) == 0:
                dir_index = 0
            else:
                dir_index = max(dirs_indices) + 1

            self.instance_folder = f'{logs_folder}/{dir_index}'
        self.train_folder = f'{self.instance_folder}/train'
        self.eval_folder = f'{self.instance_folder}/eval'
        os.makedirs(self.train_folder, exist_ok=True)
        os.makedirs(self.eval_folder, exist_ok=True)

        with open(f'{self.instance_folder}/tags.txt', 'w') as f:
            f.write('\n'.join(tags) + '\n')

        self.sample_csv = f'{self.instance_folder}/sample.csv'
        with open(self.sample_csv, 'w') as f:
            f.write('speedup,cumulative_reward\n')

        self.train_csv = f'{self.instance_folder}/train.csv'
        with open(self.train_csv, 'w') as f:
            f.write('policy_loss,value_loss,entropy,clip_factor\n')

    def log_sample(self, speedup: float, cumulative_reward: float):
        with open(self.sample_csv, 'a') as f:
            f.write(f'{speedup},{cumulative_reward}\n')

    def log_train(self, policy_loss: float, value_loss: float, entropy: float, clip_factor: float):
        with open(self.train_csv, 'a') as f:
            f.write(f'{policy_loss},{value_loss},{entropy},{clip_factor}\n')

    def log_bench_train(self, bench_name: str, speedup: float):
        bench_train = f'{self.train_folder}/{bench_name}.csv'
        if not os.path.exists(bench_train):
            with open(bench_train, 'w') as f:
                f.write('speedup\n')
                f.write(f'{speedup}\n')
        else:
            with open(bench_train, 'a') as f:
                f.write(f'{speedup}\n')

    def log_bench_eval(self, bench_name: str, speedup: float):
        bench_eval = f'{self.eval_folder}/{bench_name}.csv'
        if not os.path.exists(bench_eval):
            with open(bench_eval, 'w') as f:
                f.write('speedup\n')
                f.write(f'{speedup}\n')
        else:
            with open(bench_eval, 'a') as f:
                f.write(f'{speedup}\n')
