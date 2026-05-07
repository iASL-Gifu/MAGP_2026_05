import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

def _init_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        # SB3のデフォルトに近い直交初期化
        nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
        if module.bias is not None:
            module.bias.data.fill_(0.0)

class TinyLidarExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        # 最終的にこの Extractor が出力する次元数を設定 (この出力が SB3 の LSTM に入る)
        super().__init__(observation_space, features_dim)
        
        # 1. Lidarスキャン用の CNN (元のコードを移植)
        # observation_space['scans'] の形状を取得 (例: 1080)
        scan_dim = observation_space["scans"].shape[0]
        
        # SB3 の FrameStack を使う場合、チャネル数は n_stack になります
        # もし Stack しないなら in_channels=1 です
        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3)
        
        # 畳み込み後の次元数をダミー入力で計算
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, scan_dim)
            cnn_out = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(dummy_input)))))
            self.cnn_flatten_dim = cnn_out.view(1, -1).shape[1]
            
        # 2. State (前回アクション等) の次元数
        state_dim = observation_space["state"].shape[0]
        
        # 3. CNNの特徴量と State を結合し、最終的な features_dim に変換する層
        self.fc = nn.Linear(self.cnn_flatten_dim + state_dim, features_dim)

        # 重みの初期化
        self.apply(_init_weights)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations は Dict 形式で入ってきます
        scans = observations["scans"] # shape: (batch_size, seq_len * scan_dim) or (batch_size, scan_dim)
        states = observations["state"]
        
        # --- CNN の処理 ---
        # SB3 は Dict 入力を Flatten して渡してくることがあるので、形状を整える
        # nn.Conv1d は (batch_size, channels, length) を期待する
        x = scans.unsqueeze(1) if scans.dim() == 2 else scans
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        
        # Flatten
        cnn_features = x.view(x.size(0), -1)
        
        # --- 特徴量の結合と最終出力 ---
        # CNNの出力と states (前回のアクションや速度など) を結合
        combined_features = torch.cat((cnn_features, states), dim=1)
        
        # 指定された features_dim (この値がLSTMの input_size になる) に変換
        return F.relu(self.fc(combined_features))