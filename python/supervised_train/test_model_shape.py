import torch
from src.models.cnn import TinyLidarActionLstmPPO

def test_ppo_model():
    # 1. パラメータ設定
    input_dim = 1080
    output_dim = 2
    batch_size = 8
    seq_len = 10

    # 2. モデルのインスタンス化
    model = TinyLidarActionLstmPPO(input_dim=input_dim, output_dim=output_dim)
    model.eval() # 推論モードに切り替え
    print("[+] モデルのインスタンス化に成功しました")

    # 3. ダミーデータの作成
    # scan_seq: [batch, seq, dim]
    # prev_action: [batch, seq, action_dim]
    scan_seq = torch.randn(batch_size, seq_len, input_dim)
    prev_action_seq = torch.randn(batch_size, seq_len, 2)

    # 4. モデルの実行
    print(f"[*] 入力データ: scan_seq {scan_seq.shape}, prev_action {prev_action_seq.shape}")
    
    with torch.no_grad():
        action, value, hidden = model(scan_seq, prev_action_seq)

    # 5. 結果の検証
    print("[+] モデルのforward実行に成功しました")
    print(f"  > Action shape: {action.shape} (期待値: [8, 10, 2])")
    print(f"  > Value shape:  {value.shape}  (期待値: [8, 10, 1])")
    
    # 形状チェック
    assert action.shape == (batch_size, seq_len, output_dim), "Actionの次元が違います"
    assert value.shape == (batch_size, seq_len, 1), "Valueの次元が違います"
    print("\n[✔] すべてのテストを通過しました！")

if __name__ == "__main__":
    test_ppo_model()