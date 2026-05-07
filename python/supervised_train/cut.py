import rosbag2_py
import os

# 設定
input_bag = 'data.db3'  # 読み込むファイル名
output_bag = 'data'  # 保存先ディレクトリ名

def get_rosbag_options(path, serialization_format='cdr'):
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)
    return storage_options, converter_options

# 1. まず全体の終了時間を把握する
reader = rosbag2_py.SequentialReader()
storage_options, converter_options = get_rosbag_options(input_bag)
reader.open(storage_options, converter_options)

last_ts = 0
while reader.has_next():
    (topic, data, t) = reader.read_next()
    last_ts = max(last_ts, t)

# 終了の5秒前を計算 (単位はナノ秒)
cutoff_ts = last_ts - (5 * 10**9)

# 2. カットオフ時間までを新しいバッグに書き出す
reader = rosbag2_py.SequentialReader() # リセット
reader.open(storage_options, converter_options)

writer = rosbag2_py.SequentialWriter()
storage_options_out, converter_options_out = get_rosbag_options(output_bag)
writer.open(storage_options_out, converter_options_out)

# トピック情報の登録
topics = reader.get_all_topics_and_types()
for topic in topics:
    writer.create_topic(topic)

while reader.has_next():
    (topic, data, t) = reader.read_next()
    if t <= cutoff_ts:
        writer.write(topic, data, t)
    else:
        break

print(f"完了！ {output_bag} に保存されました。")