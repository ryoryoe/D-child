import pandas as pd
import numpy as np
import sys

#役割
#細かいメッシュを平均化することで粗いメッシュを作成する


# CSVファイルの読み込み
input_path = r"/Users/tanakaryo/Documents/openfoam/obstacle_flow_test/cube_obstacle_two_dimension2/most_laugh.csv"
output_path = r"/Users/tanakaryo/Documents/openfoam/obstacle_flow_test/cube_obstacle_two_dimension2/coarse_mesh.csv"
data = pd.read_csv(input_path)

# メッシュの範囲
x_min, x_max = -0.8, 2.4
y_min, y_max = -0.8, 2.4
mesh_size = 0.1  # 各メッシュのサイズ

# グリッドを設定
x_bins = np.arange(x_min, x_max + mesh_size, mesh_size)
y_bins = np.arange(y_min, y_max + mesh_size, mesh_size)

# x座標とy座標の列を取得
x_coords = data['Points:0']
y_coords = data['Points:1']

# グループ化して平均を計算
data['x_bin'] = np.digitize(x_coords, x_bins) - 1
data['y_bin'] = np.digitize(y_coords, y_bins) - 1

# インデックスが範囲外の場合の処理
data['x_bin'] = data['x_bin'].clip(0, len(x_bins) - 2)
data['y_bin'] = data['y_bin'].clip(0, len(y_bins) - 2)

# 各メッシュ内の平均を計算
averaged_data = data.groupby(['x_bin', 'y_bin']).mean().reset_index()

# メッシュに戻すための配列を準備 (z方向は1つだけ)
coarse_mesh = np.zeros((len(x_bins) - 1, len(y_bins) - 1, 3))  # U:0, U:1, U:2を保存

# 平均化された速度を粗いメッシュに適用
for index, row in averaged_data.iterrows():
    x_idx = int(row['x_bin'])
    y_idx = int(row['y_bin'])

    # メッシュの範囲
    x_min_bin = x_bins[x_idx]
    x_max_bin = x_bins[x_idx + 1]
    y_min_bin = y_bins[y_idx]
    y_max_bin = y_bins[y_idx + 1]

    # 障害物領域の確認 (xとyが0から1.6の範囲内にあるかどうか)
    if (x_min_bin < 1.6 and x_max_bin > 0) and (y_min_bin < 1.6 and y_max_bin > 0):
        # 障害物内の速度を0に設定
        coarse_mesh[x_idx, y_idx, 0] = 0.0
        coarse_mesh[x_idx, y_idx, 1] = 0.0
        coarse_mesh[x_idx, y_idx, 2] = 0.0

    else:
        # U:0, U:1の平均値をメッシュに適用
        coarse_mesh[x_idx, y_idx, 0] = row['U:0']
        coarse_mesh[x_idx, y_idx, 1] = row['U:1']
        # U:2は全て0に置き換える
        coarse_mesh[x_idx, y_idx, 2] = 0.0

# 結果を保存するためにリスト形式に変換
output_data = []

for y_idx in range(coarse_mesh.shape[1]):
    for x_idx in range(coarse_mesh.shape[0]):
        x_center = x_bins[x_idx] + mesh_size / 2
        y_center = y_bins[y_idx] + mesh_size / 2
        u_values = coarse_mesh[x_idx, y_idx]
        output_data.append([x_center, y_center] + list(u_values))

# データフレームに変換
output_df = pd.DataFrame(output_data, columns=['x_center', 'y_center', 'U:0', 'U:1', 'U:2'])

# CSVにエクスポート
output_df.to_csv(output_path, index=False)

print("Coarse mesh data has been saved to coarse_mesh.csv.")

