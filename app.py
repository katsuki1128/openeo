# app.py
from flask import Flask, render_template, request, send_file
import openeo
import xarray
import rioxarray
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tqdm import tqdm
import earthpy.plot as ep

# 非インタラクティブモードに設定
import matplotlib

matplotlib.use("Agg")

app = Flask(__name__)

# ======================
# CDESのバックエンドと接続とOIDC認証
# ======================
connection = openeo.connect("openeo.dataspace.copernicus.eu")
connection.authenticate_oidc()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    west = float(request.form["west"])
    south = float(request.form["south"])
    east = float(request.form["east"])
    north = float(request.form["north"])
    image_type = request.form["image_type"]

    start_date = request.form["start_date"]
    end_date = request.form["end_date"]

    center_lat = float(request.form["center_lat"])
    center_lng = float(request.form["center_lng"])
    zoom_level = int(request.form["zoom_level"])

    # 取得するバンドのリスト
    bands = [
        "B04",  # Red
        "B03",  # Green
        "B02",  # Blue
        "B08",  # NIR
        "B11",  # SWIR (for NDWI)
        "SCL",  # Scene Classification Layer
    ]

    # datacubeの設定
    datacube = connection.load_collection(
        "SENTINEL2_L2A",
        spatial_extent={
            "west": west,
            "south": south,
            "east": east,
            "north": north,
            "crs": "EPSG:4326",
        },
        # temporal_extent=["2024-03-01", "2024-03-31"],
        temporal_extent=[start_date, end_date],
        bands=bands,
        # bands=["B04", "B03", "B02", "SCL"],
        max_cloud_cover=20,
    )

    # 現在の日時を取得してファイル名を生成
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"s2_{current_time}.nc"

    # ファイルの存在を確認して、存在しない場合のみダウンロード
    if not os.path.exists(filename):
        print(f"{filename} が存在しません。ダウンロードを開始します。")
        # download_with_progress(datacube, filename)
        datacube.download(filename)
    else:
        print(f"{filename} はすでに存在します。")

    # 保存したncデータをxarrayとして読み込む
    ds = xarray.open_dataset(filename)

    # t=0の日付を取得
    t0_date = str(ds.coords["t"].values[0])

    # 画像データの選択とファイル名の設定
    if image_type == "rgb":
        data = ds[["B04", "B03", "B02"]].isel(t=0).to_array(dim="bands")
        output_image = f"static/rgb_{current_time}.png"
    elif image_type == "cir":
        data = ds[["B08", "B04", "B03"]].isel(t=0).to_array(dim="bands")
        output_image = f"static/cir_{current_time}.png"
    elif image_type == "ndvi":
        nir = ds["B08"].isel(t=0)
        red = ds["B04"].isel(t=0)
        data = (nir - red) / (nir + red)
        output_image = f"static/ndvi_{current_time}.png"
    elif image_type == "ndwi":
        nir = ds["B08"].isel(t=0)
        swir = ds["B11"].isel(t=0)
        data = (nir - swir) / (nir + swir)
        output_image = f"static/ndwi_{current_time}.png"

    # プロットの作成
    fig, ax = plt.subplots(figsize=(10, 10))

    if image_type in ["ndvi", "ndwi"]:
        cmap = "RdYlGn" if image_type == "ndvi" else "Blues"
        im = ax.imshow(data.values, cmap=cmap, vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax)
    else:
        ep.plot_rgb(
            data.values,
            rgb=(0, 1, 2),
            ax=ax,
            stretch=True,
            str_clip=0.02,
            title=f"{image_type.upper()} Image with Stretch Applied",
        )

    plt.axis("off")
    plt.savefig(output_image, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # 使用した.ncファイルを削除
    os.remove(filename)

    # 画像のパスと地理的境界を返す
    return render_template(
        "index.html",
        image_url=output_image,
        west=west,
        south=south,
        east=east,
        north=north,
        center_lat=center_lat,
        center_lng=center_lng,
        zoom_level=zoom_level,
        t0_date=t0_date,
    )


if __name__ == "__main__":
    app.run(debug=True)
