import os
import glob
# —— 核心修复：解决 GDAL 字典路径警告 ——
os.environ['GDAL_DATA'] = r'D:\anaconda3\Library\share\gdal'
os.environ['PROJ_LIB'] = r'D:\anaconda3\Library\share\proj'

import pickle
import logging
import traceback
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path

# —— 日志配置 ——
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler('prediction_and_reclass.log', encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 类别定义
CLASS_NAMES = {
    0: "亮屋顶", 1: "火", 2: "自然噪声", 3: "过火地", 4: "人工噪声"
}

NAME_TO_ID = {v: k for k, v in CLASS_NAMES.items()}


def load_one_txt(fp):
    """加载单个TXT，处理文件读取错误"""
    cols = (
        ["ID", "lon", "lat", "img_id"] +
        [f"Band{i}" for i in range(1, 9)] +
        ["BandB8A"] +
        [f"Band{i}" for i in [9, 10, 11, 12]] +
        ["solar_zenithN", "solar_azimuthN"] +
        ["NDVIMax", "NDVIMin"] +
        ["R1Max", "R1Min", "R2Max", "R2Min", "R3Max", "R3Min"] +
        ["B1Max", "B1Min", "B2Max", "B2Min"] +
        ["W1Max", "W1Min", "BandSimMax", "BandSimMin"] +
        ["R4Max", "R4Min", "R5Max", "R5Min"] +
        ["B3Max", "B3Min", "B4Max", "B4Min"]
    )

    try:
        df = pd.read_csv(fp, sep=",", header=None, names=cols, on_bad_lines='skip')
    except pd.errors.EmptyDataError:
        logger.warning(f"⚠️ 文件为空: {os.path.basename(fp)}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"❌ 读取错误 {os.path.basename(fp)}: {e}")
        return pd.DataFrame()

    return df


def predict_one_file(fp, model):
    """预测单个TXT，自动处理NaN，返回带ID的输出df"""
    df_raw = load_one_txt(fp)
    if df_raw.empty:
        return None

    meta_cols = ["ID", "lon", "lat", "img_id", "solar_zenithN", "solar_azimuthN"]
    feature_cols = [c for c in df_raw.columns if c not in meta_cols]

    # NaN 处理
    if df_raw[feature_cols].isna().any().any():
        df_clean = df_raw.dropna(subset=feature_cols).copy()
        if df_clean.empty:
            logger.error(f"❌ {os.path.basename(fp)} 剔除 NaN 后无数据，跳过")
            return None
        df_raw = df_clean

    id_df = df_raw[meta_cols].copy()
    X_test = df_raw[feature_cols].copy()

    if hasattr(model, "feature_names_in_"):
        X_test = X_test[model.feature_names_in_]

    try:
        y_pred_encoded = model.predict(X_test)

        if hasattr(model, 'label_encoder_'):
            y_pred_encoded = y_pred_encoded.astype(int)
            y_pred = model.label_encoder_.inverse_transform(y_pred_encoded)
        else:
            y_pred = y_pred_encoded

        df_out = id_df[["ID", "lon", "lat", "img_id"]].copy()
        df_out["category"] = y_pred
        df_out = pd.concat([df_out.reset_index(drop=True), X_test.reset_index(drop=True)], axis=1)

        return df_out
    except Exception as e:
        logger.error(f"❌ 模型预测运行时出错: {e}")
        return None


def load_combined_shapefiles(shapefile_folder_path):
    """提前加载并合并所有面要素，统一为 WGS84 坐标系"""
    shapefile_folder = Path(shapefile_folder_path)
    shapefiles = list(shapefile_folder.glob('*.shp'))

    if not shapefiles:
        logger.warning(f"⚠️ 在 {shapefile_folder_path} 中未找到 shapefile 文件，重分类将被跳过。")
        return None

    logger.info(f"🗺️ 找到 {len(shapefiles)} 个 shapefile，正在加载合并...")
    gdf_list = []
    target_crs = "EPSG:4326"

    for shp_file in shapefiles:
        try:
            gdf = gpd.read_file(shp_file)
            if gdf.crs != target_crs:
                gdf = gdf.to_crs(target_crs)
            gdf_list.append(gdf)
        except Exception as e:
            logger.error(f"❌ 无法读取 {shp_file.name}: {e}")

    if not gdf_list:
        return None

    gdf_combined = pd.concat(gdf_list, ignore_index=True)
    logger.info(f"✅ 成功合并 {len(gdf_combined)} 个面要素，当前坐标系: {gdf_combined.crs}")
    return gdf_combined


def fast_reclassify(df, gdf_polygons):
    """使用 Geopandas 空间连接 (sjoin) 极速重分类 (优化版：传入的df预期仅为火点)"""
    result_df = df.copy()
    original_columns = result_df.columns.tolist()

    gdf_points = gpd.GeoDataFrame(
        result_df,
        geometry=gpd.points_from_xy(result_df['lon'], result_df['lat']),
        crs="EPSG:4326"
    )

    points_in_poly = gpd.sjoin(gdf_points, gdf_polygons, how="inner", predicate="within")
    intersected_indices = points_in_poly.index.unique()
    reclassify_count = len(intersected_indices)

    if reclassify_count > 0:
        original_categories = result_df.loc[intersected_indices, 'category']
        cat_counts = original_categories.value_counts().to_dict()

        # 将落在多边形内的点的 category 强制设为 0（变为亮屋顶/噪声，后续会被过滤掉）
        result_df.loc[intersected_indices, 'category'] = 0

        logger.info(f"🔄 空间掩膜过滤完成: 发现 {reclassify_count} 个伪火点落在多边形内。")
        for cat, count in cat_counts.items():
            cat_name = CLASS_NAMES.get(cat, cat)
            logger.info(f"   - 原本被识别为 [{cat_name}] 的 {count} 个点已被剔除 (重置为 0)")
    else:
        logger.info("🔄 空间掩膜过滤完成: 没有预测火点落在掩膜多边形内。")

    return result_df[original_columns]


def predict_and_reclassify_folder(test_folder, model_path, output_dir_raw, output_dir_reclass, shapefile_folder):
    """批量预测并重分类（仅输出火点版本）"""
    os.makedirs(output_dir_raw, exist_ok=True)
    os.makedirs(output_dir_reclass, exist_ok=True)

    logger.info(f"正在加载模型: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info("✅ 模型加载成功")

    gdf_polygons = load_combined_shapefiles(shapefile_folder)

    all_files = sorted(glob.glob(os.path.join(test_folder, "*.txt")))
    if not all_files:
        logger.error(f"❌ 在 {test_folder} 未找到任何 .txt 文件")
        return

    logger.info(f"📂 共找到 {len(all_files)} 个 TXT 文件，开始流水线处理 (仅提取火点)...")
    success_count = 0

    for fp in all_files:
        name = os.path.splitext(os.path.basename(fp))[0]
        out_csv_raw = os.path.join(output_dir_raw, f"{name}_predict_fire.csv")
        out_csv_reclass = os.path.join(output_dir_reclass, f"{name}_predict_reclass_fire.csv")

        logger.info("-" * 50)
        logger.info(f"▶ 开始处理: {name}")

        # --- 环节 A: 预测 ---
        try:
            df_out = predict_one_file(fp, model)
            if df_out is None or df_out.empty:
                continue
        except Exception as e:
            logger.error(f"❌ 预测失败 {os.path.basename(fp)}: {e}")
            continue

        # 🚀 核心过滤：只提取火点 (category == 1)
        df_fire_raw = df_out[df_out["category"] == 1].copy()

        if df_fire_raw.empty:
            logger.info(f"⚪ {name} 未预测出任何火点，直接跳过保存与重分类。")
            success_count += 1
            continue

        # 保存原始预测出的火点
        try:
            df_fire_raw.to_csv(out_csv_raw, index=False, encoding="utf-8-sig")
            logger.info(f"🔥 原始火点已提取并保存: {os.path.basename(out_csv_raw)} (共 {len(df_fire_raw)} 个)")
        except PermissionError:
            logger.error(f"⛔ 权限拒绝! 无法写入: {out_csv_raw}")
            continue

        # --- 环节 B: 重分类 / 掩膜剔除 ---
        if gdf_polygons is not None:
            try:
                # 🚀 性能优化：仅将"原始火点"传入 sjoin 计算，避免百万级无用背景像素参与空间相交
                df_reclassed_all = fast_reclassify(df_fire_raw, gdf_polygons)

                # 重分类后，部分误判火点的 category 被改为了 0，这里再次过滤只保留最终的火点
                df_fire_final = df_reclassed_all[df_reclassed_all["category"] == 1]

                if df_fire_final.empty:
                    logger.info(f"⚪ {name} 的火点经面要素掩膜过滤后已清零。")
                else:
                    df_fire_final.to_csv(out_csv_reclass, index=False, encoding="utf-8-sig")
                    logger.info(f"🔥 最终有效火点已保存: {os.path.basename(out_csv_reclass)} (共 {len(df_fire_final)} 个)")
            except Exception as e:
                logger.error(f"❌ 掩膜过滤计算或保存失败: {e}")
                continue
        else:
            logger.warning(f"⚠️ 由于没有加载到面要素，跳过 {name} 的掩膜剔除步骤。")

        success_count += 1

    logger.info("=" * 60)
    logger.info(f"🎉 全部流程处理完成! 成功: {success_count} / 总数: {len(all_files)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    # ———— 核心路径配置区域 ————

    test_folder = r"Folder containing data to be detected"
    model_path = r"Path to model file\20260316_1.pkl"
    shapefile_folder = r"Timeseries mask generated by 4_bright_roof_timeseries_mask_batch\2_final"
    output_dir_raw = r"Prediction results from original model"
    output_dir_reclass = r"Reclassified results after mask application"

    # ——————————————————————————

    predict_and_reclassify_folder(
        test_folder=test_folder,
        model_path=model_path,
        output_dir_raw=output_dir_raw,
        output_dir_reclass=output_dir_reclass,
        shapefile_folder=shapefile_folder
    )