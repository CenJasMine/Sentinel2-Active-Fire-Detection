import os
import glob
# 强制指定 Anaconda 环境下的 GDAL 和 PROJ 字典路径
os.environ['GDAL_DATA'] = r'D:\anaconda3\Library\share\gdal'
os.environ['PROJ_LIB'] = r'D:\anaconda3\Library\share\proj'

import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
from scipy.ndimage import maximum_filter


# =====================================================================
# 步骤 1：栅格极速扩散与矢量化函数
# =====================================================================
def raster_to_individual_polygons(raster_path, output_shp, value_threshold=1):
    """
    将栅格转换为独立的多边形，并在转换前将符合条件的像元向外扩散一圈
    """
    if not os.path.exists(raster_path):
        print(f"  ✗ 文件不存在: {raster_path}")
        return None

    output_dir = os.path.dirname(output_shp)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with rasterio.open(raster_path) as src:
            raster_data = src.read(1)
            transform = src.transform
            crs = src.crs
            nodata = src.nodata

        # --- 形态学扩散 ---
        max_neighbors = maximum_filter(raster_data, size=3, mode='constant', cval=0)
        mask_to_expand = (raster_data < 3) & (max_neighbors >= 3)
        expanded_count = np.sum(mask_to_expand)
        raster_data[mask_to_expand] = max_neighbors[mask_to_expand]
        print(f"  ✓ 扩散完成：共 {expanded_count} 个边缘像元被同化。")

        # --- 矢量化 ---
        pixel_width = abs(transform[0])
        pixel_height = abs(transform[4])
        left = transform[2]
        top = transform[5]

        features = []
        count = 0
        rows, cols = raster_data.shape

        for row in range(rows):
            for col in range(cols):
                value = raster_data[row, col]
                if nodata is not None and value == nodata:
                    continue
                if value < value_threshold:
                    continue

                x_min = left + col * pixel_width
                x_max = x_min + pixel_width
                y_max = top - row * pixel_height
                y_min = y_max - pixel_height

                polygon = box(x_min, y_min, x_max, y_max)
                features.append({'geometry': polygon, 'gridcode': float(value)})
                count += 1

        if features:
            gdf = gpd.GeoDataFrame(features, crs=crs)
            gdf.to_file(output_shp)
            print(f"  ✓ 步骤1临时文件已生成 ({count} 个多边形)")
            return gdf
        else:
            print(f"  ✗ 没有符合条件的像元（值 >= {value_threshold}）")
            return None

    except Exception as e:
        print(f"  ✗ 步骤1处理出错: {str(e)}")
        return None


# =====================================================================
# 步骤 2：相邻多边形融合与低频过滤函数
# =====================================================================
def dissolve_adjacent_polygons_with_max(input_shp, output_shp, value_field='gridcode', min_keep_value=5):
    """
    合并相邻的多边形，保留最大值，并直接剔除最大值低于阈值的区域
    """
    if not os.path.exists(input_shp):
        print(f"  ✗ 找不到输入文件 {input_shp}！")
        return None

    gdf = gpd.read_file(input_shp)
    sindex = gdf.sindex
    visited = set()
    regions = []

    # 识别相邻区域
    for idx in gdf.index:
        if idx in visited:
            continue

        current_region = [idx]
        stack = [idx]
        visited.add(idx)

        while stack:
            current_idx = stack.pop()
            current_geom = gdf.loc[current_idx, 'geometry']
            possible_matches_idx = list(sindex.intersection(current_geom.bounds))

            for neighbor_idx in possible_matches_idx:
                if neighbor_idx not in visited:
                    neighbor_geom = gdf.loc[neighbor_idx, 'geometry']
                    if current_geom.intersects(neighbor_geom):
                        visited.add(neighbor_idx)
                        current_region.append(neighbor_idx)
                        stack.append(neighbor_idx)

        regions.append(current_region)

    dissolved_features = []
    filtered_out_count = 0

    # 合并与过滤
    for region_id, indices in enumerate(regions):
        region_gdf = gdf.loc[indices]
        max_value = region_gdf[value_field].max()

        # 核心过滤：低于阈值的连片区域直接丢弃
        if max_value < min_keep_value:
            filtered_out_count += 1
            continue

        merged_geom = unary_union(region_gdf.geometry.tolist())
        dissolved_features.append({
            'geometry': merged_geom,
            value_field: float(max_value)
        })

    if len(dissolved_features) == 0:
        print(f"  ✗ 警告：所有区域最高频次均 < {min_keep_value}，无最终输出！")
        return None

    output_dir = os.path.dirname(output_shp)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dissolved_gdf = gpd.GeoDataFrame(dissolved_features, crs=gdf.crs)
    dissolved_gdf.to_file(output_shp)

    print(f"  ✓ 步骤2融合完毕！剔除噪点斑块: {filtered_out_count} 个，最终保留: {len(dissolved_gdf)} 个。")
    return dissolved_gdf


# =====================================================================
# 主程序：全文件夹自动批量调度
# =====================================================================
if __name__ == "__main__":

    # ------------------ ❗你需要修改的参数区 ------------------
    # 1. 文件夹路径设置（程序会自动读取 input_folder 下所有的 .tif 文件）
    input_folder = r"Folder containing original TIF files "
    temp_folder = r"Folder for temporary SHP files from Step 1 "
    final_folder = r"Folder for final cleaned SHP files"

    # 2. 核心阈值设置
    threshold_step1 = 3  # Step 1: Only extract pixels >= 3
    threshold_step2 = 5  # Step 2: Filter out connected regions with maximum value < 5 after dissolving
    # --------------------------------------------------------

    print("=" * 60)
    print("🚀 启动遥感影像批量形态学提取流水线 🚀")
    print("=" * 60)

    # 自动获取所有 TIF 文件
    tif_files = glob.glob(os.path.join(input_folder, "*.tif"))
    total_files = len(tif_files)

    if total_files == 0:
        print(f"\n❌ 在 [{input_folder}] 中没有找到任何 .tif 文件，请检查路径！")
    else:
        print(f"🔍 成功扫描到 {total_files} 个 TIF 文件，准备开工...\n")

    # 开始循环处理
    for index, tif_path in enumerate(tif_files, start=1):
        base_name = os.path.splitext(os.path.basename(tif_path))[0]

        temp_shapefile = os.path.join(temp_folder, f"{base_name}_temp.shp")
        final_shapefile = os.path.join(final_folder, f"{base_name}_final.shp")

        print("-" * 60)
        print(f"▶ 正在处理 [{index}/{total_files}]: {base_name}.tif")

        try:
            # 执行第一步
            step1_result = raster_to_individual_polygons(
                raster_path=tif_path,
                output_shp=temp_shapefile,
                value_threshold=threshold_step1
            )

            if step1_result is not None:
                # 执行第二步
                step2_result = dissolve_adjacent_polygons_with_max(
                    input_shp=temp_shapefile,
                    output_shp=final_shapefile,
                    value_field='gridcode',
                    min_keep_value=threshold_step2
                )

                if step2_result is not None:
                    print(f"✅ [{base_name}] 处理成功！")

                    # (可选功能：自动清理临时文件)
                    # 如果你想让硬盘保持整洁，可以取消下面这几行的注释
                    """
                    try:
                        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                            temp_file_to_del = temp_shapefile.replace('.shp', ext)
                            if os.path.exists(temp_file_to_del):
                                os.remove(temp_file_to_del)
                    except Exception as e:
                        pass
                    """
                else:
                    print(f"⚠️ [{base_name}] 步骤2无有效输出。")
            else:
                print(f"⚠️ [{base_name}] 步骤1无有效输出。")

        except Exception as e:
            print(f"❌ 处理 [{base_name}] 时发生异常崩溃: {e}")
            continue  # 报错不要紧，跳过这个文件继续跑下一个

    print("\n" + "=" * 60)
    print(f"🎉 批量任务圆满结束！共遍历了 {total_files} 个文件。")
    print(f"📂 最终成品请前往文件夹查看: {final_folder}")
    print("=" * 60)