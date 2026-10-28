import ee
import shapefile
import os
import time
import math
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

ee.Authenticate()
ee.Initialize(project='your gee project')


def retry_on_ee_error(max_retries=3, delay=2):
    """GEE错误重试装饰器"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except ee.EEException as e:
                    if attempt < max_retries - 1:
                        print(f"GEE错误，{delay}秒后重试 ({attempt + 1}/{max_retries}): {str(e)}")
                        time.sleep(delay * (attempt + 1))
                    else:
                        raise
            return None

        return wrapper

    return decorator


def safe_max(lst, default=0):
    """安全获取最大值"""
    return max(lst) if lst else default


def safe_min(lst, default=0):
    """安全获取最小值"""
    return min(lst) if lst else default


@retry_on_ee_error(max_retries=3, delay=2)
def extract_point_values(img_id, pnt):
    """提取影像在指定点的波段值和指数"""
    try:
        img = ee.Image(img_id)
        img_info = img.getInfo()
        if img_info is None:
            return None
    except Exception as e:
        print(f"无法加载影像 {img_id}: {str(e)}")
        return None

    # 获取中心点坐标 - 一次性获取
    coords = pnt.coordinates()
    lon_val = coords.get(0).getInfo()
    lat_val = coords.get(1).getInfo()

    # 定义16个采样点的偏移量（单位：米）
    offsets = [
        (-12, 12), (-4, 12), (4, 12), (12, 12),
        (-12, 4), (-4, 4), (4, 4), (12, 4),
        (-12, -4), (-4, -4), (4, -4), (12, -4),
        (-12, -12), (-4, -12), (4, -12), (12, -12)
    ]

    # 创建所有采样点的集合
    sample_points = []
    lat_radians = lat_val * math.pi / 180.0

    for offset_x, offset_y in offsets:
        lat_offset = offset_y / 111320.0
        lon_offset = offset_x / (111320.0 * math.cos(lat_radians))

        sample_lon = lon_val + lon_offset
        sample_lat = lat_val + lat_offset
        sample_points.append(ee.Geometry.Point([sample_lon, sample_lat]))

    # 创建FeatureCollection
    points_fc = ee.FeatureCollection(sample_points)

    try:
        # 一次性采样所有16个点的B2、B3、B4、B8、B11、B12波段
        sampled = img.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']).sampleRegions(
            collection=points_fc,
            scale=10,
            geometries=False
        ).getInfo()

        # 提取数据
        b2_values = []
        b3_values = []
        b4_values = []
        b8_values = []
        b11_values = []
        b12_values = []

        for feature in sampled['features']:
            props = feature['properties']
            # 检查是否所有波段都有值
            if all(k in props and props[k] is not None for k in ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']):
                b2_values.append(props['B2'])
                b3_values.append(props['B3'])
                b4_values.append(props['B4'])
                b8_values.append(props['B8'])
                b11_values.append(props['B11'])
                b12_values.append(props['B12'])

        if not b2_values:
            print(f"警告: 影像 {img_id} 所有采样点均无有效数据")
            return None

        # 在本地计算最大值
        b2_max = max(b2_values)
        b3_max = max(b3_values)
        b4_max = max(b4_values)
        b8_max = max(b8_values)

        # 在本地计算各指数
        #NDVI=((b8 - b4) / (b8 + b4))
        ndvi_values = []
        for b8, b4 in zip(b8_values, b4_values):
            denom = b8 + b4
            if denom != 0:
                ndvi_values.append((b8 - b4) / denom)

        #Mix Index 3=((b4 - b11) / (b4 + b11)) * (1 - (b8 - b4) / (b8 + b4))
        red1_values = []
        for b4, b11, b8 in zip(b4_values, b11_values, b8_values):
            denom1 = b4 + b11
            denom2 = b8 + b4
            if denom1 != 0 and denom2 != 0:
                red1 = ((b4 - b11) / denom1) * (1 - (b8 - b4) / denom2)
                red1_values.append(red1)

#       red Index 2=((b4 - b3) / b3)
        red2_values = []
        for b4, b3 in zip(b4_values, b3_values):
            if b3 != 0:
                red2_values.append((b4 - b3) / b3)

        #blue Index 3=b2 - b3
        red3_values = [b2 - b3 for b2, b3 in zip(b2_values, b3_values)]

        #blue1=(b2 * 2 - b3 - b4) / (b2 * 2 + b3 + b4)
        blue1_values = []
        for b2, b3, b4 in zip(b2_values, b3_values, b4_values):
            denom = b2 * 2 + b3 + b4
            if denom != 0:
                blue1 = (b2 * 2 - b3 - b4) / denom
                blue1_values.append(blue1)

        #blue2=((b2 - b3) / b3)
        blue2_values = []
        for b2, b3 in zip(b2_values, b3_values):
            if b3 != 0:
                blue2_values.append((b2 - b3) / b3)

        #White Index 1=b2 + b3 + b4
        white_values = [b2 + b3 + b4 for b2, b3, b4 in zip(b2_values, b3_values, b4_values)]

        # ================== 新增：10m 像元内三色一致性指数 ==================
        #Mix Index 2= max(b2, b3, b4) - min(b2, b3, b4)
        band_sim_values = []
        for b2, b3, b4 in zip(b2_values, b3_values, b4_values):
            band_sim = max(b2, b3, b4) - min(b2, b3, b4)
            band_sim_values.append(band_sim)

        band_sim_max = safe_max(band_sim_values)
        band_sim_min = safe_min(band_sim_values)
        # ====================================================================

        # ================== 新增指数 red4 / red5 / blue3 / blue4 ==================

        # Mix Index 4 = (b4 - b11) / (b4 + b11)
        red4_values = []
        for b4, b11 in zip(b4_values, b11_values):
            denom = b4 + b11
            if denom != 0:
                red4_values.append((b4 - b11) / denom)

        # Mix Index 1 = ((b12 - b4) / b4)
        red5_values = []
        for b12, b4 in zip(b12_values, b4_values):
            if b4 != 0:
                red5_values.append((b12 - b4) / b4)

        # Red Index 1 = (b12 - b2) / b2
        blue3_values = []
        for b12, b2 in zip(b12_values, b2_values):
            if b2 != 0:
                blue3_values.append((b12 - b2) / b2)

        # Red Index 3 = b12 / (b2 + b3 + b4)
        blue4_values = []
        for b12, b2, b3, b4 in zip(b12_values, b2_values, b3_values, b4_values):
            denom = b2 + b3 + b4
            if denom != 0:
                blue4_values.append(b12 / denom)

        # ====================================================================

        # 一次性采样其他所有波段（在中心点）
        # ⚠️ 定义你需要检查的所有波段
        other_band_names = ['B1', 'B5', 'B6', 'B7', 'B8A', 'B9', 'B10', 'B11', 'B12']

        other_bands_dict = img.select(other_band_names).reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=pnt,
            scale=20
        ).getInfo()

        # 🔥🔥🔥 【新增】检查中心点波段是否包含 None 或空值 🔥🔥🔥
        if not other_bands_dict:
            print(f"警告: 影像 {img_id} 中心点采样失败")
            return None

        for b in other_band_names:
            val = other_bands_dict.get(b)
            # 如果获取不到键，或者值为 None，说明该点无效
            if val is None:
                print(f"警告: 影像 {img_id} 中心点波段 {b} 为无效值 (None)，已跳过")
                return None

        # 获取太阳角度信息
        solar_zenithN = img.get('MEAN_SOLAR_ZENITH_ANGLE').getInfo()
        solar_azimuthN = img.get('MEAN_SOLAR_AZIMUTH_ANGLE').getInfo()

        if solar_zenithN is None or solar_azimuthN is None:
            print(f"警告: 影像 {img_id} 缺失太阳角度信息，已跳过")
            return None

        # 构建返回字符串 - 使用安全函数
        return (f"{other_bands_dict.get('B1')},{b2_max},{b3_max},{b4_max},"
                f"{other_bands_dict.get('B5')},{other_bands_dict.get('B6')},"
                f"{other_bands_dict.get('B7')},{b8_max},"
                f"{other_bands_dict.get('B8A')},{other_bands_dict.get('B9')},"
                f"{other_bands_dict.get('B10')},{other_bands_dict.get('B11')},"
                f"{other_bands_dict.get('B12')},"
                f"{solar_zenithN},{solar_azimuthN},"
                f"{safe_max(ndvi_values)},{safe_min(ndvi_values)},"
                f"{safe_max(red1_values)},{safe_min(red1_values)},"
                f"{safe_max(red2_values)},{safe_min(red2_values)},"
                f"{safe_max(red3_values)},{safe_min(red3_values)},"
                f"{safe_max(blue1_values)},{safe_min(blue1_values)},"
                f"{safe_max(blue2_values)},{safe_min(blue2_values)},"
                f"{safe_max(white_values)},{safe_min(white_values)},"
                f"{band_sim_max},{band_sim_min},"
                f"{safe_max(red4_values)},{safe_min(red4_values)},"
                f"{safe_max(red5_values)},{safe_min(red5_values)},"
                f"{safe_max(blue3_values)},{safe_min(blue3_values)},"
                f"{safe_max(blue4_values)},{safe_min(blue4_values)}")

    except Exception as e:
        print(f"采样过程出错: {str(e)}")
        return None


def process_shapefile(input_file, output_folder):
    """处理单个shapefile文件"""
    fnew = None
    try:
        base_filename = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_folder, base_filename + ".txt")

        fnew = open(output_file, "w", encoding='utf-8')

        with shapefile.Reader(input_file, encoding='utf-8') as sf:
            print(f'Processing {base_filename}: The shapefile contains {sf.numRecords} points.')

            fields = sf.fields[1:]
            field_names = [field[0] for field in fields]

            # 添加字段存在性检查
            required_fields = ['x', 'y', 'ID_1', 'time']
            for field in required_fields:
                if field not in field_names:
                    raise ValueError(f"缺少必需字段: {field}")

            X_index = field_names.index('x')
            Y_index = field_names.index('y')
            ID_index = field_names.index('ID_1')
            Time_index = field_names.index('time')

            for i in range(sf.numRecords):
                print(f'Point {i + 1}:')
                try:
                    rcd = sf.record(i)
                    X = rcd[X_index]
                    Y = rcd[Y_index]
                    ID = rcd[ID_index]
                    time_str = rcd[Time_index]

                    start_date = time_str
                    end_date = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

                    point = ee.Geometry.Point(X, Y)

                    s2 = ee.ImageCollection('COPERNICUS/S2') \
                        .filterDate(start_date, end_date) \
                        .filterBounds(point)

                    s2_size = s2.size().getInfo()
                    print(f'Point {i + 1} has {s2_size} images.')

                    if s2_size == 0:
                        print(f'跳过点 {i + 1}: 无可用影像')
                        continue

                    s2_id = [f['id'] for f in s2.getInfo().get('features', [])]

                    if s2_id:
                        j = s2_id[0]
                        b_value = extract_point_values(j, point)

                        if b_value is None:
                            print(f'跳过点 {i + 1}: 无有效影像数据')
                            continue

                        print(f'{j}: {b_value}')
                        fnew.write(f"{ID},{X},{Y},{j},{b_value}\n")
                        fnew.flush()

                except Exception as e:
                    print(f'处理点 {i + 1} 时出错: {str(e)}')
                    print(f'继续处理下一个点...')
                    continue

        fnew.close()
        print(f'Finished processing {base_filename}')

        time.sleep(1)

        # 删除shapefile文件
        shapefile_base = os.path.splitext(input_file)[0]
        extensions = ['.shp', '.shx', '.dbf', '.prj', '.cpg', '.sbn', '.sbx', '.xml', '.shp.xml']
        for ext in extensions:
            file_to_delete = f"{shapefile_base}{ext}"
            if os.path.exists(file_to_delete):
                try:
                    os.remove(file_to_delete)
                    print(f"Deleted: {file_to_delete}")
                except Exception as e:
                    print(f"Could not delete {file_to_delete}: {str(e)}")

        return True
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        return False
    finally:
        if fnew and not fnew.closed:
            fnew.close()


def main():
    """主函数"""
    input_folder = "Fire points output from 1_fire_detect_attribute.py"
    output_folder = "Active fire points with extracted features"


    os.makedirs(output_folder, exist_ok=True)

    shp_files = [f for f in os.listdir(input_folder) if f.endswith('.shp')]

    if not shp_files:
        print("未找到shapefile文件！")
        return

    print(f"找到 {len(shp_files)} 个shapefile文件")

    # 串行处理（适合少量文件或调试）
    # for shp_file in shp_files:
    #     input_path = os.path.join(input_folder, shp_file)
    #     process_shapefile(input_path, output_folder)

    # 并行处理（保持max_workers=3）
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for shp in shp_files:
            input_path = os.path.join(input_folder, shp)
            future = executor.submit(process_shapefile, input_path, output_folder)
            futures.append(future)

        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"处理文件时出错: {str(e)}")


if __name__ == "__main__":
    main()
