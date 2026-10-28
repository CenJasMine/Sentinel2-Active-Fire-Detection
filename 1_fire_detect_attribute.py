# -*- coding: cp936 -*-
import os
import fnmatch
import ee
import urllib.request
import time
import geopandas as gpd
from datetime import datetime

# 设置GDAL数据文件路径
os.environ['GDAL_DATA'] = 'D:\\anaconda3\\Library\\share\\gdal'

ee.Initialize(project='your gee project')

count = 1
while (count < 100):
    print('第', count, '次执行程序')
    inputdir = "Input content: ImageID containing image information, e.g., 20190819T022559_20190819T024405_T49MHT_0_36"
    # inputdir = "D:\\11\\MSI_txt\\"
    # outputdir = "F:\\CJM\\straw_burning\\sample_new\\案例\\重庆2022山火\\Point_New\\2022_48RXU\\"
    outputdir = "Output content: Initially detected fire points"
    pattern = '*.txt'
    filenames_all = fnmatch.filter(os.listdir(inputdir), pattern)

    for filename in filenames_all:
        IDD = 'COPERNICUS/S2/' + filename.split('_')[0] + '_' + filename.split('_')[1] + '_' + filename.split('_')[2]
        fileseg = filename.split('.')
        outName = fileseg[0]
        print(outName)
        try:
            # 获取影像信息
            img = ee.Image(IDD)
            img_info = img.getInfo()

            # 提取日期信息
            date_str = img_info['properties']['DATATAKE_IDENTIFIER'].split('_')[1][:8]  # 提取日期部分 (YYYYMMDD)

            # 获取年份和格式化日期
            year = date_str[:4]
            month = date_str[4:6]
            day = date_str[6:8]

            # 格式化日期为需要的格式 (YYYY-MM-DD)
            full_date = f"{year}-{month}-{day}"

            b1 = img.select('B1').divide(10000)
            b2 = img.select('B2').divide(10000)
            b3 = img.select('B3').divide(10000)
            b4 = img.select('B4').divide(10000)
            b5 = img.select('B5').divide(10000)
            b6 = img.select('B6').divide(10000)
            b7 = img.select('B7').divide(10000)
            b8 = img.select('B8').divide(10000)
            b8a = img.select('B8A').divide(10000)
            b9 = img.select('B9').divide(10000)
            b10 = img.select('B10').divide(10000)
            b11 = img.select('B11').divide(10000)
            b12 = img.select('B12').divide(10000)

#我的点检测方式
            a1 = ((b12.subtract(b11)).divide(b8a)).gte(0.15)   #TAI阈值

            a2 = b12.gt(0.22)
            a3 = (b12.subtract(b11)).gt(0.04)

            a4 = b11.gt(0.01)
            a5 = b8a.gt(0.01)

            a6 = (b8a.subtract(b7)).lt(0.05)
            a7 = (b12.subtract(b8a)).gt(0.04)
            a8 = (b12.subtract(b1)).gt(0.1)

            ab = a1.add(a2).add(a3).add(a4).add(a5).add(a6).add(a7).add(a8)

            c = b12.lte(b11)
            d = b12.gt(1)
            dd = b8a.lt(0.65)
            cd = c.add(d).add(dd)
            # cd=c.add(d)

            e = b12.gte(b11)
            f = (b11.divide(b8a)).gt(2)
            g = b12.gt(1)
            h = b11.gt(1)
            efgh = e.add(f).add(g).add(h)

            index=(ab.eq(8)).add(cd.eq(3)).add(efgh.eq(4))
            result = index.gte(1)
#结束

            # 获取投影信息
            crs1 = result.projection().crs().getInfo()
            scale1 = result.projection().nominalScale().getInfo()

            # 创建一个网格点图像，每个像元中心一个点
            coords = ee.Image.pixelLonLat()

            # 将满足条件的像元提取为点
            points = coords.updateMask(result).sample(
                region=img.geometry(),
                scale=scale1,  # 使用原始分辨率
                projection=result.projection(),  # 确保使用相同的投影
                geometries=True,  # 返回几何对象
                factor=1  # 不进行降采样
            )

            # 下载为GeoJSON
            geojson_file = outputdir + outName + '_points.geojson'
            points_link = points.getDownloadUrl(filetype='geojson')
            urllib.request.urlretrieve(points_link, geojson_file)

            # 转换为Shapefile并添加属性
            try:
                # 转换点数据
                points_gdf = gpd.read_file(geojson_file)
                if not points_gdf.empty:
                    # 确保是GeoDataFrame类型
                    if not isinstance(points_gdf, gpd.GeoDataFrame):
                        points_gdf = gpd.GeoDataFrame(points_gdf, geometry=points_gdf.geometry)

                    # 重命名经纬度列
                    if 'longitude' in points_gdf.columns:
                        points_gdf = points_gdf.rename(columns={'longitude': 'x', 'latitude': 'y'})

                    # 如果没有经纬度列，从几何对象中提取
                    if 'x' not in points_gdf.columns:
                        points_gdf['x'] = points_gdf.geometry.x  # 经度
                        points_gdf['y'] = points_gdf.geometry.y  # 纬度

                    # 创建ID列 (从10001开始的连续整数)
                    points_gdf['ID_1'] = [f"1000{i+1:04d}" for i in range(len(points_gdf))]

                    # 添加时间字段
                    points_gdf['time'] = full_date

                    # 添加空的type字段
                    points_gdf['type'] = None

                    # 重排列顺序，使其符合标准顺序: ID, x, y, time, type
                    geometry = points_gdf.geometry
                    points_gdf = points_gdf.drop(columns='geometry')

                    # 重新排序列 - 先x后y
                    desired_columns = ['ID_1', 'x', 'y', 'time', 'type']
                    existing_columns = [col for col in desired_columns if col in points_gdf.columns]
                    other_columns = [col for col in points_gdf.columns if col not in desired_columns]
                    points_gdf = points_gdf[existing_columns + other_columns]

                    # 重新添加几何列
                    points_gdf = gpd.GeoDataFrame(points_gdf, geometry=geometry)

                    # 保存点Shapefile
                    point_shapefile = outputdir + outName + '_points.shp'
                    points_gdf.to_file(point_shapefile, driver='ESRI Shapefile')
                    print(f"已成功保存点Shapefile: {outName}_points.shp")
                    print(f"共检测到 {len(points_gdf)} 个点")
                else:
                    print(f"没有检测到燃烧区域点: {outName}")

            except Exception as e:
                print(f"转换Shapefile时出错: {e}")

        except Exception as e:
            print(f'处理出错: {str(e)}')
            time.sleep(3)
        else:
            # 处理成功后删除原始txt文件
            downfile = inputdir + filename
            os.remove(downfile)

    count = count + 1
