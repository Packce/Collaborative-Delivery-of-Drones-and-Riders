import requests
import time
import pandas as pd
import math
from pyproj import Transformer

# ================== 配置 ==================
API_KEY = "3aa6dc413d2ec52b21a755a1a7007086"
RECT_POLYGON = "106.5001608,29.46046362|106.562383,29.51490807"
# POI_TYPES = [
#     ("170000", "公司企业"),
#     ("120000", "商务住宅"),
#     ("060100", "商场"),
#     ("060101", "购物中心"),
#     ("080000", "休闲服务场所"),
#     ("090100", "综合医院"),
#     ("090101", "综合医院"),
#     ("090102", "综合医院"),
#     ("090200", "专科医院"),
#     ("090201", "专科医院"),
#     ("090202", "专科医院"),
#     ("090203", "专科医院"),
#     ("090204", "专科医院"),
#     ("090205", "专科医院"),
#     ("090206", "专科医院"),
#     ("090207", "专科医院"),
#     ("090208", "专科医院"),
#     ("090209", "专科医院"),
#     ("090210", "专科医院"),
#     ("090211", "专科医院"),
#     ("090300", "诊所"),
#     ("090400", "急救中心"),
#     ("141201", "高等院校"),
#     ("141400", "培训机构"),
# ]
POI_TYPES = [
    ("050000", "餐饮相关场所"),   # 餐饮服务大类
    ("060000", "购物相关场所"),   # 购物服务大类
    ("090600", "医药保健相关"),   # 医疗保健服务下的医药类（如药店、医疗器械等）
]
# OUTPUT_FILE = "D:\文档\大学文件\竞赛\统计建模\真正的统模\代码和数据\数据\顾客数据.csv"
OUTPUT_FILE = "D:\文档\大学文件\竞赛\统计建模\真正的统模\代码和数据\数据\商家数据.csv"

# ================== 坐标转换函数 ==================
def gcj02_to_wgs84(lng, lat):
    # 输入校验
    if lng is None or lat is None:
        return None, None
    try:
        lng = float(lng)
        lat = float(lat)
    except (ValueError, TypeError):
        return None, None

    # 中国境外范围直接返回原值（不再加密）
    if not (72.004 <= lng <= 137.8347 and 0.8293 <= lat <= 55.8271):
        return lng, lat

    a = 6378245.0
    ee = 0.00669342162296594323

    def transform(lng, lat):
        dlat = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + 0.1 * lng * lat + 0.2 * math.sqrt(abs(lng))
        dlat += (20.0 * math.sin(6.0 * lng * math.pi) + 20.0 * math.sin(2.0 * lng * math.pi)) * 2.0 / 3.0
        dlat += (20.0 * math.sin(lat * math.pi) + 40.0 * math.sin(lat / 3.0 * math.pi)) * 2.0 / 3.0
        dlat += (160.0 * math.sin(lat / 12.0 * math.pi) + 320.0 * math.sin(lat * math.pi / 30.0)) * 2.0 / 3.0
        dlng = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + 0.1 * lng * lat + 0.1 * math.sqrt(abs(lng))
        dlng += (20.0 * math.sin(6.0 * lng * math.pi) + 20.0 * math.sin(2.0 * lng * math.pi)) * 2.0 / 3.0
        dlng += (20.0 * math.sin(lng * math.pi) + 40.0 * math.sin(lng / 3.0 * math.pi)) * 2.0 / 3.0
        dlng += (150.0 * math.sin(lng / 12.0 * math.pi) + 300.0 * math.sin(lng / 30.0 * math.pi)) * 2.0 / 3.0
        radlat = lat / 180.0 * math.pi
        magic = math.sin(radlat)
        magic = 1 - ee * magic * magic
        sqrtmagic = math.sqrt(magic)
        dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * math.pi)
        dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * math.pi)
        return dlng, dlat

    dlng, dlat = transform(lng, lat)
    wgs_lng = lng * 2 - (lng + dlng)
    wgs_lat = lat * 2 - (lat + dlat)
    return wgs_lng, wgs_lat

# 自定义投影定义
utm_modified_proj = (
    "+proj=tmerc "
    "+lat_0=0 "
    "+lon_0=107.14001820 "
    "+k=0.9996 "
    "+x_0=500000 "
    # "+lon_0=106.3740291753 "
    # "+k=0.9996340917 "
    # "+x_0=425779.1001 "
    "+y_0=0 "
    "+ellps=WGS84 "
    "+towgs84=0,0,0,0,0,0,0 "
    "+units=m "
    "+no_defs"
)
transformer = Transformer.from_crs("epsg:4326", utm_modified_proj, always_xy=True)

# ================== 爬取函数 ==================
def get_pois_by_polygon(api_key, polygon, poi_type, type_name):
    url = "https://restapi.amap.com/v3/place/polygon"
    all_pois = []
    page = 1
    print(f"开始爬取 [{type_name}] ...")
    while True:
        params = {
            "key": api_key,
            "polygon": polygon,
            "types": poi_type,
            "offset": 25,
            "page": page,
            "extensions": "all"
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            if data["status"] != "1":
                print(f"  API错误: {data['info']}")
                break
            pois = data.get("pois", [])
            if not pois:
                break
            all_pois.extend(pois)
            total = int(data.get("count", 0))
            print(f"  第{page}页: 获取{len(pois)}条，累计{len(all_pois)}条，总数{total}条")
            if len(all_pois) >= total:
                break
            page += 1
            time.sleep(0.2)
        except Exception as e:
            print(f"  请求异常: {e}")
            break
    print(f"  [{type_name}] 共获取 {len(all_pois)} 条数据")
    return all_pois

def process_poi_data(pois):
    processed = []
    for poi in pois:
        location = poi.get("location", "")
        # 初始化所有坐标变量
        gcj_lng = gcj_lat = wgs_lng = wgs_lat = proj_x = proj_y = None

        if location and ',' in location:
            try:
                gcj_lng, gcj_lat = map(float, location.split(","))
                # # 1. GCJ-02 -> WGS-84
                # wgs_lng, wgs_lat = gcj02_to_wgs84(gcj_lng, gcj_lat)
                # # 2. WGS-84 -> 自定义投影
                # if wgs_lng is not None and wgs_lat is not None:
                #     proj_x, proj_y = transformer.transform(wgs_lng, wgs_lat)
                proj_x, proj_y = transformer.transform(gcj_lng, gcj_lat)
            except Exception as e:
                print(f"坐标转换出错: {e}, location={location}")

        biz_ext = poi.get("biz_ext", {})
        item = {
            "id": poi.get("id", ""),
            "名称": poi.get("name", ""),
            "地址": poi.get("address", ""),
            "电话": poi.get("tel", ""),
            "经度_GCJ02": gcj_lng,
            "纬度_GCJ02": gcj_lat,
            "经度_WGS84": wgs_lng,
            "纬度_WGS84": wgs_lat,
            "Center_X": proj_x,
            "Center_Y": proj_y,
            "类型代码": poi.get("typecode", ""),
            "类型名称": poi.get("type", ""),
            "省份": poi.get("pname", ""),
            "城市": poi.get("cityname", ""),
            "区域": poi.get("adname", ""),
            "营业时间": biz_ext.get("opening_hours", ""),
            "评分": biz_ext.get("rating", "")
        }
        processed.append(item)
    return pd.DataFrame(processed)

# ================== 主程序 ==================
def main():
    all_raw_pois = []
    for poi_type, type_name in POI_TYPES:
        pois = get_pois_by_polygon(API_KEY, RECT_POLYGON, poi_type, type_name)
        all_raw_pois.extend(pois)
        time.sleep(0.5)

    # 去重
    unique_pois = {}
    for poi in all_raw_pois:
        poi_id = poi.get("id")
        if poi_id and poi_id not in unique_pois:
            unique_pois[poi_id] = poi

    print(f"\n总计获取原始数据 {len(all_raw_pois)} 条，去重后 {len(unique_pois)} 条")

    if unique_pois:
        df = process_poi_data(list(unique_pois.values()))
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print(f"数据已保存至 {OUTPUT_FILE}")
        print(f"\n前5条预览:\n{df.head()}")
    else:
        print("未获取到数据。")

if __name__ == "__main__":
    main()