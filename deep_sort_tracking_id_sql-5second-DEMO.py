import argparse
import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (check_img_size, check_imshow, non_max_suppression,
                           scale_coords, xyxy2xywh, strip_optimizer, set_logging,
                           increment_path)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import defaultdict
import pyodbc
from datetime import datetime, timedelta
import requests
from PIL import Image, ImageDraw, ImageFont

# ----------------------------- 全域變數區 -----------------------------
location_id = None
spatial_density = 0
limit_the_number_of_people = 0
congestion_threshold = 0
near_congestion_threshold = 0
comfort_threshold = 0
normal_threshold = 0

# Line Notify 配置
access_token = 'OQftvj6sWK7m2n1BD0fAmMPV5YGyTJsmoysrxejwzbh'  # 請替換為您的存取權杖
dashboard_url = "http://127.0.0.1:3000/d/fe18lsbehefpce/..."

# SQL Server 連接設定
conn_str = (
    r'DRIVER={SQL Server};'
    r'SERVER=127.0.0.1;'
    r'DATABASE=researchdata;'
    r'UID=La208;'
    r'PWD=La208-2+'
)

# 建立資料庫連接
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

# 顏色設定
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}
pts = defaultdict(list)

# ----------------------------- 函數區 -----------------------------
def send_line_notify(message, token):
    """
    發送消息至 Line Notify
    :param message: 要發送的消息內容
    :param token: Line Notify 的存取權杖
    """
    url = "https://notify-api.line.me/api/notify"
    headers = {
        "Authorization": f"Bearer {token}"
    }
    data = {
        "message": message
    }
    try:
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()
        return response.status_code, response.text
    except requests.exceptions.RequestException as e:
        return None, str(e)

def get_location_info(location_id):
    query = """
    SELECT LocationName, Area
    FROM Location
    WHERE LocationID = ?
    """
    cursor.execute(query, (location_id,))
    result = cursor.fetchone()
    if result:
        return result.LocationName, result.Area
    else:
        return None, None

def create_person(detection_id):
    insert_query = """
    INSERT INTO Person (DetectionID)
    VALUES (?)
    """
    cursor.execute(insert_query, (detection_id,))
    conn.commit()
    cursor.execute("SELECT @@IDENTITY AS ID")
    return cursor.fetchone()[0]

def insert_stay_record(person_id, location_id, arrival_time, departure_time, duration):
    if person_id is None or location_id is None:
        print(f"警告：由於存在空值，跳過停留記錄插入。PersonID: {person_id}, LocationID: {location_id}")
        return
    query = """
    INSERT INTO StayRecord (PersonID, LocationID, ArrivalTime, DepartureTime, Duration)
    VALUES (?, ?, ?, ?, ?)
    """
    duration_str = format(duration, '.10f').rstrip('0').rstrip('.')
    cursor.execute(query, (person_id, location_id, arrival_time, departure_time, duration_str))
    conn.commit()

def insert_traffic_stats(location_id, start_time, end_time, person_count, average_duration, occupancy_rate):
    if location_id is None:
        print(f"警告：由於存在空值，跳過 TrafficStats 插入。LocationID: {location_id}")
        return
    query = """
    INSERT INTO TrafficStats (LocationID, StartTime, EndTime, PersonCount, AverageDuration, OccupancyRate)
    VALUES (?, ?, ?, ?, ?, ?)
    """
    average_duration_str = format(average_duration, '.10f').rstrip('0').rstrip('.')
    cursor.execute(query, (location_id, start_time, end_time, person_count, average_duration_str, occupancy_rate))
    conn.commit()

def insert_congestion_record(location_id, start_time, end_time, average_occupancy_rate):
    if location_id is None:
        print(f"警告：由於存在空值，跳過 CongestionRecord 插入。LocationID: {location_id}")
        return
    query = """
    INSERT INTO CongestionRecord (LocationID, StartTime, EndTime, AverageOccupancyRate)
    VALUES (?, ?, ?, ?)
    """
    cursor.execute(query, (location_id, start_time, end_time, average_occupancy_rate))
    conn.commit()

def insert_comfort_record(location_id, start_time, end_time, average_occupancy_rate):
    if location_id is None:
        print(f"警告：由於存在空值，跳過 ComfortRecord 插入。LocationID: {location_id}")
        return
    query = """
    INSERT INTO ComfortRecord (LocationID, StartTime, EndTime, AverageOccupancyRate)
    VALUES (?, ?, ?, ?)
    """
    cursor.execute(query, (location_id, start_time, end_time, average_occupancy_rate))
    conn.commit()

def compute_color_for_labels(label):
    """
    簡單函數，根據類別固定生成顏色。
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_boxes(img, bbox, names, identities=None, active_objects=None, offset=(0, 0)):
    """
    繪製檢測框、模糊框內內容，以及在每個框的右上角顯示 ID 和滯留時間。
    """
    height, width, _ = img.shape
    text_info = []  # 儲存每個框的文字及其位置

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # 確保座標在圖像範圍內
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(x2, img.shape[1])
        y2 = min(y2, img.shape[0])

        # 對人的邊界框內容進行模糊
        person_roi = img[y1:y2, x1:x2]
        person_roi = cv2.GaussianBlur(person_roi, (99, 99), 30)
        img[y1:y2, x1:x2] = person_roi

        # 繪製檢測框
        color = (255, 0, 0)
        if identities is not None:
            color = compute_color_for_labels(int(identities[i]))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 收集文字信息及其位置
        if identities is not None:
            identity = int(identities[i])
            if active_objects and identity in active_objects:
                arrival_time = active_objects[identity]['arrival_time']
                duration = datetime.now() - arrival_time
                duration_minutes = round(duration.total_seconds() / 60.0, 2)
            else:
                duration_minutes = 0.0
            text = f"ID: {identity}, 滯留: {duration_minutes} min"
            text_position = (x2, y1)  # 框的右上角
            text_info.append((text, text_position, color))

    # 使用 PIL 來繪製中文文字
    im_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(im_pil)

    font_path = "C:/Windows/Fonts/simsun.ttc"
    try:
        font = ImageFont.truetype(font_path, 20)
    except IOError:
        print(f"字體文件 {font_path} 不存在。請確保字體文件存在或更改 font_path 為有效路徑。")
        font = ImageFont.load_default()

    for text, position, color in text_info:
        text_bbox = font.getbbox(text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        x, y = position
        if x + text_width > width:
            x = width - text_width
        if y - text_height < 0:
            y = text_height
        draw.text((x, y - text_height), text, font=font, fill=(255, 255, 255))

    # 轉回 OpenCV 圖像
    img = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)
    return img

def detect(save_img=False):
    global location_id, spatial_density, limit_the_number_of_people
    global congestion_threshold, frame_count, fps, location_name

    source = opt.source
    weights = opt.weights
    view_img = opt.view_img
    save_txt = opt.save_txt
    imgsz = opt.img_size
    trace = not opt.no_trace

    save_img = not opt.nosave and not str(source).endswith('.txt')

    # 初始化一些變數
    person_count = 0

    # 目錄
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # 設置 FFMPEG 低延遲選項 (若使用本機攝影機，這不影響，可保留或移除)
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

    # 初始化
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'

    # 加載模型
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)

    # 若需要 trace model
    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()

    # 不做分類
    classify = False

    vid_path, vid_writer = None, None
    view_img = check_imshow()
    cudnn.benchmark = True

    # 如果 source 是 int，代表使用本機攝影機
    dataset = LoadStreams(source, img_size=imgsz, stride=stride)

    names = model.module.names if hasattr(model, 'module') else model.names

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()

    # ----------------- deep_sort 初始化 -----------------
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    deepsort = DeepSort(
        cfg_deep.DEEPSORT.REID_CKPT,
        max_dist=0.7,
        min_confidence=0.3,
        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=0.7,
        max_age=50,
        n_init=3,
        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
        use_cuda=(device.type != 'cpu')
    )

    frame_count = 0
    fps = 30

    # 原本設定為 5 秒
    last_insert_time_5s_datetime = datetime.now()
    interval_departed_objects = []
    active_objects = {}
    max_age = deepsort.tracker.max_age
    video_writers = {}
    last_congestion_status = None
    last_congestion_notification_time = datetime.min
    congestion_notification_interval = timedelta(minutes=5)
    overstay_duration_threshold = timedelta(minutes=30)

    # ----------------- 開始讀取影像流 -----------------
    for path, img, im0s, vid_cap in dataset:
        start_time = time.time()

        # 取得 FPS (若為本機攝影機，則可從 vid_cap 取得)
        if vid_cap:
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or fps is None:
                fps = 24
        else:
            fps = 24

        # 每幀重新初始化 person_count
        person_count = 0

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 預熱
        if device.type != 'cpu':
            if (old_img_b != img.shape[0] or 
                old_img_h != img.shape[2] or 
                old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for _ in range(3):
                    model(img, augment=opt.augment)[0]

        t1 = time_synchronized()
        with torch.no_grad():
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # 進行 NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, 
                                   classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        inference_time = t2 - t1
        nms_time = t3 - t2

        for i, det in enumerate(pred):
            p, s, im0, frame = dataset.sources[i], f'{i}: ', im0s[i].copy(), dataset.count
            p = Path(p)

            # 建立 VideoWriter
            if save_img:
                if p not in video_writers:
                    width = im0.shape[1]
                    height = im0.shape[0]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    output_path = save_dir / f"{p.stem}_{i}.mp4"
                    video_writers[p] = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                    print(f"初始化 VideoWriter，輸出路徑：{output_path}")

            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # 打印檢測結果
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # 更新 deep sort
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)

                current_ids = set()
                for track in deepsort.tracker.tracks:
                    if not track.is_confirmed():
                        continue
                    id_ = track.track_id
                    current_ids.add(id_)
                    if id_ not in active_objects:
                        person_id = create_person(int(id_))
                        arrival_time = datetime.now()
                        active_objects[id_] = {
                            'arrival_time': arrival_time,
                            'person_id': person_id,
                            'missing_frames': 0,
                            'notified_over_30': False
                        }
                    else:
                        active_objects[id_]['missing_frames'] = 0

                # 找出消失目標
                ids_to_remove = []
                for id_ in list(active_objects.keys()):
                    if id_ not in current_ids:
                        active_objects[id_]['missing_frames'] += 1
                        if active_objects[id_]['missing_frames'] >= max_age:
                            data = active_objects[id_]
                            departure_time = datetime.now()
                            duration_timedelta = departure_time - data['arrival_time']
                            duration_minutes = duration_timedelta.total_seconds() / 60.0
                            person_id = data['person_id']
                            arrival_time = data['arrival_time']

                            insert_stay_record(person_id, location_id, arrival_time, departure_time, duration_minutes)
                            interval_departed_objects.append({
                                'duration': duration_minutes,
                                'person_id': person_id
                            })
                            ids_to_remove.append(id_)

                for id_ in ids_to_remove:
                    del active_objects[id_]

                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, 4]
                    im0 = draw_boxes(im0, bbox_xyxy, names, identities, active_objects)

            # 更新 person_count
            person_count = len(active_objects)

            # 計算佔用率
            occupancy_rate = 0
            if spatial_density * limit_the_number_of_people != 0:
                occupancy_rate = (person_count / (spatial_density * limit_the_number_of_people)) * 100
            occupancy_rate = min(round(occupancy_rate, 2), 100)

            # 狀態與顏色判定
            if occupancy_rate >= 100:
                status = "壅塞"
                status_color = (0, 0, 255)
            elif occupancy_rate >= 80:
                status = "即將壅塞"
                status_color = (0, 165, 255)
            elif occupancy_rate >= 60:
                status = "正常"
                status_color = (0, 255, 0)
            else:
                status = "舒適"
                status_color = (255, 0, 0)

            end_time = time.time()
            total_time = end_time - start_time
            fps_display = 1 / total_time if total_time > 0 else 0

            print(f"Inference Time: {inference_time*1000:.1f} ms, "
                  f"NMS Time: {nms_time*1000:.1f} ms, "
                  f"FPS: {fps_display:.1f}")
            print(f"Person Count: {person_count}")

            # 使用 PIL 來繪製中文文字
            im_pil = Image.fromarray(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
            draw_pil = ImageDraw.Draw(im_pil)

            font_path = "C:/Windows/Fonts/simsun.ttc"
            try:
                font_status = ImageFont.truetype(font_path, 30)
                font_count = ImageFont.truetype(font_path, 30)
            except IOError:
                print(f"字體文件 {font_path} 不存在。")
                font_status = ImageFont.load_default()
                font_count = ImageFont.load_default()

            status_color_rgb = (status_color[2], status_color[1], status_color[0])
            white_color_rgb = (255, 255, 255)

            draw_pil.text((10, 10), f"狀態: {status}", font=font_status, fill=status_color_rgb)
            draw_pil.text((10, 50), f"人數: {person_count}", font=font_count, fill=white_color_rgb)

            im0 = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)

            if save_img and p in video_writers:
                video_writers[p].write(im0)

            # 每 5 秒寫入一次資料庫
            if (datetime.now() - last_insert_time_5s_datetime) >= timedelta(seconds=5):
                current_time = datetime.now()
                start_time_db = last_insert_time_5s_datetime
                end_time_db = current_time

                total_duration = sum(obj['duration'] for obj in interval_departed_objects)
                total_person_count = len(interval_departed_objects)
                if total_person_count > 0:
                    average_duration = total_duration / total_person_count
                else:
                    average_duration = 0

                # 重新計算最新的佔用率
                occupancy_rate = 0
                if spatial_density * limit_the_number_of_people != 0:
                    occupancy_rate = (person_count / (spatial_density * limit_the_number_of_people)) * 100
                occupancy_rate = min(round(occupancy_rate, 2), 100)

                # 寫入資料庫
                insert_traffic_stats(location_id, start_time_db, end_time_db, 
                                     person_count, average_duration, occupancy_rate)

                if occupancy_rate >= 100:
                    insert_congestion_record(location_id, start_time_db, end_time_db, occupancy_rate)
                if occupancy_rate < 60:
                    insert_comfort_record(location_id, start_time_db, end_time_db, occupancy_rate)

                interval_departed_objects.clear()
                last_insert_time_5s_datetime = current_time

            # 若達到壅塞或即將壅塞時，發送 Line 通知
            current_time = datetime.now()
            if status in ["即將壅塞", "壅塞"]:
                if (status != last_congestion_status) or (
                    current_time - last_congestion_notification_time > congestion_notification_interval
                ):
                    message = (
                        "\n\n"
                        f"⚠️ 警示通知：{location_name}現場人數達到{status}狀態！ ⚠️\n\n"
                        "📊 現況：\n"
                        f"- 當前人數：{person_count}人\n"
                        f"- 建議容納人數：{int(congestion_threshold)}人\n\n"
                        "🛠️ 請立即採取以下措施：(...簡化...)"
                        f"\n🔗 查看儀表板詳情：{dashboard_url}"
                    )
                    status_code, response_text = send_line_notify(message, access_token)
                    if status_code == 200:
                        print(f"已發送{status}狀態通知")
                        last_congestion_status = status
                        last_congestion_notification_time = current_time
                    else:
                        print(f"發送通知失敗: {response_text}")

            # 滯留超過30分鐘通知
            for id_ in active_objects:
                data = active_objects[id_]
                duration = current_time - data['arrival_time']
                if duration >= overstay_duration_threshold:
                    if not data.get('notified_over_30', False):
                        message = (
                            "\n\n"
                            f"⚠️ 滯留通知：ID為 {id_} 的人員已在 {location_name} 滯留超過30分鐘！ ⚠️\n\n"
                            "🛠️ 請立即採取以下措施：(...簡化...)"
                            f"\n🔗 查看儀表板詳情：{dashboard_url}"
                        )
                        status_code, response_text = send_line_notify(message, access_token)
                        if status_code == 200:
                            print(f"已發送ID {id_} 的滯留通知")
                            data['notified_over_30'] = True
                        else:
                            print(f"發送通知失敗: {response_text}")

        # 如果要顯示畫面（允許用滑鼠自由拖曳視窗大小）
        if view_img:
            window_name = f"Stream {i}"
            # 建立可縮放視窗
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            # 可以設定初始大小，若想讓其自動調整也可省略這行
            cv2.resizeWindow(window_name, 1540, 1280)
            # 直接顯示原影像
            cv2.imshow(window_name, im0)
            cv2.waitKey(1)

        frame_count += 1

    print(f'完成。 ({time.time() - t0:.3f} 秒)')

    # 釋放 VideoWriter
    if save_img:
        for writer in video_writers.values():
            writer.release()

# ----------------------------- 主程式入口 -----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='模型路徑')
    parser.add_argument('--source', type=str, default='0', help='來源，預設為0(本機攝影機)或可放檔案路徑、RTSP')
    parser.add_argument('--img-size', type=int, default=480, help='推論圖像大小（像素）')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='物體置信門檻')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS 的 IOU 門檻')
    parser.add_argument('--device', default='0', help='cuda 設備，例如 0 或 0,1,2,3 或 cpu')
    parser.add_argument('--view-img', action='store_true', help='顯示結果')
    parser.add_argument('--save-txt', action='store_true', help='將結果保存為 *.txt')
    parser.add_argument('--save-conf', action='store_true', help='保存置信度於 --save-txt 標籤')
    parser.add_argument('--nosave', action='store_true', help='不保存圖像/視頻')
    parser.add_argument('--classes', nargs='+', type=int, default=[0], help='按類別篩選：--class 0, 或 --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='類別無關的 NMS')
    parser.add_argument('--augment', action='store_true', help='增強推論')
    parser.add_argument('--update', action='store_true', help='更新所有模型')
    parser.add_argument('--project', default='runs/detect', help='保存結果至項目/名稱')
    parser.add_argument('--name', default='exp', help='保存結果至項目/名稱')
    parser.add_argument('--exist-ok', action='store_true', help='允許已有的項目/名稱，不遞增')
    parser.add_argument('--no-trace', action='store_true', help='不追蹤模型')
    opt = parser.parse_args()
    print(opt)

    # 讓使用者選擇位置 (假設還需要做位置選擇邏輯)
    print("請選擇一個位置：")
    print("-----------")
    print("1. 醫院大廳左半部")
    print("2. 醫院大廳右半部")
    print("3. 醫院咖啡廳 A 區")
    print("4. Mlab-Test-1")
    print("5. Mlab-Test-2")
    print("-----------")

    while True:
        try:
            location_choice = int(input("請輸入位置編號 (1-5): "))
            if 1 <= location_choice <= 5:
                location_id = location_choice
                break
            else:
                print("請輸入有效的位置編號 (1-5)。")
        except ValueError:
            print("請輸入有效的數字。")

    location_name, spatial_density = get_location_info(location_id)
    print(f"已選擇位置：{location_name}，空間大小：{spatial_density} 平方公尺")

    print("-----------")

    # 是否變更每平方米容納人數參數
    question = str(input("請問是否需要變更每平方米容納人數參數(需要變更請輸入1，無須變更請輸入0):"))
    while True:
        if question == '1':
            while True:
                try:
                    limit_the_number_of_people = int(input("請輸入每平方米容納人數(默認值為:3):"))
                    if limit_the_number_of_people > 0:
                        break
                    else:
                        print("請勿輸入無效參數!!")
                except ValueError:
                    print("您輸入的不是有效的數字，請重新輸入。")
            break
        elif question == '0':
            limit_the_number_of_people = 3
            break
        else:
            print("請輸入正確數值")
            question = str(input("請問是否需要變更每平方米容納人數參數(需要變更請輸入1，無須變更請輸入0):"))

    congestion_threshold = spatial_density * limit_the_number_of_people
    comfort_threshold = round(congestion_threshold * 0.4, 3)
    normal_threshold = round(congestion_threshold * 0.6, 3)
    near_congestion_threshold = round(congestion_threshold * 0.8, 3)

    print(f"人流壅塞界限: {congestion_threshold}")
    print(f"即將壅塞界限: {near_congestion_threshold}")
    print(f"人流舒適界限: {comfort_threshold}")
    print(f"人流正常界限: {normal_threshold}")
    print("-----------")

    with torch.no_grad():
        if opt.update:
            for w in ['yolov7.pt']:
                detect()
                strip_optimizer(w)
        else:
            detect()
