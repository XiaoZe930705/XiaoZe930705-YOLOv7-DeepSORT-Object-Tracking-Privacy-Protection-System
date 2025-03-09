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
from utils.general import (check_img_size, check_requirements, check_imshow, non_max_suppression,
                           apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging,
                           increment_path, check_file)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque, defaultdict
import pyodbc
from datetime import datetime, timedelta
import requests
from PIL import Image, ImageDraw, ImageFont  # 新增

# 全域變數
location_id = None
spatial_density = 0
limit_the_number_of_people = 0
congestion_threshold = 0
near_congestion_threshold = 0
comfort_threshold = 0
normal_threshold = 0

# Line Notify 配置
access_token = 'OQftvj6sWK7m2n1BD0fAmMPV5YGyTJsmoysrxejwzbh'  # 請替換為您的存取權杖
dashboard_url = "http://127.0.0.1:3000/d/fe18lsbehefpce/18d6023f-901b-59bd-aff1-a26fd9ceed32?orgId=1&showCategory=Panel+options&from=1732941598588&to=1732942198588"

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

def xyxy_to_xywh(*xyxy):
    """計算相對於像素值的邊界框。"""
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

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

    for key in list(data_deque):
        if identities is not None and key not in identities:
            data_deque.pop(key)

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
        color = compute_color_for_labels(int(identities[i])) if identities is not None else (255, 0, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 收集文字信息及其位置
        if identities is not None:
            identity = int(identities[i])
            if identity in active_objects:
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

    # 設定字體（請確保字體文件存在或替換為您系統中的中文字體）
    font_path = "C:/Windows/Fonts/simsun.ttc"  # 您可以根據需要更改此路徑
    try:
        font = ImageFont.truetype(font_path, 20)
    except IOError:
        print(f"字體文件 {font_path} 不存在。請確保字體文件存在或更改 font_path 為有效路徑。")
        font = ImageFont.load_default()

    for text, position, color in text_info:
        # 調整文字位置，避免超出圖像範圍
        text_bbox = font.getbbox(text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        x, y = position
        if x + text_width > width:
            x = width - text_width
        if y - text_height < 0:
            y = text_height
        draw.text((x, y - text_height), text, font=font, fill=(255, 255, 255))  # 白色文字

    # 轉回 OpenCV 圖像
    img = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)

    return img

def detect(save_img=False):
    global location_id, spatial_density, limit_the_number_of_people, congestion_threshold, frame_count, fps, location_name

    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace

    save_img = not opt.nosave and not source.endswith('.txt')

    # 初始化一些變數
    person_count = 0

    # 目錄
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # 設置 FFMPEG 低延遲選項
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

    # 初始化
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'

    # 加載模型
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()

    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)
        modelc.load_state_dict(
            torch.load('weights/resnet101.pt', map_location=device, weights_only=True)['model']
        ).to(device).eval()


    vid_path, vid_writer = None, None
    view_img = check_imshow()
    cudnn.benchmark = True

    # 修改 LoadStreams 以設置緩衝區大小
    dataset = LoadStreams(source, img_size=imgsz, stride=stride)

    names = model.module.names if hasattr(model, 'module') else model.names

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()

    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                       max_dist=0.7,
                       min_confidence=0.3,
                       nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                       max_iou_distance=0.7,
                       max_age=50,
                       n_init=3,
                       nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                       use_cuda=True)

    frame_count = 0
    fps = 30

    last_insert_time_60s_datetime = datetime.now()
    interval_departed_objects = []

    active_objects = {}
    max_age = deepsort.tracker.max_age

    video_writers = {}

    last_congestion_status = None
    last_congestion_notification_time = datetime.min
    congestion_notification_interval = timedelta(minutes=5)
    overstay_duration_threshold = timedelta(minutes=30)

    for path, img, im0s, vid_cap in dataset:
        start_time = time.time()  # 開始時間

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

        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        t1 = time_synchronized()
        with torch.no_grad():
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        inference_time = t2 - t1
        nms_time = t3 - t2

        for i, det in enumerate(pred):
            p, s, im0, frame = dataset.sources[i], '%g: ' % i, im0s[i].copy(), dataset.count

            p = Path(p)

            if save_img:
                if p not in video_writers:
                    width = im0.shape[1]
                    height = im0.shape[0]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    output_path = save_dir / f"{p.stem}_{i}.mp4"
                    video_writers[p] = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                    print(f"初始化 VideoWriter，輸出路徑：{output_path}")

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)

                current_ids = set()
                for track in deepsort.tracker.tracks:
                    if not track.is_confirmed():
                        continue
                    id = track.track_id
                    current_ids.add(id)
                    if id not in active_objects:
                        person_id = create_person(int(id))
                        arrival_time = datetime.now()
                        active_objects[id] = {
                            'arrival_time': arrival_time,
                            'person_id': person_id,
                            'missing_frames': 0,
                            'notified_over_30': False
                        }
                    else:
                        active_objects[id]['missing_frames'] = 0

                ids_to_remove = []
                for id in list(active_objects.keys()):
                    if id not in current_ids:
                        active_objects[id]['missing_frames'] += 1
                        if active_objects[id]['missing_frames'] >= max_age:
                            data = active_objects[id]
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
                            ids_to_remove.append(id)
                for id in ids_to_remove:
                    del active_objects[id]

                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, 4]
                    # Debug: print identities
                    print(f"Detected identities: {identities}")
                    im0 = draw_boxes(im0, bbox_xyxy, names, identities, active_objects)

            # 更新 person_count
            person_count = len(active_objects)

            # 計算佔用率
            occupancy_rate = (person_count / (spatial_density * limit_the_number_of_people)) * 100 if spatial_density * limit_the_number_of_people != 0 else 0
            occupancy_rate = min(round(occupancy_rate, 2), 100)

            # 確定狀態和顏色
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

            # 在左上角顯示狀態和人數，字體稍大
            end_time = time.time()
            total_time = end_time - start_time
            fps_display = 1 / total_time if total_time > 0 else 0

            # 移除畫面上的推論時間和 FPS 顯示，改為後台輸出
            print(f"Inference Time: {inference_time*1000:.1f} ms, NMS Time: {nms_time*1000:.1f} ms, FPS: {fps_display:.1f}")
            print(f"Person Count: {person_count}")

            # 使用 PIL 來繪製中文文字
            # Convert OpenCV image (im0) to PIL Image
            im_pil = Image.fromarray(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
            draw_pil = ImageDraw.Draw(im_pil)

            # Set font (You need to have a font that supports Chinese characters)
            font_path = "C:/Windows/Fonts/simsun.ttc"  # 請確保字體文件存在或替換為您系統中的中文字體
            try:
                font_status = ImageFont.truetype(font_path, 30)
                font_count = ImageFont.truetype(font_path, 30)
            except IOError:
                print(f"字體文件 {font_path} 不存在。請確保字體文件存在或更改 font_path 為有效路徑。")
                font_status = ImageFont.load_default()
                font_count = ImageFont.load_default()

            # Convert colors from BGR to RGB
            status_color_rgb = (status_color[2], status_color[1], status_color[0])
            white_color_rgb = (255, 255, 255)

            # Draw status and person count
            draw_pil.text((10, 10), f"狀態: {status}", font=font_status, fill=status_color_rgb)
            draw_pil.text((10, 50), f"人數: {person_count}", font=font_count, fill=white_color_rgb)

            # Convert back to OpenCV image
            im0 = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)

            if save_img and p in video_writers:
                video_writers[p].write(im0)

            if (datetime.now() - last_insert_time_60s_datetime) >= timedelta(seconds=60):
                current_time = datetime.now()
                start_time_db = last_insert_time_60s_datetime
                end_time_db = current_time

                total_duration = sum(obj['duration'] for obj in interval_departed_objects)
                total_person_count = len(interval_departed_objects)

                if total_person_count > 0:
                    average_duration = total_duration / total_person_count
                else:
                    average_duration = 0

                occupancy_rate = (person_count / (spatial_density * limit_the_number_of_people)) * 100 if spatial_density * limit_the_number_of_people != 0 else 0
                occupancy_rate = min(round(occupancy_rate, 2), 100)

                insert_traffic_stats(location_id, start_time_db, end_time_db, person_count, average_duration, occupancy_rate)

                if occupancy_rate >= 100:
                    average_occupancy = occupancy_rate
                    insert_congestion_record(location_id, start_time_db, end_time_db, average_occupancy)
                if occupancy_rate < 60:
                    average_occupancy = occupancy_rate
                    insert_comfort_record(location_id, start_time_db, end_time_db, average_occupancy)

                interval_departed_objects.clear()
                last_insert_time_60s_datetime = current_time

            current_time = datetime.now()
            if status in ["即將壅塞", "壅塞"]:
                if (status != last_congestion_status) or (current_time - last_congestion_notification_time > congestion_notification_interval):
                    message = (
                        "\n\n"
                        f"⚠️ 警示通知：{location_name}現場人數達到{status}狀態！ ⚠️\n\n"
                        "📊 現況：\n"
                        f"- 當前人數：{person_count}人\n"
                        f"- 建議容納人數：{int(congestion_threshold)}人\n\n"
                        "🛠️ 請立即採取以下措施：\n"
                        "-----------\n"
                        "#### 1. 現場監測與通報\n"
                        "1. **櫃檯人員觀察壅塞**\n"
                        "   - 櫃檯人員注意到等待患者過多或排隊時間明顯延長。\n"
                        "   - 人員初步判斷是否達到壅塞標準。\n\n"
                        "2. **通知管理部門**\n"
                        "   - 櫃檯人員通過電話或內部即時通訊工具通知資訊室，說明壅塞狀況。\n"
                        "   - 必要時，同時通知護理站或院務處，請求現場支援。\n\n"
                        "#### 3. 調度支援人員\n"
                        "1. **通知增援人員**\n"
                        "   - 資訊室向院內預備人員或其他科室請求支援，增派人員到大廳。\n"
                        "   - 櫃檯服務人員根據資訊室指示，調整分工應對。\n\n"
                        "2. **實施分流措施**\n"
                        "   - 安排護理人員或志願者，引導患者至較空閒的櫃檯或自助服務區。\n"
                        "   - 必要時開放臨時服務窗口。\n\n"
                        "#### 4. 持續監控與更新\n"
                        "1. **動態監控**\n"
                        "   - 現場支援人員持續關注壅塞狀況，實時更新資訊室。\n"
                        "   - 資訊室根據最新情況，提供進一步的分流建議。\n\n"
                        "2. **事件結束**\n"
                        "   - 壅塞解除後，現場人員與資訊室記錄處理過程，形成報告。\n\n"
                        "🔗 查看儀表板詳情：\n"
                        f"{dashboard_url}"
                    )
                    status_code, response_text = send_line_notify(message, access_token)
                    if status_code == 200:
                        print(f"已發送{status}狀態通知")
                        last_congestion_status = status
                        last_congestion_notification_time = current_time
                    else:
                        print(f"發送通知失敗: {response_text}")

            for id in active_objects:
                data = active_objects[id]
                current_time = datetime.now()
                duration = current_time - data['arrival_time']
                if duration >= overstay_duration_threshold:
                    if not data.get('notified_over_30', False):
                        message = (
                            "\n\n"
                            f"⚠️ 滯留通知：ID為 {id} 的人員已在 {location_name} 滯留超過30分鐘！ ⚠️\n\n"
                            "🛠️ 請立即採取以下措施：\n"
                            "-----------\n"
                            "#### 1. 滯留檢測與通報\n"
                            "1. **櫃檯人員觀察滯留**\n"
                            "   - 櫃檯人員注意到滯留患者。\n"
                            "   - 人員初步判斷滯留原因及影響。\n\n"
                            "2. **通知管理部門**\n"
                            "   - 櫃檯人員通過電話或內部即時通訊工具通知資訊室，說明滯留狀況。\n"
                            "   - 必要時，同時通知護理站或院務處，請求現場支援。\n\n"
                            "#### 2. 調查與評估\n"
                            "1. **調查滯留原因**\n"
                            "   - 資訊室人員根據報告進行調查，了解滯留原因。\n\n"
                            "2. **評估影響**\n"
                            "   - 評估滯留對現場秩序和服務效率的影響。\n\n"
                            "#### 3. 調度支援人員\n"
                            "1. **通知增援人員**\n"
                            "   - 資訊室向院內預備人員或其他科室請求支援，增派人員到大廳。\n"
                            "   - 櫃檯服務人員根據資訊室指示，調整分工應對。\n\n"
                            "2. **實施處理措施**\n"
                            "   - 派遣護理人員或志願者協助滯留患者。\n"
                            "   - 必要時，提供額外的協助服務。\n\n"
                            "#### 4. 持續監控與更新\n"
                            "1. **動態監控**\n"
                            "   - 現場支援人員持續關注滯留狀況，實時更新資訊室。\n"
                            "   - 資訊室根據最新情況，提供進一步的協助建議。\n\n"
                            "2. **事件結束**\n"
                            "   - 滯留解除後，現場人員與資訊室記錄處理過程，形成報告。\n\n"
                            "🔗 查看儀表板詳情：\n"
                            f"{dashboard_url}"
                        )
                        status_code, response_text = send_line_notify(message, access_token)
                        if status_code == 200:
                            print(f"已發送ID {id} 的滯留通知")
                            data['notified_over_30'] = True
                        else:
                            print(f"發送通知失敗: {response_text}")

        # 只顯示第一個流（或您選擇的流）
        if view_img:
            window_name = f"Stream {i} - {p.name}"
            cv2.imshow(window_name, im0)
            cv2.waitKey(1)

        frame_count += 1

    print(f'完成。 ({time.time() - t0:.3f} 秒)')

    if save_img:
        for writer in video_writers.values():
            writer.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='模型路徑')
    parser.add_argument('--source', type=str, default='', help='來源')
    parser.add_argument('--img-size', type=int, default=480, help='推論圖像大小（像素）')  # 減少圖像大小以加快速度
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
    # 添加命令行參數，用於選擇要處理的攝影機編號
    parser.add_argument('--camera-index', type=int, default=-1, help='要處理的攝影機編號，-1表示顯示所有攝影機')
    opt = parser.parse_args()
    print(opt)

    # 讓用戶選擇位置
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

    # 獲取選擇的位置信息
    location_name, spatial_density = get_location_info(location_id)
    print(f"已選擇位置：{location_name}，空間大小：{spatial_density} 平方公尺")

    # 添加分隔線
    print("-----------")

    # 詢問用戶是否要調整默認容納人數
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

    # 計算人流壅塞界限
    congestion_threshold = spatial_density * limit_the_number_of_people
    comfort_threshold = round(congestion_threshold * 0.4, 3)
    normal_threshold = round(congestion_threshold * 0.6, 3)
    near_congestion_threshold = round(congestion_threshold * 0.8, 3)

    print(f"人流壅塞界限: {congestion_threshold}")
    print(f"即將壅塞界限: {near_congestion_threshold}")
    print(f"人流舒適界限: {comfort_threshold}")
    print(f"人流正常界限: {normal_threshold}")

    # 添加分隔線
    print("-----------")

    # 添加 RTSP 來源，請自行填寫用戶名、密碼和 IP 地址，並設置緩衝區參數
    rtsp_url_1 = "rtsp://admin:Mlab03+-@192.168.0.118:554/cam/realmonitor?channel=1&subtype=0&buffer_size=1&fps=30"
    rtsp_url_2 = "rtsp://admin:Mlab03+-@192.168.0.118:554/cam/realmonitor?channel=2&subtype=0&buffer_size=1&fps=30"

    # 如果未在命令行指定 source，則使用預設的 RTSP 流
    if not opt.source:
        rtsp_urls = [rtsp_url_1, rtsp_url_2]
        camera_names = ["球形攝影機", "槍型攝影機"]
        # 如果用戶指定了攝影機編號，則只使用對應的 RTSP 流
        if opt.camera_index >= 0 and opt.camera_index < len(rtsp_urls):
            opt.source = rtsp_urls[opt.camera_index]
        else:
            # 讓用戶選擇要處理的攝影機
            print("請選擇要處理的攝影機：")
            print("-----------")
            for idx, name in enumerate(camera_names):
                print(f"{idx}. {name}")
            print("-----------")
            while True:
                try:
                    camera_choice = int(input(f"請輸入攝影機編號 (0-{len(rtsp_urls)-1}): "))
                    if 0 <= camera_choice < len(rtsp_urls):
                        opt.source = rtsp_urls[camera_choice]
                        break
                    else:
                        print(f"請輸入有效的攝影機編號 (0-{len(rtsp_urls)-1})。")
                except ValueError:
                    print("請輸入有效的數字。")
    else:
        # 如果命令行指定了 source，則使用命令行的值
        opt.source = opt.source

    with torch.no_grad():
        if opt.update:
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
