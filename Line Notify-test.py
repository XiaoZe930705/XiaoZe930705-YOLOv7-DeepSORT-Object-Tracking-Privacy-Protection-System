import requests

def send_line_notify(message, token):
    """
    發送訊息至 Line Notify
    :param message: 要發送的訊息內容
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

if __name__ == "__main__":
    access_token = 'OQftvj6sWK7m2n1BD0fAmMPV5YGyTJsmoysrxejwzbh'  # 替換為你的存取權杖
    dashboard_url = "http://127.0.0.1:3000/d/fe18lsbehefpce/18d6023f-901b-59bd-aff1-a26fd9ceed32?orgId=1&showCategory=Panel+options&from=1732941598588&to=1732942198588"
    
    # 設定要發送的訊息，包含提醒內容與網址
    message = (
        "\n\n"  # 在第一行添加兩個換行符號，確保自動名稱後的內容分開
        "⚠️ 警示通知：醫院大廳左半部現場人數即將達到壅塞狀態！ ⚠️\n\n"
        "📊 現況：\n"
        "- 當前人數：26人\n"
        "- 建議容納人數：32人\n\n"
        "🛠️ 請立即採取以下措施：\n"
        "1. 通知現場人員：請立即告知大廳人員暫停新進人員進入。\n"
        "2. 引導現有人員疏散：請協助引導已在大廳的民眾，通過其他出入口有序離開。\n"
        "3. 滯留過久處理：\n"
        "   - 先查看監控畫面，確認滯留人員的情況。\n"
        "   - 通知相關人員（如客服、櫃檯或保全）進行處理。\n"
        "   - 視情況加開櫃檯，提升現場處理效率。\n"
        "   - 派遣保全前往現場，協助疏散或維持秩序。\n\n"
        "🔗 查看儀表板詳情：\n"
        f"{dashboard_url}"
    )
    
    status, response = send_line_notify(message, access_token)
    
    if status == 200:
        print("訊息發送成功！")
    elif status is None:
        print(f"發送失敗，錯誤訊息：{response}")
    else:
        print(f"發送失敗，狀態碼：{status}")
        print(f"回應內容：{response}")
