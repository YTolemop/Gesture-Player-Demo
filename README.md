# Gesture Player Demo

這是已完成辨識實作、包含錄製資料與訓練模型的手勢遊戲 Demo。遊戲網站：[https://ga-game.codpy.org/](https://ga-game.codpy.org/)。

資料流程是：鏡頭 → MediaPipe 原始結果 → 左右手整理 → 特徵正規化 → 三通道 KNN → 拒絕與時間濾波 → 遊戲指令。

本包能分別輸出 `left`、`right`、`both`。左右手單手指令可同時送出；雙手合成手勢辨識成功時獨占輸出，左右欄位回到 `NONE`。

## 專案架構與資料流程

`main.py` 是操作入口，`core/` 取得鏡像影像與 MediaPipe 原始結果，`student/` 完成三通道辨識，`network/` 負責與競技場通訊。Demo 已有辨識實作，可先使用現有模型，再閱讀各階段如何銜接。

```text
main.py                    命令列入口：鏡頭、錄製、訓練、測試、辨識與連線
core/
  camera.py                讀取攝影機，提供鏡像影像
  landmarks.py             呼叫 MediaPipe，取得原始手部資料
  types.py                 定義原始手部資料格式
  preview.py               顯示影像與原始關節疊圖
  resources.py             下載與驗證 MediaPipe 關節模型
student/
  pipeline.py              串接訓練、模型載入、三通道辨識與狀態重設
  data.py                  讀取錄製資料、整理左右手與通道
  normalization.py         特徵正規化
  knn.py                   KNN 分類與拒絕判斷
  filtering.py             時間濾波
  labels.py、types.py       分類標籤與辨識結果格式
network/
  commands.py              定義合法的 left／right／both 輸出
  arena_client.py          處理配對、傳送指令與連線狀態
config.example.json        預設辨識設定；config.json 可覆寫設定
protocol/                  競技場通訊規格與範例封包
tests/                     工具、辨識流程與介面契約的測試
```

```text
錄製 record
  鏡頭 → MediaPipe → 原始關節 + 標籤／通道／用途 → data/
  鏡頭 → 鏡像影片（供人工檢查）

訓練 train 與測試 test
  train 資料 → 左右手整理 → 特徵正規化 → 三通道 KNN 資料與設定 → models/model.npz
  test 資料  → 相同特徵轉換 → 已訓練模型 → 獨立測試報告

辨識 recognize 與遊玩 run
  鏡頭 → MediaPipe → RawHandSample → Pipeline.predict() → Commands
                                         ↑                 └→ run 傳送至競技場
                                  載入三通道模型與設定
```

`recognize` 用於本機檢查辨識結果，`run` 才會連線傳送遊戲指令。`left`、`right` 可同時輸出，`both` 成功時獨占輸出；三通道的處理集中在 `student/pipeline.py`。

`assets/models/hand_landmarker.task` 是 MediaPipe 關節模型，`models/model.npz` 才是自訂手勢分類模型。修改特徵轉換後，需重新訓練，確保訓練與即時辨識使用一致的特徵。

閱讀程式時，建議先看 `core/types.py` 的輸入與 `network/commands.py` 的輸出，再看 `student/pipeline.py` 如何串接各演算法。

## 1. 設定環境


需要 Python 3.13、攝影機，以及首次下載依賴與關節模型所需的網路。建議使用 VS Code 編輯程式。遊戲網址：[https://ga-game.codpy.org/](https://ga-game.codpy.org/)。

先完整解壓縮學員包，在 VS Code 選「檔案 → 開啟資料夾」，開啟內含 `main.py` 的專案資料夾，再選「終端機 → 新增終端機」。接下來的指令都在這個資料夾執行。

### 建立環境與安裝套件

依自己的系統選一組指令。`.venv` 是專案自己的 Python 環境；`requirements-lock.txt` 列出所需套件與固定版本。這一段只需在第一次使用或重建環境時執行。

Windows：使用 PowerShell。

```powershell
py -3.13 --version
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe main.py prepare
.\.venv\Scripts\python.exe main.py check
```

macOS：使用 Terminal 的 zsh 或 bash。M 系列晶片請使用原生 Apple Silicon 的 Python 與終端機環境。

```bash
python3.13 --version
python3.13 -m venv .venv
./.venv/bin/python -m pip install -r requirements-lock.txt
./.venv/bin/python main.py prepare
./.venv/bin/python main.py check
```

Linux：使用本機桌面環境的 bash 或 zsh，並先安裝 Python 3.13。

```bash
python3.13 --version
python3.13 -m venv .venv
./.venv/bin/python -m pip install -r requirements-lock.txt
./.venv/bin/python main.py prepare
./.venv/bin/python main.py check
```

若 Linux 在建立 `.venv` 時提示缺少 `venv` 或 `ensurepip`，請安裝與 Python 3.13 對應的 venv 套件。以套件來源有提供該版本的 Ubuntu／Debian 系統為例：

```bash
sudo apt update
sudo apt install python3.13-venv
```

安裝後重新執行建立環境與安裝套件的指令。關節預覽使用 OpenCV 視窗，不需要另外安裝 Tk 套件。

各系統第一行應顯示 `Python 3.13.x`。`prepare` 下載並驗證關節模型；`check` 成功時會顯示 `Dependencies and landmark model OK`。這只代表依賴與關節模型可用，尚未測試鏡頭或你的辨識程式。

上面各行分別負責不同的事，若某一步失敗，先處理該步的錯誤：

| 指令部分 | 用途 |
| --- | --- |
| `--version` | 確認使用的 Python 版本 |
| `-m venv .venv` | 建立這個專案專用的 Python 環境 |
| `-m pip install -r requirements-lock.txt` | 用該環境的 Python 安裝指定套件 |
| `main.py prepare` | 執行專案工具，準備關節模型 |
| `main.py check` | 檢查套件與關節模型是否可用 |

### 後續操作使用同一個環境

後面的範例統一使用 `python main.py ...`。每次重新開啟終端機，先回到專案資料夾並啟用環境。

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS／Linux：

```bash
source .venv/bin/activate
```

若 PowerShell 不允許執行啟用腳本，可以不啟用，將後續指令開頭的 `python` 換成 `.\.venv\Scripts\python.exe`。macOS／Linux 也可直接使用 `./.venv/bin/python`。

在 VS Code 按 `Ctrl+Shift+P`（Mac 為 `Cmd+Shift+P`），搜尋 `Python: Select Interpreter`，選本專案 `.venv` 的 Python，讓編輯器也使用相同環境。

## 2. 使用現有資料與模型

不需要重新錄製就能使用。`data/` 保留 20 組 NPZ／JSON 批次，共 2,154 筆訓練樣本與 651 筆獨立測試樣本。新模型為 `models/model.npz`，內含三個分類通道、設定、標籤順序與訓練批次清單。

先確認鏡頭能抓到關節：

```powershell
python main.py camera --preview --seconds 10
```

再測試即時辨識，不連線遊戲：

```powershell
python main.py recognize --preview
```

終端機約每秒顯示一次偵測數量與 `commands`。預覽只顯示原始關節、模型左右手分類與分數，辨識結果請看終端機。原始左右手名稱可能與送出的通道不同，因為 Demo 的辨識流程另有左右手修正設定。

按 `Ctrl+C`，或在預覽視窗按 `Q`、`Esc`、關閉視窗停止。不加 `--preview` 時只顯示終端機狀態；可用 `--seconds 10` 限制時間。所有操作都在內含 `main.py` 的專案資料夾執行。

若開到錯誤鏡頭，對 `camera`、`record`、`recognize` 或 `run` 加上 `--camera 1`，並在錄製和辨識時使用相同索引。

## 3. 現有手勢與三個輸出通道

### 十個遊戲動作

| 序號 | 遊戲動作 | 對應指令 |
| --- | --- | --- |
| 1 | 向左移動 | `MOVE_LEFT` |
| 2 | 向右移動 | `MOVE_RIGHT` |
| 3 | 跳躍 | `JUMP` |
| 4 | 蹲下 | `CROUCH` |
| 5 | 防禦 | `BLOCK` |
| 6 | 普通攻擊 | `NORMAL_ATTACK` |
| 7 | 攻擊技能 | `ATTACK_SKILL` |
| 8 | 防禦技能 | `DEFENSE_SKILL` |
| 9 | 控制技能 | `CONTROL_SKILL` |
| 10 | 終極技能 | `ULTIMATE` |

現有資料的通道安排如下。標籤代表遊戲動作，手勢外觀以實際錄製內容為準，資料沒有額外保存文字姿勢定義。

| 通道 | 現有指令類別 |
| --- | --- |
| left | `MOVE_LEFT`、`JUMP`、`CROUCH` |
| right | `MOVE_RIGHT`、`NORMAL_ATTACK`、`ATTACK_SKILL`、`CONTROL_SKILL` |
| both | `BLOCK`、`DEFENSE_SKILL`、`ULTIMATE` |

辨識先檢查雙手合成模型。若通過距離與投票門檻，輸出 `both`，並將 `left`、`right` 設為 `NONE`。即使雙手濾波仍在確認，也不會先送出單手動作。

若沒有辨識到雙手合成手勢，就分別辨識左右手。例如左手做 `MOVE_LEFT`、右手做 `NORMAL_ATTACK` 時，三欄會是 `left=MOVE_LEFT`、`right=NORMAL_ATTACK`、`both=NONE`。不會再把它們壓成一個指令。

左右通道各自判斷缺手、信心與濾波；一隻手移開，不會把另一隻手的合法輸出一起清掉。全部缺手時，三欄都回到 `NONE`。

## 4. 連線遊戲

### PVE 連線（不需要房號）

1. 開啟[遊戲網站](https://ga-game.codpy.org/)，選擇 PVE「一命連戰」。
2. 控制來源選擇「手勢辨識器」，按「產生配對碼」。
3. 在含有 `main.py` 的 Demo 或 initial 專案資料夾啟動辨識，將下方 `012345` 換成網站當下顯示的六位數配對碼：

```powershell
.\.venv\Scripts\python.exe main.py run --server "https://ga-game.codpy.org" --pair-code "012345" --model models/model.npz
```

4. 看到 `Paired` 後回到網站確認準備、選擇技能，倒數後開始挑戰。按「結束挑戰」會保存進行中挑戰的未完成紀錄、離開房間並回到首頁；再次遊玩需重新產生配對碼並啟動辨識器。

配對碼用來連接自己的 Python 辨識器，並非雙人邀請碼；保留開頭的零，PVE 不必加 `--room`。配對碼過期或斷線後，請重新產生。網址不用加 `/controller`。macOS／Linux 將指令開頭換成 `./.venv/bin/python`；已啟用虛擬環境時可使用 `python`。initial 需先完成自己的辨識實作並訓練模型，Demo 可使用隨附的模型。

`run` 會自動將鏡頭影像與手部骨架送到網站，無須加 `--preview`；加上 `--preview` 會另外開啟本機關節視窗。若有多支鏡頭，加上 `--camera 1` 選擇另一支。

若出現 `No module named 'cv2'`，代表目前 Python 缺少 OpenCV；先用同一個虛擬環境安裝套件，再檢查：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe main.py prepare
.\.venv\Scripts\python.exe main.py check
```

若尚未建立 `.venv`，先執行 `py -3.13 -m venv .venv`。雙人對戰則在網站建立或加入房間，同樣使用自己的配對碼；`--room` 可選填。

`run` 會使用同一支鏡頭，自動將 320×240 JPEG 與手部骨架傳到網站，約每 150ms 一次；無須加 `--preview`。`--preview` 另外開啟本機關節視窗。網頁影像保持與本機相同的鏡像方向；非 4:3 畫面等比例補黑邊，骨架一同轉換。`camera`、`record`、`recognize` 不傳送網站預覽。

每個影格都做辨識，網路輸出上限約每秒 10 次。配對、封包序號、心跳與中立狀態由 `network/arena_client.py` 處理。比賽狀態改變時會清除辨識歷史，並短暫傳送中立狀態。停止時送出中立指令並關閉連線；斷線後重新取得配對碼，再啟動程式。

移動、蹲下、防禦是持續狀態。跳躍、攻擊與技能在進入指令時觸發，維持手勢不會連發；需離開指令至少 6 個遊戲 tick 後再做一次。遊戲為 60 tick/s。左右手相同指令合併一次，相反移動抵銷；冷卻、資源與硬直等限制由遊戲判斷。

## 5. 重新訓練與評估

使用現有資料重新建立三通道模型：

```powershell
python main.py train --data data --model models/model.npz
python main.py test --data data --model models/model.npz --output models/evaluation-three-channel.json
```

`train` 只讀取用途為 `train` 的資料，`test` 只讀取 `test`。原有 NPZ 的用途記錄在同名 JSON 中；新 JSONL 的用途記錄在每行 `purpose` 欄位中。尚未標用途的原始 JSONL 視為訓練資料。

本次實際使用的訓練數量：

| 通道 | 可用訓練樣本 | 略過樣本 | 獨立測試樣本 | 本次離線正確率 |
| --- | --- | --- | --- | --- |
| left | 673 | 0 | 198 | 100% |
| right | 847 | 1 | 258 | 100% |
| both | 627 | 6 | 195 | 100% |

略過的 7 筆為側別不符或缺少雙手所需資料，原始檔案沒有刪除或修改。測試包含最終三通道路由，但不包含時間濾波。

另以獨立測試資料中各單手類別的前 5 筆，組合成 300 組左右手輸入，驗證兩側可同時輸出，全部通過。這是合成輸入測試，不是實際同時擺出雙手的鏡頭測試。既有資料沒有完整的未知手勢測試集，因此這些結果不代表對陌生手勢、不同使用者或現場環境也有相同效果。

原有 `models/knn.npz`、`knn.json`、`evaluation.json` 保留作為原始成果；目前程式使用 `models/model.npz` 與 `evaluation-three-channel.json`。

## 6. 增加自己的錄製資料

新錄製同時保存所有手的原始結果，`--channel` 是人工標註，不會替你篩選影像裡的手。例：先擺好右手攻擊手勢，再逐次錄製訓練與測試資料。

```powershell
python main.py record --label NORMAL_ATTACK --channel right --purpose train --seconds 10 --preview
python main.py record --label NORMAL_ATTACK --channel right --purpose test --seconds 10 --preview
```

雙手合成手勢的例子：

```powershell
python main.py record --label BLOCK --channel both --purpose train --seconds 10 --preview
```

每次在 `data/` 新建 MP4 與 JSONL，不覆寫已錄資料。MP4 是未畫關節的鏡像影像、不含聲音，固定 30 FPS；分析以 JSONL 時間戳為準。JSONL 保存影像座標、三維座標、原始左右手分類與分數，沒有手時也會記錄空清單。

新標籤若不是遊戲指令，需在設定的 `gesture_commands` 加入對應，例如把自訂名稱映射到 `NORMAL_ATTACK`。原有 NPZ 資料透過 `label_channels` 決定通道；新錄製則使用明確標註的 `channel`。

錄完檢查影片，移除或另存做錯手勢的資料，再重新執行 `train` 與 `test`。請將訓練與測試分次錄製，不要只把相鄰影格隨機拆開。

## 7. 各階段在 Demo 裡如何實作

這些就是學員從原始 MediaPipe 結果到遊戲輸出需要完成的工作；本 Demo 已提供以下實作。

| 階段 | 本 Demo 的做法 | 程式位置 |
| --- | --- | --- |
| 擷取與檢查 | 鏡像鏡頭影像，使用 MediaPipe VIDEO 模式，最多偵測兩隻手；預覽畫出原始結果 | `core/camera.py`、`core/landmarks.py`、`core/preview.py` |
| 原始結果整理 | 根據分類分數選左右手、套用左右手修正；同側出現重複結果時清除該側，保留另一側 | `student/data.py` 的 `resolve_hands()` |
| 資料清理 | 檢查形狀、有限數值、影像尺寸及手掌尺度；原有資料按指定通道處理，側別錯誤或缺手則略過 | `student/data.py` |
| 正規化 | 換算像素比例、減去手腕位置、除以手腕到中指掌指關節的距離 | `student/normalization.py` |
| 特徵 | 兩手各 63 維座標、2 維存在標記、3 維相對位置，共 131 維 | `student/normalization.py` |
| 訓練 | 分別保存 left／right／both 的特徵與標籤，三個模型共用相同特徵規格 | `student/pipeline.py` 的 `train()` |
| KNN 與拒絕 | 歐氏距離、最近 k 筆投票；最近距離與投票比例都通過門檻才接受 | `student/knn.py` |
| 時間濾波 | 每通道獨立確認候選結果，缺手或未知結果立即清除，回合切換時全部重置 | `student/filtering.py` |
| 路由 | 雙手合成優先且獨占，否則左右手並行；回傳完整 `Commands` | `student/pipeline.py` 的 `decide()`、`predict()` |
| 評估 | 只使用獨立測試批次，輸出各通道混淆矩陣與正確率 | `student/pipeline.py` 的 `evaluate()` |
| 網路 | 以 Socket.IO 配對，送出三欄完整狀態，處理心跳、世代與比賽切換 | `network/arena_client.py` |

### 座標與特徵的細節

影像正規化座標轉成像素比例時，x、z 乘影像寬度，y 乘影像高度。每隻手以手腕為原點，以手腕到第 9 號關節的距離為尺度；尺度小於 10px 時拒絕。缺少的手保留零向量，存在的手在對應標記填入 2。

雙手相對位置使用兩手腕的差，除以平均手掌尺度。每隻手的 z 有不同參考原點，因此相對位置的 z 設成 0，只保留相對 x、y。單手通道只放入自己那隻手，另一隻手填空，避免同時操作時互相污染特徵。

這套特徵處理平移與尺度，但沒有做旋轉不變化。新原始資料雖保留 `world_landmarks`，目前分類沿用影像座標，以重用既有錄製資料。不同手的三維座標各自以該手幾何中心為原點，不能直接用來計算雙手場景間距。

### 門檻與濾波

設定從 `config.example.json` 載入；可複製成 `config.json` 覆寫。

| 設定 | 本 Demo 預設值 | 用途 |
| --- | --- | --- |
| `swap_handedness` | `true` | 保留原本錄製時使用的左右手修正 |
| `handedness_threshold` | `0.7` | 左右手分類分數門檻，不是手勢辨識信心 |
| `feature_version` | `mixed-hands-131-v1` | 特徵規格標記 |
| `k` | `5` | KNN 鄰居數 |
| `distance_threshold` | `1.0` | 最近鄰距離上限 |
| `vote_threshold` | `0.6` | 得票比例下限 |
| `filter_enabled` | `false` | 沿用既有 Demo 設定；改成 true 可開啟濾波 |

模型內保存訓練時的設定。修改特徵、側別處理或 KNN 設定後要重新訓練；`filter_enabled` 可在啟動時切換，不需重訓。原有 NPZ 已整理成左右手，不會在讀取時再次交換；新 MediaPipe 結果才會套用 `swap_handedness`。

開啟濾波時，一般候選至少確認兩幀且持續 0.05 秒；連續高得票（至少 0.8）候選可縮短為 0.02 秒。短暫有效類別跳動可銜接原動作最多 0.05 秒；未知、缺手、時間倒退或超過 0.5 秒的中斷會清除歷史。

### 原始資料格式

`core/types.py` 定義 `RawHandSample`，其中 `hands` 是本幀各隻手的清單，空清單代表無手。每隻手包含 `landmarks`、`world_landmarks` 與 `handedness`；後者保留分類名稱、index、display_name 與 score。順序不代表固定左右手，也不是追蹤 ID。

原有 NPZ 批次只保存已整理的影像座標、存在標記與標籤，沒有三維座標或原始分類分數。讀取器直接使用已保存的左右手，不會捏造缺少欄位，也不會重寫原始資料。

## 8. 檢查與常見問題

執行自動測試：

```powershell
python -m pytest -q
```

測試涵蓋原始結果保存、預覽不污染錄影、資料用途分離、獨立測試資料、300 組左右手合成輸入、雙手獨占、濾波隔離、缺手清除，以及網路封包與狀態切換。

| 狀況 | 檢查方式 |
| --- | --- |
| 缺少套件 | 使用本專案 `.venv` 安裝 `requirements-lock.txt` |
| 缺少關節模型 | 執行 `python main.py prepare` |
| 模型無法載入 | 確認使用 `models/model.npz`；可用現有 `data/` 重新訓練 |
| 鏡頭無法開啟 | 檢查攝影機權限、其他程式占用與 `--camera` 索引 |
| 預覽無法開啟 | 使用可開視窗的桌面環境，確認沒有改裝 headless OpenCV；可先不加 `--preview` |
| 左右手輸出不對 | 核對原始預覽與 `swap_handedness`，修改側別設定後重訓 |
| 單手可以、雙手一起時不穩 | 查看是否誤觸 `both`、鏡頭是否抓到兩隻手，補錄同時操作情境做驗證 |
| 誤辨成雙手合成手勢 | 檢查雙手樣本、距離門檻與特徵，補充未知姿勢測試；模型不保證拒絕所有陌生姿勢 |
| 配對失敗 | 確認網站可開啟、房間與配對碼正確，斷線後重新取得配對碼 |
| 已配對但沒動作 | 確認網站已準備開始，查看終端機 commands 與遊戲冷卻／資源限制 |
| 攻擊不會連發 | 離開該指令至少 6 個遊戲 tick，再做一次手勢 |

鏡頭權限：Windows 需允許桌面應用程式使用攝影機；macOS 需在隱私權設定允許 Terminal／VS Code；Linux 需有裝置存取權限。`run` 會將縮小的鏡頭影像與二維手部骨架經遊戲伺服器轉送到同場網頁；原始錄影與完整關節資料留在本機。

## 指令速查

| 指令 | 用途 |
| --- | --- |
| `prepare` | 下載並驗證 MediaPipe 關節模型 |
| `check` | 檢查依賴及關節模型，不開鏡頭 |
| `camera` | 檢查鏡頭與原始偵測，可加 `--preview` |
| `record` | 錄製原始資料，指定 label、channel、purpose |
| `train` | 由現有資料建立三通道模型 |
| `test` | 評估獨立測試資料並輸出 JSON 報告 |
| `recognize` | 不連線遊戲，查看即時辨識指令 |
| `run` | 即時辨識並連線遊戲 |

使用 `python main.py --help` 或各子命令的 `--help` 查看完整參數。主程式採終端機流程，預覽使用 OpenCV 視窗。錄製資料與模型隨 Demo 提供；`.venv` 需在各自電腦建立。
