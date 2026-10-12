---
title: 手勢訓練器：學員操作與程式實作指南

---

# 手勢訓練器：從第一筆資料到控制競技場

本文預設學員已依課前通知安裝好 **VS Code 與 Python 3.13.2**。

>若還未完成，請先安裝完畢後再進行以下步驟，如安裝遇到問題，請妥善利用網路資源，或尋求助教協助。

## 目錄

- [1. 下載專案](#step-1)
- [2. 建立.venv與安裝套件](#step-2)
- [3. 開啟工具，確認鏡頭](#step-3)
- [4. 錄下十種手勢](#step-4)
- [5. 完成正規化、KNN與選做濾波](#step-5)
- [6. 訓練並試試辨識效果](#step-6)
- [7. 連上競技場](#step-7)
- [選做：左手、雙手與混合手勢](#mixed-hands)
- [其它](#others)

<a id="step-1"></a>
## 1. 下載專案

1. 開啟 [GitHub專案](https://github.com/YTolemop/Gesture-Player)，按 **Code → Download ZIP**。
![螢幕擷取畫面 2026-10-10 164705](https://hackmd.io/_uploads/SkYAsuwsfx.png)
3. 完整解壓縮到自己的資料夾，不要在壓縮檔裡直接執行。
4. 開啟VS Code。選「檔案 → 開啟資料夾」，開啟解壓後的 **Gesture-Player-main**。
![螢幕擷取畫面 2026-10-10 180343](https://hackmd.io/_uploads/rkepatwjfl.png)
6. 選「終端機 → 新增終端機」。接下來的命令都在這裡輸入。
![螢幕擷取畫面 2026-10-10 164645](https://hackmd.io/_uploads/ryKk3Ovifg.png)


完成時：
![image](https://hackmd.io/_uploads/B1NIAKDjGg.png)

> Windows用PowerShell，Mac／Linux用zsh或bash。

<details>
<summary>已經會使用Git？也可以這樣下載</summary>

```bash
git clone https://github.com/YTolemop/Gesture-Player.git
cd Gesture-Player
```

</details>

<a id="step-2"></a>
## 2. 建立.venv與安裝套件

Python已安裝好，這一步替專案建立獨立環境並安裝需要的套件。

| 你的系統 | 使用哪組指令 |
|---|---|
| Windows x64 | Windows PowerShell |
| Mac（M系列晶片） | Mac／Linux，使用原生Apple Silicon環境 |
| Linux x86_64 | Mac／Linux，需有桌面環境、glibc 2.28以上與Tk支援 |

### 在終端機建立環境

Windows先確認版本，再建立環境：

```powershell
python --version
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-lock.txt
& '.\.venv\Scripts\python.exe' main.py prepare
& '.\.venv\Scripts\python.exe' main.py check
```

Mac／Linux使用：

```bash
python3.13 --version
python3.13 -m venv .venv
./.venv/bin/python -m pip install -r requirements-lock.txt
./.venv/bin/python main.py prepare
./.venv/bin/python main.py check
```

第一行應顯示 **Python 3.13.2**。`.venv` 就是本專案自己的Python環境；`requirements-lock.txt` 列出要安裝的套件與版本。只在第一次使用或重建環境時做這一段，不需要每次開工具都重做。

完成時：
![image](https://hackmd.io/_uploads/SJxS19vifx.png)

### 讓VS Code選對Python

按 **Ctrl+Shift+P**（Mac為 **Cmd+Shift+P**）搜尋 **Python: Select Interpreter**，選本專案的.venv。
![螢幕擷取畫面 2026-10-10 181208](https://hackmd.io/_uploads/Bk4WxcvjMx.png)
![螢幕擷取畫面 2026-10-10 181235](https://hackmd.io/_uploads/BJw-e5Pofg.png)

<a id="step-3"></a>
## 3. 開啟工具，確認鏡頭

以下保留右手單手的入門路線。工具本身也接受左手與雙手，不需要切換模式；想在同一模型混合使用，完成基礎閱讀後接著看「[左手、雙手與混合手勢](#mixed-hands)」。算法仍由學員實作，採集成功不代表目前的正規化已支援所有手勢。

之後每次開啟，只要用這一行。

**Windows PowerShell**

```powershell
& '.\.venv\Scripts\python.exe' main.py gui
```

**Mac／Linux**

```bash
./.venv/bin/python main.py gui
```

1. 在「即時辨識」頁，模式先保持 **preview**（預覽）。
2. 鏡頭索引先選0，按「開啟鏡頭」。如果沒有鏡頭畫面再切換索引。
![螢幕擷取畫面 2026-10-10 181529](https://hackmd.io/_uploads/Hy1Kb9woMl.png)
3. 把右手完整放進畫面，確認手上有點與連線。
4. 看右側採集頁的偵測文字，確認自己的右手被標為右手。
![螢幕擷取畫面 2026-10-10 182004](https://hackmd.io/_uploads/rJB_-5vsGx.png)

| 常見問題 | 解決方法 |
|---|---|
| 鏡頭黑畫面 | 關閉其他使用鏡頭的程式、換索引，確認系統攝影機權限 |
| 偵測左右手反了 | 關閉工具，修改config.example.json中的swap_handedness，切換其true／false，重開確認 |

>鏡頭權限：Windows在設定的攝影機隱私權允許桌面應用程式；Mac在「系統設定 → 隱私權與安全性 → 攝影機」允許VS Code／Terminal後重啟；Linux需本機桌面與攝影機裝置的存取權限，必要時請管理員協助。請在本機桌面操作，不在WSL或無桌面遠端終端開啟鏡頭GUI。


<a id="step-4"></a>
## 4. 錄下十種手勢

先為每一類想一個**不同的固定姿勢**，把對應記下來。例如張開五指代表跳、握拳代表蹲下；這只是例子，你可以自行設計。想要用左手或者是雙手也都可以，這邊為了簡單選擇使用右手單手進行教學。

| 類別編號 | 標籤 | 遊戲動作 |
|---|---|---|
| 0 | MOVE_LEFT | 向左移動 |
| 1 | MOVE_RIGHT | 向右移動 |
| 2 | JUMP | 跳躍 |
| 3 | CROUCH | 蹲下 |
| 4 | BLOCK | 格擋 |
| 5 | NORMAL_ATTACK | 普通攻擊 |
| 6 | ATTACK_SKILL | 攻擊技能 |
| 7 | DEFENSE_SKILL | 防禦技能 |
| 8 | CONTROL_SKILL | 控制技能 |
| 9 | ULTIMATE | 大招 |

### 先完成一個類別，再換下一個

1. 到「採集/訓練」，選一個「錄製類別」，用途選 **train**。
2. 按「開始錄製」，倒數三秒期間擺好右手。
![螢幕擷取畫面 2026-10-10 182156](https://hackmd.io/_uploads/S1BWG5Pjfl.png)
3. 保持姿勢約10～15秒，可以輕微改變位置、距離；確認「有效」筆數增加。
4. 按「停止錄製」，資料會自動保存，列表出現類別、用途、筆數與時間。
![螢幕擷取畫面 2026-10-10 182448](https://hackmd.io/_uploads/Sk1e79wsfx.png)
![螢幕擷取畫面 2026-10-10 182504](https://hackmd.io/_uploads/By4e75PjGx.png)
5. 同一類改成 **test**，先放下手，再重新擺出姿勢錄一次並保存。
![螢幕擷取畫面 2026-10-10 182719](https://hackmd.io/_uploads/H1cXmqDsfg.png)
6. 換下一類，直到十類都有train與test。

完成時：資料列表中，每一類至少有一筆train和一筆test。

>需重錄時選對那列，按「刪除所選資料」。
![螢幕擷取畫面 2026-10-10 182831](https://hackmd.io/_uploads/ByHOX5wizl.png)

保存位置是`data/`，每次有一個.npz（關鍵點、類別）和同名.json（用途等資訊）。這時不用按訓練，先繼續完成下一步的算法。

<a id="step-5"></a>
## 5. 完成三個算法

你要接手的是這段流程：

```text
右手21個點 → 正規化成63個數值 → KNN判斷手勢 → 可選濾波 → 競技場
```

只修改下面三個檔案，保留已有的類別、函式名稱與參數，將 `raise NotImplementedError(...)` 換成你的實作。不需要另寫main，也不用自己呼叫鏡頭、讀取錄製檔或連線競技場。

| 順序 | 檔案與函式 | 要解決的問題 | 回傳結果 |
|---|---|---|---|
| 先寫 | `student/normalization.py` 的 `transform` | 同一姿勢換位置、變大小，仍能比較 | `FeatureResult`，含63個float32值 |
| 再寫 | `student/knn.py` 的 `fit`、`predict` | 用已錄製的樣本判斷新手勢 | `fit` 回None；`predict` 回 `Prediction` |
| 選做 | `student/filtering.py` 的 `update`、`reset` | 避免分類短暫跳動就切換動作 | `update` 回標籤字串；`reset` 回None |

**資料會透過函式參數送進來。** 接下來每段先說明計算方法，再提供可對照的完整實作。先理解每一步，完成基本版本後，再嘗試修改策略。

### A. 正規化：消除位置與大小影響

同一個手勢移到畫面另一邊，所有點的座標都會改變；靠近鏡頭，點之間的距離也會變大。我們先換成一致單位，再讓手腕成為原點，最後除以手掌大小。

#### 第1步：取得資料，換成一致的像素尺度

`transform` 收到的 `sample` 是一筆手部資料。`sample.right` 是21列、3欄的NumPy陣列，每一列代表一個點，每列的三欄依序是x、y、z。`sample.width`、`sample.height` 是影像的像素寬度與高度。

原始x、y使用畫面比例，z使用畫面寬度作為尺度，因此x與z乘以寬度，y乘以高度：

```python
points = np.array(sample.right, dtype=np.float64, copy=True)
points[:, 0] *= sample.width
points[:, 1] *= sample.height
points[:, 2] *= sample.width
```

第一行把已有的資料複製成陣列，使用64位元浮點數計算；不是讀檔。`copy=True` 避免後面修改到原始資料。`np` 來自檔案開頭的 `import numpy as np`。

`points[:, 0]` 表示「全部列的第0欄」，也就是21個x；第1欄是y，第2欄是z。例如畫面寬640像素，x為0.5，換算後就是320。

#### 第2步：所有點減去手腕，消除位置影響

第0個點是手腕。Python索引從0開始，所以用 `points[0]` 取得它的xyz：

```python
wrist = points[0]
centered = points - wrist
```

NumPy會讓每一列都減去同一組手腕xyz。手腕因此變成 `(0, 0, 0)`，其餘點表示相對於手腕的位置。當整隻手平移時，這些相對位置不會跟著平移。

#### 第3步：除以手掌大小，消除整體縮放影響

第9個點是中指根部。我們用手腕到中指根部的三維距離代表手掌大小：

```python
palm_size = np.linalg.norm(centered[9])
normalized = centered / palm_size
```

手腕已經在原點，所以 `centered[9]` 就是手腕到中指根部的向量。`np.linalg.norm` 計算向量長度，即 `sqrt(x² + y² + z²)`。

所有座標除以這個長度後，手腕到中指根部的距離變成1。例如某點離手腕40、手掌尺度80，相除是0.5；整體放大兩倍後，80除以160仍是0.5。這能消除整體縮放的影響，但不能保證不同拍攝角度的手勢完全相同。

我們沒有旋轉或翻轉座標，因此仍保留手勢方向。

#### 第4步：排成63個數值並回傳

21個點乘以每點3個數值，共63個值：

```python
features = normalized.reshape(63).astype(np.float32)
```

`reshape(63)` 依序排成「第0點x、y、z，第1點x、y、z……」；`astype(np.float32)` 轉成接口要求的32位元浮點數。最後把它放進 `FeatureResult`，工具才知道如何使用。

#### 正規化完整實作

在 `student/normalization.py` 保留原有接口，實作如下：

```python
import numpy as np
from core.types import RawHandSample, FeatureResult


class StudentNormalizer:
    def transform(self, sample: RawHandSample, config: dict) -> FeatureResult:
        # 本入門算法只接受右手單獨入鏡；工具仍可採集左手與雙手。
        if sample.right is None or sample.left is not None:
            return FeatureResult(
                valid=False,
                reason='此入門算法僅接受右手單手，混合手勢請使用進階範例',
                featureVersion=config['feature_version'],
            )
        # 換成一致的像素尺度；建立副本，不修改輸入。
        points = np.array(sample.right, dtype=np.float64, copy=True)
        points[:, 0] *= sample.width
        points[:, 1] *= sample.height
        points[:, 2] *= sample.width

        # 以手腕為原點，再除以手掌尺度。
        centered = points - points[0]
        palm_size = np.linalg.norm(centered[9])
        normalized = centered / palm_size

        features = normalized.reshape(63).astype(np.float32)
        return FeatureResult(
            valid=True,
            features=features,
            featureVersion=config['feature_version'],
        )
```

### B. KNN：找出最相似的樣本，再投票

KNN的想法是：新手勢像哪些已錄製的手勢，就讓最相似的k筆樣本投票。例如k=5，最近五筆的類別是 `[2, 2, 3, 2, 3]`，類別2得到三票，得票比例為 `3 / 5 = 0.6`。

我們分兩個函式完成：`fit` 保存訓練資料與設定；`predict` 比較一筆新資料、找鄰居、投票並決定是否接受分類。

#### 第1步：在fit保存訓練資料與設定

工具會提供以下參數：

| 參數 | 內容 |
|---|---|
| `features` | `(N, 63)` 陣列，每列是一筆正規化後的手勢；N是樣本數 |
| `labels` | `(N,)` 整數陣列，每筆對應同一列手勢的類別編號0～9 |
| `config` | 包含k、投票門檻與距離門檻的設定 |

`features[0]` 和 `labels[0]` 分別是第一筆手勢與它的正確類別。保存方式如下：

```python
self.training_features = features.copy()
self.training_labels = labels.copy()
self.k = config.get('k', 5)
self.vote_threshold = config.get('vote_threshold', 0.6)
self.distance_threshold = config.get('distance_threshold', 1.0)
```

`self` 代表目前的分類器物件。資料存到 `self` 上，之後的 `predict` 才能取得；只寫區域變數 `training_features = ...`，不會保存成物件屬性。

`config.get('k', 5)` 表示取得k，若沒有這個欄位就使用5。`fit` 不分類，也不需要自己寫模型檔；函式結束時回傳None即可。

#### 第2步：在predict計算每筆樣本的距離

`predict` 收到的 `features` 是一筆新手勢，形狀 `(63,)`。讓每筆訓練特徵減去它，再算每一列的長度：

```python
differences = self.training_features - features
distances = np.linalg.norm(differences, axis=1)
```

相减後仍是 `(N, 63)`；`axis=1` 表示沿每一列的63個數值計算，得到N個距離。這就是歐氏距離：差異的平方相加，再開平方。距離越小，表示兩筆正規化特徵越接近。

#### 第3步：取最近的k筆樣本

```python
order = np.argsort(distances, kind='stable')
neighbor_indices = order[:self.k]
neighbor_labels = self.training_labels[neighbor_indices]
```

`argsort` 回傳的是由近到遠排列的「樣本索引」，不是距離本身。例如距離是 `[0.8, 0.2, 0.5]`，索引順序就是 `[1, 2, 0]`。`[:self.k]` 取前k個，再用這些索引取得鄰居的類別。

`kind='stable'` 讓距離相同的樣本維持原本順序，結果較容易重現。

#### 第4步：統計票數，選出類別

```python
counts = np.bincount(neighbor_labels, minlength=len(LABELS))
class_id = int(np.argmax(counts))
ratio = float(counts[class_id] / self.k)
```

`bincount` 計算每個類別有幾票；`argmax` 取得票數最多的類別編號。這個基本版本遇到同票時，選編號較小的類別，因為 `argmax` 會取第一個最大值。之後也可以改成以候選類別的距離決勝。

`ratio` 是鄰居投票比例，**不是辨識準確率，也不是校準後的機率**。

#### 第5步：太遠或票數不集中，就拒絕分類

即使是沒錄過的姿勢，也一定找得到最近的樣本。因此不能只看誰票數最多，還要設定接受條件：

```python
distance = float(distances[neighbor_indices[0]])
valid = (
    distance <= self.distance_threshold
    and ratio >= self.vote_threshold
)
label = LABELS[class_id] if valid else 'NONE'
```

這裡要求「最近樣本夠近」和「得票比例夠高」兩項都通過。`LABELS[class_id]` 把編號轉成標籤字串，例如類別2對應 `JUMP`。拒絕時使用 `NONE`，表示現在沒有可接受的分類。門檻只能協助拒絕，不能保證所有未知手勢都會被拒絕。

#### KNN完整實作

把以下內容放進 `student/knn.py` 對應的方法。**沿用原檔的類別名稱與函式簽名**；下面以 `StudentKNN` 示範類別結構：

```python
import numpy as np
from core.labels import LABELS
from core.types import Prediction


class StudentKNN:
    def fit(self, features, labels, config):
        self.training_features = features.copy()
        self.training_labels = labels.copy()
        self.k = int(config.get('k', 5))
        self.vote_threshold = float(config.get('vote_threshold', 0.6))
        self.distance_threshold = float(config.get('distance_threshold', 1.0))
        return None

    def predict(self, features):
        differences = self.training_features - features
        distances = np.linalg.norm(differences, axis=1)

        order = np.argsort(distances, kind='stable')
        neighbor_indices = order[:self.k]
        neighbor_labels = self.training_labels[neighbor_indices]

        counts = np.bincount(neighbor_labels, minlength=len(LABELS))
        class_id = int(np.argmax(counts))
        ratio = float(counts[class_id] / self.k)
        distance = float(distances[neighbor_indices[0]])

        valid = (
            distance <= self.distance_threshold
            and ratio >= self.vote_threshold
        )
        label = LABELS[class_id] if valid else 'NONE'
        return Prediction(
            label=label,
            valid=valid,
            nearest_distance=distance,
            vote_ratio=ratio,
        )
```

### C. 濾波：快速確認新手勢，略過短暫跳動（選做）

先保持濾波關閉，確認KNN能辨識後再做。假設連續結果是「JUMP、JUMP、BLOCK、JUMP」，中間短暫的BLOCK可能只是抖動；每次分類一變就輸出，角色也會跟著跳動。

但如果每次換手勢都要等很久，玩家會覺得動作慢半拍。這裡使用**短時間加上連續次數確認**：新手勢至少連續兩次一致，最近連續兩次的投票比例都達0.8時，使用20ms門檻；其他結果使用50ms門檻。已確認的手勢可以持續輸出，短暫跳到另一個有效類別時最多保留原動作50ms；未知分類或無手時立即停止。

20ms與50ms是濾波的確認門檻，不是整個系統的反應時間。程式要等下一筆分類到來才能確認；鏡頭更新率、關鍵點偵測和網路傳送都會影響實際速度。

#### 第1步：分清楚候選與穩定輸出

濾波每次收到的是KNN回傳的 `prediction`，不是21個座標：

| 資料 | 用途 |
|---|---|
| `prediction.label` | 這次分類的標籤，例如 `JUMP` |
| `prediction.valid` | KNN是否接受這次分類 |
| `prediction.vote_ratio` | 最近鄰投給勝出類別的比例，例如0.8表示八成；不是辨識正確率 |
| `timestamp` | 這次更新的單調遞增時鐘秒數，不是影格編號或日期 |

先在 `__init__` 設定三個門檻，再呼叫 `reset()` 初始化狀態：

```python
self.hold_seconds = 0.05
self.fast_hold_seconds = 0.02
self.strong_vote = 0.8
self.reset()
```

`hold_seconds` 是一般確認時間，也用來限制保留舊動作的時間；`fast_hold_seconds` 是高投票比例的確認時間；`strong_vote` 是使用快速確認的投票門檻。

在 `reset()` 保存以下資料，下一次更新才能接著判斷：

| 狀態 | 用途 |
|---|---|
| `candidate` | 正在確認的新標籤，初始為None |
| `start_time` | 第一次看到目前候選的時間，初始為None |
| `frames` | 目前候選連續出現幾次，初始為0 |
| `strong_frames` | 目前候選最近連續幾次達到投票門檻，初始為0 |
| `stable_label` | 已確認、可以輸出的標籤，初始為NONE |
| `transition_start` | 開始離開穩定標籤的時間，初始為None |
| `last_timestamp` | 上一筆更新時間，初始為None |

`candidate` 和 `stable_label` 分開保存。例如已確認JUMP，接著只收到一次BLOCK：BLOCK是候選，但還不能直接把穩定輸出改成BLOCK。

#### 第2步：無效分類立即停止，舊資料不接著算

```python
if not math.isfinite(timestamp) or not prediction.valid or prediction.label == 'NONE':
    self.reset()
    return 'NONE'
```

未知分類不能延續上一個攻擊，所以要清除候選與穩定輸出，立即回傳NONE。`math.isfinite` 也會擋下無效時間值，檔案開頭記得加入 `import math`。無手時，共用工具會直接停止輸出並重設濾波，不會等待確認時間。

另外，時間倒退、重複使用同一時間，或兩筆更新相隔超過0.5秒，都不能把前一筆資料當成連續證據：

```python
if self.last_timestamp is not None and (timestamp <= self.last_timestamp or timestamp - self.last_timestamp > 0.5):
    self.reset()
self.last_timestamp = timestamp
```

例如鏡頭暫停後重新送來JUMP，必須重新確認，不能直接沿用暫停前的JUMP。

#### 第3步：已確認的手勢，直接繼續輸出

```python
if prediction.label == self.stable_label:
    self.candidate = None
    self.transition_start = None
    return self.stable_label
```

第一次看到JUMP時還沒有穩定輸出，需要往下確認；確認後，只要仍收到有效的JUMP，就直接回傳JUMP，不必每筆再等一次。

如果上一筆短暫變成BLOCK，這一筆又回到穩定的JUMP，也會清掉BLOCK候選及切換計時。這樣單次分類跳動就不會把動作切開。

#### 第4步：計算候選連續次數與投票比例

```python
if prediction.label != self.candidate:
    self.candidate = prediction.label
    self.start_time = timestamp
    self.frames = 1
    self.strong_frames = int(prediction.vote_ratio >= self.strong_vote)
else:
    self.frames += 1
    self.strong_frames = self.strong_frames + 1 if prediction.vote_ratio >= self.strong_vote else 0
```

候選改變時，從第一次開始計數。`int(True)` 是1，`int(False)` 是0，因此第一筆達到0.8就把 `strong_frames` 設成1，否則設成0。

候選相同時，`frames` 加1；投票比例達到門檻，`strong_frames` 才繼續累加。只要某次投票比例不足，就把高投票比例的連續次數歸零，但仍保留同一候選的一般次數與開始時間。

例如JUMP的投票比例依序為「0.6、1.0」，雖然標籤連續一致，仍不能使用快速門檻，因為只有最近一次達到0.8。

#### 第5步：次數與時間都通過，才確認新手勢

```python
elapsed = timestamp - self.start_time
hold = self.fast_hold_seconds if self.strong_frames >= 2 else self.hold_seconds
if self.frames >= 2 and elapsed >= hold:
    self.stable_label = self.candidate
    self.transition_start = None
    return self.stable_label
```

這裡有兩個條件：**同一候選至少連續兩次，而且從候選開始到現在已達到確認時間。** 單筆結果就算投票比例是1.0，也不能直接通過。

假設鏡頭每秒更新30次，兩筆約相隔33ms：

| 時間 | 分類與投票比例 | 輸出 |
|---|---|---|
| 12.000秒 | JUMP，0.8 | NONE，先記錄候選 |
| 12.033秒 | JUMP，0.8 | JUMP，連續兩次且超過20ms |
| 12.066秒 | JUMP，0.8 | JUMP，已確認，直接延續 |

如果每筆的投票比例都是0.6，第二筆只有33ms，還沒達到一般50ms門檻；第三筆約66ms才會通過。更新率較高時，也可能需要超過兩筆才能累積20ms，所以不能只用影格數來代替時間。

#### 第6步：短暫跳動保留舊動作，但不能一直保留

新候選還沒確認時，如果之前已有穩定輸出，就給短暫跳動一個小空檔：

```python
if self.stable_label != 'NONE':
    if self.transition_start is None:
        self.transition_start = timestamp
    if timestamp - self.transition_start < self.hold_seconds:
        return self.stable_label
return 'NONE'
```

例如已確認JUMP，接著只有一次BLOCK，再回到JUMP：短暫的BLOCK期間先回傳原本的JUMP，回到JUMP後繼續輸出。若BLOCK連續出現並通過確認，就改成BLOCK。

`transition_start` 只在第一次離開穩定標籤時設定，不能每次候選改變就重算。否則候選一直在BLOCK、CROUCH、MOVE_RIGHT之間跳動，舊的MOVE_LEFT可能永遠停不下來。從開始切換滿50ms的那筆更新起，若仍沒確認新候選，就回NONE；沒有新更新時，共用輸出也會在0.5秒後失效。

這段只保留**有效分類之間的短暫跳動**。收到未知分類或手離開畫面，仍由第2步或共用工具立即停止，不會保留舊動作。

#### 濾波完整實作

以下整段放進 `student/filtering.py`，**沿用原有類別名稱與函式簽名**。不要在同一類別中留下兩份 `update` 或 `reset`：

```python
import math
from core.types import Prediction


class StudentFilter:
    def __init__(self):
        self.hold_seconds = 0.05
        self.fast_hold_seconds = 0.02
        self.strong_vote = 0.8
        self.reset()

    def reset(self):
        self.candidate = None
        self.start_time = None
        self.frames = 0
        self.strong_frames = 0
        self.stable_label = 'NONE'
        self.transition_start = None
        self.last_timestamp = None
        return None

    def update(self, prediction: Prediction, timestamp: float) -> str:
        if not math.isfinite(timestamp) or not prediction.valid or prediction.label == 'NONE':
            self.reset()
            return 'NONE'

        # 暫停或時間重設後，不能沿用舊影格的確認結果。
        if self.last_timestamp is not None and (timestamp <= self.last_timestamp or timestamp - self.last_timestamp > 0.5):
            self.reset()
        self.last_timestamp = timestamp

        if prediction.label == self.stable_label:
            self.candidate = None
            self.transition_start = None
            return self.stable_label

        if prediction.label != self.candidate:
            self.candidate = prediction.label
            self.start_time = timestamp
            self.frames = 1
            self.strong_frames = int(prediction.vote_ratio >= self.strong_vote)
        else:
            self.frames += 1
            self.strong_frames = self.strong_frames + 1 if prediction.vote_ratio >= self.strong_vote else 0

        elapsed = timestamp - self.start_time
        hold = self.fast_hold_seconds if self.strong_frames >= 2 else self.hold_seconds
        if self.frames >= 2 and elapsed >= hold:
            self.stable_label = self.candidate
            self.transition_start = None
            return self.stable_label

        # 短暫有效分類跳動時保留原動作，但切換計時不能一直重來。
        if self.stable_label != 'NONE':
            if self.transition_start is None:
                self.transition_start = timestamp
            if timestamp - self.transition_start < self.hold_seconds:
                return self.stable_label
        return 'NONE'
```

#### 實測與調整

保存後重開工具，勾選「使用學員濾波（選做）」，比較關閉與開啟時的反應。不需要重新訓練KNN，因為這次修改的是分類後的輸出流程。

先測固定手勢，再測快速切換、短暫分類跳動、未知手勢及手離開畫面。穩定動作應能持續輸出，新手勢確認後應切換，未知或無手時應立即停止。

如果動作仍慢，先看KNN的原始分類是否已經穩定，再調整 `hold_seconds` 或 `fast_hold_seconds`；如果常誤切換，可以增加確認時間，或提高 `strong_vote`。每次只改一個參數並記錄結果，保留至少連續兩次確認；也記得 `hold_seconds` 同時控制舊動作的保留時間。

### 接口測試與實際確認

「接口」就是輸入和輸出的約定。完成程式後先保存，再執行：

**Windows PowerShell**

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_student_contracts.py tests/test_filtering.py
```

**Mac／Linux**

```bash
./.venv/bin/python -m pytest tests/test_student_contracts.py tests/test_filtering.py
```

看到 `passed` 表示通過這份測試涵蓋的檢查，還需要在下一節訓練與實測，確認辨識效果。

<a id="step-6"></a>
## 6. 訓練並試試辨識效果

1. 保存程式，關閉工具再重新開啟，讓它載入新程式。
2. 在「採集/訓練」按 **訓練**。工具只使用train資料，成功後會自動載入模型。
3. 按 **測試**，使用test檢查模型。報告會存成`models/evaluation.json`。
![螢幕擷取畫面 2026-10-10 200526](https://hackmd.io/_uploads/BkH89svoMx.png)
4. 到「即時辨識」，開鏡頭，把preview改成 **recognize**（辨識）。
5. 逐一做十個姿勢，再把手移出畫面，確認停止輸出。
![螢幕擷取畫面 2026-10-10 200715](https://hackmd.io/_uploads/Hydi5iviGg.png)

> 下次重開工具，要先按「載入KNN模型」，不必每次重新訓練。
![螢幕擷取畫面 2026-10-10 200826](https://hackmd.io/_uploads/rypAcsDofg.png)

<a id="step-7"></a>
## 7. 連上競技場

1. 網站進「假人練習」，控制來源選「手勢辨識器」，按「產生配對碼」。
![螢幕擷取畫面 2026-10-10 201741](https://hackmd.io/_uploads/S1zyCoPizg.png)
![螢幕擷取畫面 2026-10-10 201915](https://hackmd.io/_uploads/SkOxRivizx.png)
2. Python工具載入模型、開鏡頭、切recognize，先做一個手勢確認分類可用。
![螢幕擷取畫面 2026-10-10 201841](https://hackmd.io/_uploads/BkI1RsDsfe.png)
4. 到「配對」填主機網址和配對碼；單人練習房間碼留空。
![螢幕擷取畫面 2026-10-10 201931](https://hackmd.io/_uploads/HyCx0jwsfl.png)
4. 按「配對」。成功後按鈕會變成「解除配對」。
5. 網站顯示辨識器已連線、算法可用後，按「確認準備」，選技能並等待倒數。
6. 試左右移動、普通攻擊與技能；最後把手移開，確認角色停止接收動作。

## 想改善辨識？再調整KNN參數

學員可以調整 `k`、`vote_threshold`、`distance_threshold`。直接修改專案內的 `config.example.json`。JSON數值不要加引號，並保留正確逗號格式。

| 參數 | 預設值 | 允許範圍與用途 |
|---|---|---|
| `k` | 5 | 正整數，不可超過實際可用的訓練樣本數；決定取幾個最近鄰居投票 |
| `vote_threshold` | 0.6 | 0～1；最低得票比例，越高越要求鄰居分類一致，可能增加NONE |
| `distance_threshold` | 1.0 | 非負有限數值；最近距離的拒絕門檻，越小越嚴格，可能增加NONE。單位是正規化特徵距離，不是像素或公分 |

例如k=5、vote_threshold=0.6時，至少3個鄰居投同一類才符合投票條件；是否接受分類仍取決於你實作的距離拒絕規則。不要把投票比例當成實際辨識準確率。

**設定不會自動替你實作算法。** 你的 `fit` 必須讀取設定並存到self，`predict` 必須實際使用它們。可用的讀取方式：

```python
config.get('k', 5)
config.get('vote_threshold', 0.6)
config.get('distance_threshold', 1.0)
```

`get` 的第二個參數是欄位缺少時的預設值；只有呼叫get而沒有保存、使用結果，不會改變分類行為。若在算法中把k或門檻寫死，修改JSON也不會生效。

修改後依序：

1. 保存 `config.example.json`，關閉並重開工具，讓設定重新載入。
2. 按「訓練」，以新設定重新建立並載入模型；目前模型相容檢查包含這些參數，舊模型不能直接用新設定載入。
3. 按「測試」，比較獨立test結果，再做即時辨識。

建議每次只改一個參數並記錄結果。只調KNN參數通常不用重新採集；修改正規化特徵定義則需要重新計算特徵並訓練。工具接受左手、右手與雙手，是否能分類取決於你的正規化及訓練資料。舊設定hand_mode、single_hand與mirror_left已不控制採集；特徵中的翻轉策略由學員實作。

<a id="mixed-hands"></a>
## 選做：左手、雙手與混合手勢

要開放更多手勢，請用本節的正規化替換第5節A的入門版本，並重新訓練。**同一份資料、同一個模型可以同時包含左手單手、右手單手與雙手手勢，不需要切換模式，也不強迫十種類別都用雙手。**

不論使用幾隻手，整個姿勢每次只對應一個遊戲動作。雙手共同代表一個類別，例如雙手張開代表BLOCK；不是左手輸出移動、右手同時輸出攻擊。分類器回傳一個`Prediction`，其label是十個動作之一；拒絕分類則回NONE。

### 設計、採集與缺手

1. 為各類別記錄使用哪隻手、手型及兩手的相對位置。例如MOVE_LEFT用左手、MOVE_RIGHT用右手、BLOCK用雙手。
2. 同一類的train與test使用相同設計；若同一動作有多種替代姿勢，每種都各錄train與test。
3. 雙手類別錄製時確認左右手都有關鍵點；雙手遮擋、交叉或靠太近可能造成偵測失敗。畫面左右位置不代表解剖學左右手，請分別伸出左、右手確認標示，需要時調整swap_handedness。
4. 原始資料採集只要求至少一隻有效的手，不知道你選的類別需要幾隻手。錄雙手類別時若漏掉一隻，工具仍可能收進單手資料；請檢查偵測文字，出錯的批次重錄。
5. 分類拒絕、無手或無效追蹤時應輸出NONE。雙手少一隻時也可能成為有效的已訓練單手姿勢，不能一律停止所有單手分類；請測試這些過渡姿勢，必要時改手勢或在算法中增加拒絕條件。

`sample.left`與`sample.right`各是`(21,3)`陣列，未偵測到的手是`None`。保存檔原本就有左右兩個欄位及存在標記，因此舊右手原始資料可繼續重算特徵；要學新的左手或雙手姿勢仍須補錄。

### 固定長度的混合手勢特徵

工具接受學員自訂的非空一維float32特徵`(D,)`，但**同一模型所有樣本的D與欄位意義必須一致**。不能單手回63維、雙手回126維。以下範例固定使用131維：

| 索引 | 內容 |
|---|---|
| 0～62 | 左手：手腕歸零、除以自己的手掌大小；缺手填0 |
| 63～125 | 右手：同樣正規化；缺手填0 |
| 126～127 | 左、右手存在標記；存在填2，缺手填0 |
| 128～130 | 右手腕減左手腕的xyz，除以兩手平均手掌大小；缺任一手填0 |

各手的63維保留手型與方向；最後3維保留兩手的相對位置，避免兩手各自歸零後丟失這項資訊。左、右欄位固定，不能按偵測順序或畫面左右排列。兩手深度z各有自己的參考原點，因此相對位置的z在此填0，只使用可靠的影像平面x、y；不是兩手真實三維距離。

存在標記使用2作為距離權重，讓缺手與握拳不會只靠零值混在一起。用本範例的歐氏距離及distance_threshold=1.0時，手數／左右存在狀態不相同的樣本距離至少2，會被拒絕；若提高門檻，這項保護也會變弱。可自行調整特徵權重，但要重新評估門檻。

### 混合手勢正規化完整實作

以下整段替換`student/normalization.py`。保留原有類別及函式接口，不要把兩份transform放在同一類別中。

```python
import numpy as np
from core.types import RawHandSample, FeatureResult


class StudentNormalizer:
    def transform(self, sample: RawHandSample, config: dict) -> FeatureResult:
        features = np.zeros(131, dtype=np.float32)
        wrists = [None, None]
        sizes = [None, None]

        for slot, hand in enumerate((sample.left, sample.right)):
            if hand is None:
                continue
            points = np.array(hand, dtype=np.float64, copy=True)
            points *= np.array([sample.width, sample.height, sample.width])
            wrists[slot] = points[0].copy()
            centered = points - points[0]
            sizes[slot] = np.linalg.norm(centered[9])
            start = slot * 63
            features[start:start + 63] = (centered / sizes[slot]).reshape(63)
            features[126 + slot] = 2.0

        if wrists[0] is None and wrists[1] is None:
            return FeatureResult(
                valid=False, reason='無手',
                featureVersion=config['feature_version'],
            )
        if wrists[0] is not None and wrists[1] is not None:
            relative = (wrists[1] - wrists[0]) / ((sizes[0] + sizes[1]) / 2)
            relative[2] = 0.0  # 每隻手的z參考原點不同，只保留相對x、y。
            features[128:131] = relative

        return FeatureResult(
            valid=True, features=features,
            featureVersion=config['feature_version'],
        )
```

共用工具會先檢查入鏡的每隻手：尺寸、21個點、有限數值及至少10px的手掌尺度，再呼叫此算法。算法可以自行拒絕不支援的樣本，但必須回傳`valid=False`且features為None。

### KNN、版本與重新訓練

第5節B的KNN完整實作可直接沿用：NumPy減法與axis=1會自動處理D個欄位，程式沒有把維度寫死。該節的63維解說對應右手入門版本；本節fit收到`(N,131)`，predict收到`(131,)`。自訂特徵也遵循`(N,D)`與`(D,)`的接口。

修改特徵欄位、維度、正規化或權重後：

1. 更新實際使用設定檔的feature_version，例如`mixed-hands-131-v1`。一般使用config.example.json；若有config.json，工具優先讀取它。相同維度也可能有不同意義，工具不能自動判斷算法內容，請主動更新版本。
2. 保存算法與設定，關閉並重開工具。
3. 用原始資料重新「訓練」，再「測試」。工具會記錄模型維度與版本，拒絕不相容的模型或辨識輸入。舊格式的KNN模型需重新訓練。
4. 實測左手、右手、雙手、兩手位置變化、單雙手切換、雙手缺一手，以及全部離開畫面；確認每次只有一個動作，未知姿勢能拒絕，無手時停止輸出。

特徵變長後，舊距離門檻不一定合適。保持train／test獨立，再依實測調整k、距離與投票門檻。每種類別都需要足夠樣本，單手與雙手樣本數也應避免差距過大。

<a id="others"></a>
## 其它

<details>
<summary>各個資料夾與設定檔在做什麼？</summary>

| 位置 | 用途 | 學員要改嗎？ |
|---|---|---|
| student | 正規化、KNN、濾波算法 | 要，濾波選做 |
| data | 錄製後生成的原始關鍵點與用途 | 用介面管理 |
| models | 訓練後生成的個人模型、測試報告 | 不手改 |
| .venv | 本機Python與套件，每台電腦自行建立 | 不手改 |
| .vscode | 編輯器解譯器與測試設定 | 通常不用 |
| assets | MediaPipe關鍵點模型，不是個人KNN模型 | 不用 |
| core、ui、network | 鏡頭、資料讀寫、介面、安全檢查、配對 | 不用 |
| protocol | 與競技場共用的標籤與訊息格式 | 不用 |
| tests | 接口與共用工具測試 | 執行即可 |
| requirements-lock.txt | 固定套件版本，含間接依賴 | 不用 |
| pyproject.toml | 專案與pytest設定 | 不用 |
| config.example.json | 工具設定，直接在此調整KNN參數 | 需要時再調 |
| .gitignore | 避免把環境、個人資料與模型加入Git | 不用 |

套件損壞時可重新安裝requirements-lock，必要時重建.venv；保留student、data、models與你修改過的config.example.json。不要任意升級套件。

</details>

<details>
<summary>工具怎麼呼叫我的算法？</summary>

| 你的函式 | 共用工具入口 | 結果去向 |
|---|---|---|
| transform | core/features.py的feature | 檢查後交給訓練或分類 |
| fit | core/model_io.py的訓練、載入 | 保存資料／重建分類器 |
| predict | ui/app.py即時辨識、core/model_io.py測試 | 評估、濾波或直接輸出 |
| update、reset | core/features.py的OutputRouter | 控制輸出與清除歷史 |

訓練只讀train，測試只讀test；共用工具先驗證至少一隻手及所有入鏡手的尺寸、數值與手掌尺度，再呼叫正規化。你只要使用參數、計算並return。正規化定義改變後需要更新設定檔的feature_version、重算特徵並重新訓練；main.py不再固定特徵版本或限定右手。

</details>

<details>
<summary>第三方資源與授權</summary>


本工具只使用 Hand Landmarker，不使用 Gesture Recognizer。

模型保存位置：`assets/models/hand_landmarker.task`。
來源：https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
SHA-256：`fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`。
MediaPipe 原始碼採 Apache-2.0；官方模型依官方模型資源與使用條款使用。官方指南：https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python 。部署前確認組織適用條款。

OpenCV（Apache-2.0）、NumPy（BSD-3-Clause）、python-socketio（MIT）、Pillow（HPND）、pytest（MIT）及其依賴的授權隨安裝套件保存；完整版本在 requirements-lock.txt。
安裝時可下載，活動執行不下載模型。未提供離線套件安裝包，不複製 .venv 到其他電腦。

</details>
