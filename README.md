# wowfishtool

魔兽世界自动钓鱼脚本(Python)。抛竿后截图、模板匹配找鱼漂位置、移动鼠标;监听系统音频,
听到上钩声就自动右键拉杆。

> macOS 是实测平台。Windows 部分标 ⚠️ 的地方只是按原理写的,没有在真实 Windows 设备上跑过,
> 遇到问题欢迎反馈。

## 安装

### 1. 环境准备

Python 3.9+,推荐 3.11~3.13。

**macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**(没装过 Python/Git,推荐只装 PyCharm 一个东西):

```text
1. 装 PyCharm Community(免费):https://www.jetbrains.com/pycharm/download/

2. 拿代码:PyCharm 欢迎界面 → Get from VCS → 贴项目地址 → Clone
   备用方案:GitHub 页面 → Code → Download ZIP → 解压 → PyCharm File → Open 选中该文件夹

3. 装 Python:File → Settings → Project → Python Interpreter → Add Interpreter →
   Add Local Interpreter → Virtualenv Environment → Base interpreter 选 3.11 或 3.12

4. 后面所有"打开终端敲命令"的步骤,都用 PyCharm 底部的"终端"面板(不是"Python 控制台")
```

### 2. 系统权限

**macOS**:

```text
系统设置 → 隐私与安全性 → 辅助功能 → 勾选"终端"/"PyCharm"
系统设置 → 隐私与安全性 → 屏幕录制 → 勾选"终端"/"PyCharm"
改完重启终端/IDE
```

**Windows**:

```text
终端/IDE 和 WoW 都以管理员身份运行，或者两边都不用管理员权限——不要一边有一边没有
```

### 3. 配置音频回环设备(用来"听"上钩声音)

先调游戏内声音设置:

```text
声音 → 音效音量调大于 0
声音 → 音乐/环境声音调低或关闭
声音 → 关闭"后台/非焦点窗口时降低音量"
```

**macOS:装 [BlackHole](https://github.com/ExistentialAudio/BlackHole)**

```bash
brew install blackhole-2ch
```

```text
音频 MIDI 设置 → "+" → 新建"多输出设备" → 勾选 BlackHole 2ch + 你平时用的输出设备
系统设置 → 声音 → 输出 → 选中这个多输出设备
魔兽世界 → 声音设置 → 确认输出没被锁定到别的设备
```

**Windows:[VB-Audio Virtual Cable](https://vb-audio.com/Cable/)**(BlackHole 不支持 Windows)

```text
以管理员身份运行 VBCABLE_Setup_x64.exe → 重启电脑
Windows 设置 → 声音 → 输出设备 → 选 "CABLE Input"
  想同时自己听到：控制面板 → CABLE Input → 侦听 → 勾选"通过此设备播放" → 指向耳机/音箱
魔兽世界 → 声音设置 → 输出也选 "CABLE Input"
```

配置完用 `tests/test_listen.py` 实测能不能收到游戏声音、峰值均值多少,再照实际数值调
`fishing.py` 里 `listen()` 的 `threshold`。

### 4. 安装 Python 依赖

#### 4.1 macOS:先装系统的 portaudio(Windows 跳过这步)

```bash
brew install portaudio
```

#### 4.2 安装依赖(macOS / Windows 都要)

```bash
pip install -r requirements.txt
```

## 使用

### 用 PyCharm 运行(Windows)

```text
File → Open → 选中项目文件夹（包含 fishing.py 的那一层）
File → Settings → Project → Python Interpreter → 确认是项目里的 .venv
PyCharm 底部"终端"面板 → pip install -r requirements.txt
左侧项目树双击 fishing.py → 点行号左边的绿色三角形运行

# 运行面板打印出 "Press F10 to start fishing, F11 to stop" 后，切回游戏窗口按 F10/F11
```

macOS / 习惯命令行的可以跳过这节。

### 命令行运行(通用 / macOS)

1. 画质调到 4 档左右,关掉水面反射
2. 镜头拉到最近,拉到看不见自己人物为止
3. 钓鱼快捷键放快捷栏第 1 格(对应 `fishing.py` 里的 `CAST_KEY`)
4. 运行:

   ```bash
   python fishing.py
   ```

5. 切回游戏窗口,按 **F10** 开始自动钓鱼,**F11** 停止

### 抛竿落点参考

抛竿离角色本体太近,落点可能落进屏幕下方两块被强制排除识别的区域(默认角色状态框,以及
Modern 布局在右下角的同款副本,大致宽度 26%~36% 和 62%~76%、高度 71%~83%)——这两块区域
不管当时实际有没有状态框都会被判定"不可能是鱼漂",落进去就永远识别不到。抛竿离角色本体
远一点,让落点落在下图红框内:

![红框是鱼漂应该落点的安全区域](docs/float_safe_zone.png)

### 断线自动恢复

连续 `MAX_CONSECUTIVE_MISSES`(默认 10)次没抓到鱼,脚本会按 3 次回车尝试重新进入游戏,
再失败 10 次就彻底停止,回到"按 F10 重新开始"的状态。细节见 `fishing.py` 里
`try_recover_from_disconnect()` 的注释。

### 自动上鱼饵(可选)

游戏里做个宏绑快捷键 **2**:

```
/use 鱼饵的名字
/use 16
```

脚本开始钓鱼时按一次 `2`,之后每隔 `BAIT_REAPPLY_INTERVAL_SECONDS`(默认 10 分钟多)自动
再按一次。快捷键不是 2 的话改 `fishing.py` 里的 `BAIT_KEY`。

### 调试模式

`float_detector.py` 里 `DEBUG_SNAPSHOTS` 默认 `False`。改成 `True` 后会把识别失败的截图
自动存到 `debug/`(已在 `.gitignore` 里):

- 完全没找到鱼漂 → `debug/notfound_<时间戳>.png`
- 找到了鱼漂但点击点退化成兜底位置 → `debug/fallback_<时间戳>.png`(带绿框+红点标注)

调试完记得改回 `False`。

### 补一个新的鱼漂模板

`var/` 目录下的模板不一定适用于所有天气/水域/光照。识别不准时可以自己补一张:

1. 正常钓一竿,截图会存到 `var/fishing_session.png`(每次覆盖,想保留就手动复制)
2. 用截图工具把鱼漂本体单独裁出来(参考 `var/fishing_float_1.png` 的裁法)
3. 存成 `var/fishing_float_<下一个数字>.png`
4. 重新跑,不用改代码,`find_float()` 会自动纳入比较

## 运行测试

```bash
pytest
```

`tests/test_find_float.py`、`tests/test_audio_listener.py` 是自动化回归测试,不需要开游戏或
真实麦克风,改动识别/音频逻辑后应该先跑一遍。

`tests/test_click.py`、`tests/test_listen.py` 是需要人工确认结果的手动诊断脚本,当成普通
Python 脚本运行(`python tests/test_click.py`),不是 pytest 用例:

- `test_click.py`:验证模拟右键点击能不能传到别的窗口,主要排查 macOS 辅助功能权限
- `test_listen.py`:验证音频回环设备能不能收到游戏声音,配置 BlackHole/VB-Cable 后先跑这个

## 故障排查

**脚本运行了,但游戏里鼠标/键盘毫无反应**

- macOS:权限给的是"终端"还是"PyCharm"要跟实际运行 Python 的程序一致;改完权限必须重启
  终端/IDE 才生效。不确定权限有没有生效,跑 `tests/test_click.py` 单独验证,不用开游戏。
- Windows:如果 WoW 以管理员身份运行,终端/IDE 也必须以管理员身份运行,否则 Windows UIPI
  会静默丢弃低权限进程发给高权限窗口的输入——脚本自己一切正常,游戏就是没反应。两边权限
  必须一致,不要一边有一边没有。

**声音配置好了,脚本却一直"没听到咬钩"**

- 检查游戏"后台/非焦点窗口时降低音量"选项有没有关——一旦触发,游戏会把音量压得很低甚至
  静音,回环设备录到的上钩声音也会跟着变小到过不了 threshold,但看起来一切正常。Discord
  等软件的"检测到别人说话就压低其他程序音量"选项同理要关。
- Windows 上如果除 VB-Cable 外还装过其他虚拟声卡(VoiceMeeter 等),系统里会有好几个名字都
  带 "CABLE" 的录音设备,脚本按设备编号自动挑第一个匹配的,容易挑错——只保留 VB-Cable 一个。
- `audio_listener.py` 默认按系统类型找名字带 "blackhole"(macOS)或 "cable"(Windows)的
  输入设备;如果你的设备名不带这几个字,把 `LOOPBACK_NAME_HINT` 改成能匹配到的关键字。

**`pip install` 装 `pyaudio` 报错**(如 "Microsoft Visual C++ 14.0 or greater is required")

说明没有匹配你 Python 版本的现成 wheel:

```text
python -m pip install --upgrade pip
python --version   # 去 https://pypi.org/project/PyAudio/#files 确认有没有对应版本的 .whl
                    # 没有的话换一个稍旧的解释器版本（比如 3.11/3.12）重建虚拟环境
pip install -r requirements.txt
```

**IDE 里运行报 `ModuleNotFoundError`**

十有八九是"装的地方"和"跑的地方"不是同一个 Python 环境——检查 IDE 运行配置选的解释器路径
和 `pip install` 时激活的是不是同一个 `.venv`。

**鱼漂识别不准 / 找不到鱼漂**

- 画质越高、水面反射越强,越容易把周围场景颜色"染"到鱼漂上(尤其黄昏/夜晚),拖累识别精度——
  调低画质、关掉水面反射能明显改善。
- 极暗夜晚场景(尤其带月光反光纹理的水面)是已知薄弱点——水面波纹饱和度可能比鱼漂本体还高,
  击穿"鱼漂必须比水更艳"这条假设。这种情况脚本会自动放弃颜色门槛、只信形状匹配分数;如果
  还是识别不准,大概率是形状匹配分数不够,按上面"补一个新的鱼漂模板"补一张当前场景的模板。
- 镜头没拉到最近的话,人物/坐骑模型可能挡住鱼漂落点区域(`FLOAT_SEARCH_X_RANGE`/
  `FLOAT_SEARCH_Y_RANGE` 定义的范围)。

**定位点跑到角色状态框(头像/血条)上**

颜色门槛专挑"比水面更艳"的像素,角色状态框(金边头像、绿血条、蓝蓝条)正好又艳又亮,一旦
真正的鱼漂因为太暗被颜色门槛刷掉,状态框就可能顶上来被误当成鱼漂。已经把两个已知位置从
搜索范围里排除(`float_detector.py` 里的 `UI_EXCLUDE_REGIONS`,按屏幕宽高百分比定义):

1. 默认常驻的角色状态框(左下角)
2. WoW 默认 Edit Mode「Modern」布局在右下角的同款副本

按 2560×1410 实测量出,留了余量;如果你的分辨率/UI 缩放/摆放不一样,可能还是会被别的 UI
元素撞上——打开"调试模式",从 `debug/fallback_*.png` 或普通截图里量出那个 UI 元素的像素
范围,换算成宽高百分比加进 `UI_EXCLUDE_REGIONS`(格式 `((x0, x1), (y0, y1))`,0~1 之间)。
