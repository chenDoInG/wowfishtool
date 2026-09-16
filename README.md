# wowfishtool

魔兽世界自动钓鱼脚本(Python)。原理:抛竿后截图,用模板匹配找到鱼漂位置并把鼠标移过去;
监听系统音频,检测到持续的"上钩"声音后自动右键拉杆。

> macOS 是作者日常实测的平台;Windows 部分标了 ⚠️ 的地方是按原理写的配置说明,**没有在真实
> Windows 设备上跑过**(作者手头没有 Windows 机器)。`fishing.py` 里截图/找窗口/模拟点击这些
> 逻辑用的都是跨平台库,理论上能跑,遇到问题欢迎反馈。

## 安装

### 1. 环境准备(通用)

需要 Python 3.9+。建议用虚拟环境:

**macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. 系统权限(容易漏掉,漏了会表现成"脚本跑了但鼠标/键盘没反应")

**macOS**:脚本靠 `pyautogui`/`pynput` 模拟鼠标键盘、靠截图识别画面,这两件事在 macOS 上都
需要单独授权,而且授权对象是**实际运行 Python 的那个程序**(终端、PyCharm 或其他 IDE),不是
Python 本身:

1. 系统设置 → 隐私与安全性 → **辅助功能**,把终端/PyCharm 加进去并打开(控制鼠标键盘要用)
2. 系统设置 → 隐私与安全性 → **屏幕录制**,同样把终端/PyCharm 加进去并打开(截图要用,macOS
   10.15+ 强制要求,不给权限截图会是黑屏或全黑图片)
3. 改动权限后**必须重启终端/IDE**才会生效,改完立刻测试大概率还是不行,先重启一遍

不确定权限有没有生效,可以先跑 `tests/test_click.py`(见下方"运行测试")单独验证鼠标点击能不能
正常传到别的窗口,不用开游戏。

**Windows**:如果 WoW 是以管理员身份运行的,模拟鼠标键盘的进程权限必须**跟游戏一致**,否则
Windows 的 UIPI(用户界面特权隔离)会静默丢弃"权限低的进程发给权限高的窗口"的输入事件,表现
就是脚本日志一切正常、但游戏里毫无反应。简单的办法:要么两边都用管理员权限运行(终端/IDE 和
WoW 都右键"以管理员身份运行"),要么两边都不用管理员权限,不要一边有一边没有。

### 3. 配置音频回环设备(用来"听"上钩声音)

脚本靠监听系统音频里的上钩声音来判断什么时候拉杆,所以需要一个能把游戏声音路由进来的
虚拟回环录音设备,而不是对着麦克风收环境噪音。

**macOS:安装 [BlackHole](https://github.com/ExistentialAudio/BlackHole)**

```bash
brew install blackhole-2ch
```

装完之后:

1. 打开「音频 MIDI 设置」(启动台 → 其他 → 音频 MIDI 设置,或者直接搜索),点左下角 `+`
   新建一个「多输出设备」,勾选 BlackHole 2ch 和你平时用的扬声器/耳机(这样声音会同时发到
   两边,你自己也能听到游戏声音,不是静音操作)
2. 系统设置 → 声音 → 输出,选中刚建好的这个多输出设备
3. 打开魔兽世界的声音设置,确认输出没有被游戏自己锁定到别的设备上(部分游戏会记住上次用的
   输出设备,需要在游戏内选项里也确认一下)
4. `audio_listener.py` 里默认会自动找系统里名字带 "blackhole" 的输入设备,一般不用改代码;
   如果你的设备名不带这几个字(比如用了别的中文/自定义命名),把 `LOOPBACK_NAME_HINT` 改成
   实际能匹配到的关键字

**Windows:BlackHole 不支持 Windows** ⚠️,需要装一个功能类似的虚拟声卡,推荐二选一:

- [VB-Audio Virtual Cable](https://vb-audio.com/Cable/)(免费,最简单,推荐先试这个)
- [VoiceMeeter](https://vb-audio.com/Voicemeeter/)(功能更全,配置也更复杂,免费)

以 VB-Cable 为例大致步骤:

1. 下载安装包,解压后右键"以管理员身份运行" `VBCABLE_Setup_x64.exe` 安装(装虚拟声卡驱动
   必须用管理员权限,否则装了但系统里找不到设备)
2. 装完重启一次电脑(虚拟声卡驱动通常要求重启才会生效)
3. Windows 设置 → 系统 → 声音,把**输出设备**选成 "CABLE Input"——注意这样一来你自己就听不到
   游戏声音了,如果想同时听到,需要在"声音控制面板"里把 CABLE Input 的"侦听"打开,勾选
   "通过此设备播放"并指向你实际的耳机/音箱
4. 魔兽世界游戏内的声音输出设备也确认一下,同样指向 "CABLE Input"(或者干脆用系统默认,只要
   系统默认就是 CABLE Input)
5. 把 `audio_listener.py` 里的 `LOOPBACK_NAME_HINT` 改成设备名关键字:

   ```python
   LOOPBACK_NAME_HINT = 'cable'
   ```

配置完用 `tests/test_listen.py` 实测一下(见下方"运行测试"),确认游戏声音能被正常录到、
峰值/均值大概是多少,再照实际数值调 `fishing.py` 里 `listen()` 的 `threshold`。

### 4. 安装 Python 依赖

macOS 上 `pyaudio` 依赖系统的 `portaudio` 库,先装好再 `pip install`:

```bash
brew install portaudio
```

**Windows** 上 `pyaudio` 从 0.2.12 版本起,官方就直接在 PyPI 提供编译好的 wheel(已经把
PortAudio 打包进去了),覆盖 Python 3.8~3.13、32/64 位,正常情况下不需要单独装 PortAudio、
也不需要装 Visual Studio 编译工具,`pip install -r requirements.txt` 跟着装完就行。

如果装 `pyaudio` 这一步报错(常见报错是类似 "Microsoft Visual C++ 14.0 or greater is
required" 或者 "ERROR: No matching distribution found for pyaudio"),说明 pip 没找到对应
你 Python 版本的现成 wheel,想自己编译又缺编译环境,按顺序试:

1. **先确认 pip 是最新的**,旧版本 pip 有时候认不出新发布的 wheel:

   ```powershell
   python -m pip install --upgrade pip
   ```

2. **再确认 Python 版本和位数有没有现成 wheel**:打开
   [PyPI 上 PyAudio 的 Files 页面](https://pypi.org/project/PyAudio/#files),文件名里
   `cp312` 对应 Python 3.12、`win_amd64` 是 64 位、`win32` 是 32 位——对照你的 Python
   版本(`python --version`)看有没有匹配的 `.whl`。如果你的 Python 版本太新、还没发布
   对应 wheel,最省事的办法是换一个稍旧一点、肯定有 wheel 的 Python 版本(比如 3.11 或
   3.12)重新建虚拟环境

两边都一样:

```bash
pip install -r requirements.txt
```

## 使用

1. 打开魔兽世界,**建议把画质调到 4 档左右,关掉水面反射/镜面效果**——画质越高,水面的
   环境反射越强,会把周围场景的颜色"染"到鱼漂本体上(尤其黄昏/夜晚这种有色调滤镜的场景),
   鱼漂颜色跟水面颜色混到分不清,点击定位会掉精度。画质调低之后水面反射弱,鱼漂本身的颜色
   更干净,识别更稳
2. 把钓鱼快捷键放在快捷栏第 1 格(对应 `fishing.py` 里的 `CAST_KEY = '1'`,改了快捷栏位置
   要同步改这个常量)
3. 运行:

   ```bash
   python fishing.py
   ```

   脚本启动后会先检查 WoW 进程是否在跑,不在的话会直接退出,提示先打开游戏
4. 切回游戏窗口(脚本会提示"等 2 秒切窗口",利用这个时间点回游戏),按 **F10** 开始自动
   钓鱼,按 **F11** 停止——热键是全局监听的,不需要终端/IDE 窗口在前台

### 连续失败自动停止 / 断线自动尝试恢复

如果连续 `MAX_CONSECUTIVE_MISSES`(默认 10)次抛竿都没抓到鱼,脚本会认为大概率出了问题
(不只是手气不好),先按 3 次回车、间隔几秒去猜测性地尝试"如果是被服务器踢下线了,能不能
按几次回车重新进入游戏"(会经过断线弹窗确认 → 登录/服务器选择 → 角色选择 → 进入游戏这几屏,
每一步之间都留了等待时间),最后再按一次 Esc 把可能误开的聊天框关掉。

如果这次尝试之后还是连续 10 次失败,脚本会彻底停止,回到"按 F10 重新开始"的状态,不会无限
重试下去。这套恢复逻辑是**盲操作**,不会真的判断"是不是断线了",本质是"矬子里拔将军"式的
低成本兜底,不保证一定能救回来,只保证不会比什么都不做更差(见 `fishing.py` 里
`try_recover_from_disconnect()` 的注释)。

### 自动上鱼饵(可选)

如果用的鱼饵有持续时间/次数限制,需要定期重新上饵,可以在游戏里做一个宏绑在快捷键
**2** 上(鱼竿在主手栏位的话):

```
/use 鱼饵的名字
/use 16
```

脚本会在开始钓鱼时先按一次 `2`,之后每隔 `fishing.py` 里
`BAIT_REAPPLY_INTERVAL_SECONDS`(默认 10 分钟多一点,故意留了几秒余量避免鱼饵还没到期就被
提前刷新)自动再按一次,不用手动操作。如果你的鱼竿在
副手栏位,把宏里的 `/use 16` 改成 `/use 17`;如果快捷键不是 2,改 `fishing.py` 里的
`BAIT_KEY`。

### 调试模式(排查识别问题用)

`float_detector.py` 里有个开关 `DEBUG_SNAPSHOTS`,默认是 `False`,正常使用不会往磁盘写
任何调试文件。遇到识别不准、想事后复盘的时候,把它改成 `True`:

```python
# float_detector.py
DEBUG_SNAPSHOTS = True
```

开启后,遇到下面两种情况会自动把当时的截图存进 `debug/` 目录(独立于存正常运行数据的
`var/` 目录,不会跟模板图混在一起,也已经加进 `.gitignore`,不会被提交):

- **完全没找到鱼漂**(重试一次也没找到,放弃这一竿):存成 `debug/notfound_<时间戳>.png`,
  就是当次原始截图
- **找到了鱼漂,但点击点定位退化成了兜底**(底座颜色没匹配上,只能用匹配框的大致位置代替
  精确坐标):存成 `debug/fallback_<时间戳>.png`,并且会在图上画出匹配框(绿框)和最终
  点击点(红点),方便直接看出偏没偏

调试完记得改回 `False`,不然长时间挂机会攒一堆截图占地方。

## 鱼漂识别不准怎么办

不同天气、水域、光照下鱼漂的颜色/水面观感差别很大,`var/` 目录下自带的模板不一定每种场景都
适用。`find_float()` 会自动加载 `var/fishing_float_*.png` 下所有模板并取匹配分最高的那个,
所以遇到识别不准(找不到鱼漂,或者鼠标老是点偏)时,可以自己截一张当前场景的样本补进去:

1. 正常钓一竿,抛竿后脚本会把截图存到 `var/fishing_session.png`(每次都会覆盖,想保留现场
   的话记得赶紧手动复制一份出来,或者按上一节打开 `DEBUG_SNAPSHOTS` 让失败案例自动留档)
2. 用任意看图/截图工具打开这张图,把鱼漂那一小块区域(浮标本体,不用太大,参考现有的
   `var/fishing_float_1.png` / `fishing_float_2.png` 裁多大)单独裁出来另存
3. 存成 `var/fishing_float_<下一个数字>.png`(数字接着现有的往后排,不冲突就行)
4. 重新跑一次,不用改代码,`find_float()` 会自动把这张新模板也纳入比较

注意鱼漂只会出现在角色前方的水面这一片区域(`float_detector.py` 里
`FLOAT_SEARCH_X_RANGE`/`FLOAT_SEARCH_Y_RANGE` 定义的范围),截图时保证摄像机角度和平时钓鱼
差不多就行,不用整张图都很干净。

极暗的夜晚场景(尤其水面还带月光反光纹理的)目前是已知的识别薄弱点,现有的"形状匹配 + 颜色
门槛"这套逻辑在这类场景下经常连形状本身都会被水纹噪声干扰,不是加一张模板就能稳定解决的,
如果你经常在这类场景钓鱼,遇到问题多提供几张实际截图会比较有帮助。

## 运行测试

```bash
pytest
```

`tests/test_find_float.py`、`tests/test_audio_listener.py` 是自动化回归测试,不需要开游戏或
真实麦克风,改动识别/音频逻辑之后应该先跑一遍确认没有回归。

`tests/test_click.py`、`tests/test_listen.py` 是需要人工确认结果的手动诊断脚本,直接运行
查看输出即可:

- `test_click.py`:验证模拟右键点击能不能实际传到别的窗口/程序,主要用来排查 macOS 辅助功能
  权限有没有生效——如果这个脚本点了没反应,先去查权限设置,不用怀疑 `fishing.py` 的逻辑
- `test_listen.py`:验证音频回环设备能不能正常收到游戏声音、峰值/均值大概多少,配置
  BlackHole / VB-Cable 之后应该先跑这个,再去调 `listen()` 的 `threshold`
