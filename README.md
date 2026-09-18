# wowfishtool

魔兽世界自动钓鱼脚本(Python)。原理:抛竿后截图,用模板匹配找到鱼漂位置并把鼠标移过去;
监听系统音频,检测到持续的"上钩"声音后自动右键拉杆。

> macOS 是作者日常实测的平台;Windows 部分标了 ⚠️ 的地方是按原理写的配置说明,**没有在真实
> Windows 设备上跑过**(作者手头没有 Windows 机器)。`fishing.py` 里截图/找窗口/模拟点击这些
> 逻辑用的都是跨平台库,理论上能跑,遇到问题欢迎反馈。

## 安装

### 0. ⚠️ Windows 新手向导:电脑上完全没装过 Python 和 Git,推荐用 PyCharm

如果你是 Windows 用户,电脑上从没装过 Python、也没装过 Git,不想先去两边官网分别下载安装,
推荐只装 PyCharm 一个东西,剩下的交给它自动搞定。这条路径没在真实 Windows 设备上跑过,原理上
应该没问题,遇到跟下面描述对不上的地方欢迎反馈。

```text
# 1. 只装 PyCharm，先不管 Python 和 Git —— 后面两步会用 PyCharm 自带的功能补装
https://www.jetbrains.com/pycharm/download/  →  下载 Community（社区版，免费）→ 安装

# 2. 拿到项目代码（不用提前装 Git）
PyCharm 欢迎界面 → Get from VCS
  # 已经开着某个项目的话，菜单里是 File → New → Project from Version Control
URL 栏贴项目地址 → Clone
  # 系统里没有 Git 的话，PyCharm 会弹提示说找不到 Git 可执行文件，并给一个
  # "download and install" 按钮 —— 点它，PyCharm 自动下载安装 Git，装完再点一次 Clone
  # 不用自己去 Git 官网下载

# 嫌上面这步麻烦，或者公司网络限制导致自动下载 Git 失败，走这条备用路线：
浏览器打开项目的 GitHub 页面 → 绿色 Code 按钮 → Download ZIP → 解压到任意文件夹
PyCharm: File → Open → 选中解压出来的文件夹（包含 fishing.py 的那一层）
  # 这样完全不需要 Git

# 3. 装 Python 解释器（不用提前去 python.org 下载）—— 项目在 PyCharm 里打开之后
File → Settings → Project: <项目名> → Python Interpreter
Add Interpreter → Add Local Interpreter → Virtualenv Environment
  # Base interpreter 下拉框里如果没有能用的 Python 版本，旁边通常会有下载/获取新解释器的
  # 入口（不同 PyCharm 版本文案略有差异，一般是个 "Download" 按钮，或者下拉列表里直接
  # 列出可下载的 Python 版本），选 3.11 或 3.12
  #   —— 参考下面"环境准备"：版本太新容易遇到 pyaudio 还没发布对应预编译包的问题，
  #      不要选最新的 3.13+/3.14
点下载 → PyCharm 自动下载安装这个 Python 版本，不用去官网手动装
确认 Base interpreter 指向刚装好的版本，Location 保持默认（项目根目录下的 .venv）→ OK
  # 创建虚拟环境

# 4. 剩下的安装步骤（系统权限、音频回环设备、pip install -r requirements.txt）照后面
#    几节的说明做，凡是要求"打开终端敲命令"的地方，都用 PyCharm 底部的"终端"面板
#    （不是"Python 控制台"）—— 它会自动激活项目的虚拟环境
```

### 1. 环境准备(通用)

需要 Python 3.9+,**推荐用 3.11~3.13**。太新的版本(比如刚发布不久的 3.14)经常会遇到
某个依赖包还没来得及发布对应的预编译安装包,装的时候直接报编译失败——不是这个项目本身的
问题,是 Python 版本发布得比这些包的更新快。`pyaudio` 目前就是这样:官方已经给 3.13
发了 Windows 预编译包,但还没跟上 3.14,3.14 上 `pip install pyaudio` 会报找不到
`portaudio.h`,只能从源码编译,而编译又要求先手动装好 C++ 编译器和 PortAudio 库本身,
相当麻烦;[Python 官方论坛上已经有人反馈过一模一样的情况](https://discuss.python.org/t/pyaudio-wont-install-on-python-3-14-4-but-will-on-3-13-windows/107148),
最后是靠换回 3.13 解决的,没有更省事的办法。建议用虚拟环境:

**macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**

推荐直接用 PyCharm 创建虚拟环境,不用自己敲命令、也不用先去 python.org 下载——见上面
"0. Windows 新手向导"一节(`Add Interpreter → Add Local Interpreter → Virtualenv
Environment`),没装过 Python 的话 PyCharm 还能顺带帮你下载安装。

### 2. 系统权限(容易漏掉,漏了会表现成"脚本跑了但鼠标/键盘没反应")

脚本靠 `pyautogui`/`pynput` 模拟鼠标键盘、靠截图识别画面,这两件事在 macOS 上都需要单独授权,
而且授权对象是**实际运行 Python 的那个程序**(终端、PyCharm 或其他 IDE),不是 Python 本身:

```text
# macOS
系统设置 → 隐私与安全性 → 辅助功能 → 加入并勾选"终端"/"PyCharm"   # 控制鼠标键盘要用
系统设置 → 隐私与安全性 → 屏幕录制 → 加入并勾选"终端"/"PyCharm"   # 截图要用，macOS 10.15+
                                                                #  强制要求，不给权限
                                                                #  会黑屏/全黑截图
重启终端/IDE
  # 改动权限后必须重启才会生效，改完立刻测试大概率还是不行，先重启一遍

# 不确定权限生效没有：跑 tests/test_click.py（见下方"运行测试"）单独验证鼠标点击能不能
# 正常传到别的窗口，不用开游戏
```

如果 WoW 是以管理员身份运行的,模拟鼠标键盘的进程权限必须**跟游戏一致**,否则 Windows 的
UIPI(用户界面特权隔离)会静默丢弃"权限低的进程发给权限高的窗口"的输入事件,表现就是脚本日志
一切正常、但游戏里毫无反应:

```text
# Windows —— 二选一，不要一边有一边没有
终端/IDE 和 WoW 都右键"以管理员身份运行"
# 或者
两边都不用管理员权限运行
```

### 3. 配置音频回环设备(用来"听"上钩声音)

脚本靠监听系统音频里的上钩声音来判断什么时候拉杆,所以需要一个能把游戏声音路由进来的
虚拟回环录音设备,而不是对着麦克风收环境噪音。

**先调一下游戏内的声音设置(通用,跟系统无关)**,不然回环设备装得再对也可能听不清:

```text
游戏选项 → 声音 → 音效（Sound Effects）音量调大于 0
  # 上钩的"啵"一声走的就是这个通道，关掉或调到 0 脚本就永远听不到
游戏选项 → 声音 → 音乐/环境声音 调低或关闭
  # 背景音乐、风声水声这类持续性声音会垫高录到的背景噪音基准线，让上钩那一下没那么好分辨，
  # listen() 的 threshold 更难调准
游戏选项 → 声音 → 关闭"后台/非焦点窗口时降低音量"（不同客户端版本叫法可能不太一样，自己找一下）
  # ⚠️ 重点：这个功能是设计给你切出去挂着的，一旦触发，游戏会把音量压得很低甚至静音，
  # 回环设备录到的上钩声音也会跟着变小到过不了 threshold —— 脚本会一直"没听到咬钩"，
  # 但看起来什么都正常
  # 同理：如果开着 Discord 之类软件，也检查它有没有"检测到别人说话就压低其他程序音量"的
  # 选项（通常叫"衰减"/attenuation），开着的话一样会把游戏声音压下去
```

**macOS:安装 [BlackHole](https://github.com/ExistentialAudio/BlackHole)**

```bash
brew install blackhole-2ch
```

```text
启动台 → 其他 → 音频 MIDI 设置 → 左下角 "+" → 新建"多输出设备"
勾选 BlackHole 2ch + 你平时用的扬声器/耳机
  # 这样声音会同时发到两边，自己也能听到游戏声音，不是静音操作
系统设置 → 声音 → 输出 → 选中刚建好的这个多输出设备
魔兽世界 → 声音设置 → 确认输出没有被游戏自己锁定到别的设备上
  # 部分游戏会记住上次用的输出设备，需要在游戏内选项里也确认一下

# audio_listener.py 默认会自动找系统里名字带 "blackhole" 的输入设备，一般不用改代码
# 如果你的设备名不带这几个字（比如用了别的中文/自定义命名），把 LOOPBACK_NAME_HINT
# 改成实际能匹配到的关键字
```

**Windows:BlackHole 不支持 Windows** ⚠️,需要装一个功能类似的虚拟声卡:
[VB-Audio Virtual Cable](https://vb-audio.com/Cable/)(免费,最简单,唯一实测过的方案)

```text
# ⚠️ 只装这一个，不要额外再装别的虚拟声卡（VoiceMeeter、"VB-Audio Point" 之类
# VB-Audio 家其他产品——没测过，不保证能用）。装多个虚拟声卡之后，Windows 里会同时
# 出现好几个名字都带 "CABLE" 的录音设备，而下面的自动匹配是按设备编号从小到大挑第一个
# 名字匹配的，装了不止一个的话很可能挑到的不是你系统输出实际接的那一个，到时候人听得到
# 游戏声音、脚本却收不到，还不容易看出是这个原因。已经装了不止一个的话，把用不上的从
# "设置 → 应用"里卸载掉，只留 VB-Cable 这一个

右键 VBCABLE_Setup_x64.exe → 以管理员身份运行
  # 装虚拟声卡驱动必须用管理员权限，否则装了但系统里找不到设备
重启电脑
  # 虚拟声卡驱动通常要求重启才会生效
Windows 设置 → 系统 → 声音 → 输出设备 → 选 "CABLE Input"
  # 这样一来自己就听不到游戏声音了，想同时听到：
  声音控制面板 → CABLE Input → 侦听 → 勾选"通过此设备播放" → 指向实际耳机/音箱
魔兽世界 → 声音设置 → 输出设备也选 "CABLE Input"
  # 或者干脆用系统默认，只要系统默认就是 CABLE Input

# audio_listener.py 会自动按系统类型选关键字，Windows 下默认就是找名字带 "cable" 的
# 录音设备，一般不用改代码；只有你的设备名不带这几个字（比如自己改过设备命名），
# 才需要把 LOOPBACK_NAME_HINT 改成实际能匹配到的关键字
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

> ⚠️ **用 PyCharm 之类 IDE 的话,运行报 `ModuleNotFoundError: No module named 'xxx'`
> 十有八九是"装的地方"和"跑的地方"不是同一个 Python 环境。** 常见情况:开了终端手动建了
> `.venv` 并且 `pip install` 成功了,但 PyCharm 的运行配置(右上角/右下角显示的解释器)
> 用的是另一个解释器(比如 PyCharm 自动创建的默认 venv,或者系统全局 Python),`pip
> install` 装的包压根不在它能找到的地方。排查方法:PyCharm 右下角状态栏点一下当前解释器
> 名字,确认它显示的路径(比如 `.venv\Scripts\python.exe`)跟你执行 `pip install -r
> requirements.txt` 时终端里 `.venv` 是不是同一个;不确定的话直接在 PyCharm 底部的
> "终端" 面板里(它默认会自动激活项目配置的解释器)重新跑一遍
> `pip install -r requirements.txt` 最保险。

## 使用

### 用 PyCharm 运行(Windows)

```text
文件 → 打开 → 选中这个项目文件夹（包含 fishing.py 的那一层）

文件 → 设置 → 项目: wow-fishing → Python 解释器
  # 确认右上角下拉框显示的是项目里的虚拟环境（形如 ...\wow-fishing\.venv\Scripts\python.exe），
  # 不是系统全局 Python；列表里没有的话，点"添加解释器" → "Virtualenv 环境"，
  # 指向项目根目录下的 .venv 文件夹（前提是已经按"环境准备"那步建好了）

PyCharm 底部"终端"面板 → pip install -r requirements.txt
  # 是标签写着"终端"的那个面板（不是"Python 控制台"/"Python 进程输出"），它默认会
  # 自动激活项目的虚拟环境；如果 PyCharm 自己弹出"检测到 requirements.txt，是否安装"
  # 的提示，点安装也可以，效果一样

左侧项目树双击 fishing.py → 点编辑器行号左边的绿色三角形（或右键文件 → 运行 'fishing'）
  # PyCharm 会在底部"运行"面板里启动脚本 —— 这个面板只用来看输出，不能在里面手动敲命令，
  # 跟"终端"面板是两回事，不要混

# 看到运行面板打印出 "Press F10 to start fishing, F11 to stop" 之后，切回游戏窗口，
# 按 F10/F11 操作 —— 热键是系统级全局监听，不需要 PyCharm 窗口在前台
```

macOS 用户 / 习惯直接用命令行的可以跳过上面这节,参考下面的通用步骤。

### 命令行运行(通用 / macOS)

1. 打开魔兽世界,**建议把画质调到 4 档左右,关掉水面反射/镜面效果**——画质越高,水面的
   环境反射越强,会把周围场景的颜色"染"到鱼漂本体上(尤其黄昏/夜晚这种有色调滤镜的场景),
   鱼漂颜色跟水面颜色混到分不清,点击定位会掉精度。画质调低之后水面反射弱,鱼漂本身的颜色
   更干净,识别更稳
2. **把镜头拉到最近,拉到看不见自己人物为止**——鱼漂落点在角色正前方不远的水面上(见
   `float_detector.py` 里 `FLOAT_SEARCH_X_RANGE`/`FLOAT_SEARCH_Y_RANGE` 那片区域),镜头
   稍微拉远一点,自己的人物/坐骑模型就会挡在这片区域前面,把鱼漂整个或者部分遮住,前面
   所有识别逻辑都无从谈起。拉到最近(接近第一人称、看不到自己人物)能保证这片水面完全空出来
3. 把钓鱼快捷键放在快捷栏第 1 格(对应 `fishing.py` 里的 `CAST_KEY = '1'`,改了快捷栏位置
   要同步改这个常量)
4. 运行:

   ```bash
   python fishing.py
   ```

   脚本启动后会先检查 WoW 进程是否在跑,不在的话会直接退出,提示先打开游戏
5. 切回游戏窗口(脚本会提示"等 2 秒切窗口",利用这个时间点回游戏),按 **F10** 开始自动
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
**2** 上:

```
/use 鱼饵的名字
/use 16
```

鱼竿是双手武器,只能放在主手栏位(装备栏第 16 格),不能放副手,所以 `/use 16` 不用改。
脚本会在开始钓鱼时先按一次 `2`,之后每隔 `fishing.py` 里
`BAIT_REAPPLY_INTERVAL_SECONDS`(默认 10 分钟多一点,故意留了几秒余量避免鱼饵还没到期就被
提前刷新)自动再按一次,不用手动操作。如果快捷键不是 2,改 `fishing.py` 里的 `BAIT_KEY`。

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

极暗的夜晚场景(尤其水面还带月光反光纹理的)曾经是识别薄弱点——水面波纹本身的饱和度经常比
鱼漂本体还高,把颜色门槛的"鱼漂必须比水更艳"这条假设直接击穿,导致鱼漂明明在画面里清晰可见
却一直报"没找到"。现在遇到这种颜色门槛在整个搜索范围内都分不出任何东西的情况(不只是鱼漂,
连水面噪声都清一色被判定"不够艳"),会自动放弃颜色门槛、只信形状匹配的分数,跟"水面饱和度顶到
HSV 上限"那种更极端的场景用的是同一套降级逻辑。如果还是遇到识别不准,多半是形状匹配本身分数
不够(参考上一节补模板),而不是颜色门槛的问题了;遇到问题多提供几张实际截图仍然有帮助。

### 定位点到了角色状态框(头像/血条/蓝条)上怎么回事

`float_detector.py` 里的颜色门槛专挑"比水面更艳"的像素,而角色自己的状态框(金色描边的
头像图标、绿色血条、蓝色蓝条)正好就是又艳又亮,一旦上面那条"极暗夜晚"薄弱点命中——真正的
鱼漂因为太暗/太不显眼被颜色门槛刷掉——算法会转而在剩下能通过颜色门槛的区域里挑分数最高的,
状态框就有可能顶上来,让脚本很confident 地点到状态框而不是鱼漂上。

已经把两个已知会撞上这个问题的位置从搜索范围里显式排除掉了(`float_detector.py` 里的
`UI_EXCLUDE_REGIONS`,按屏幕宽高的百分比定义):

1. 一直常驻显示的默认角色状态框(左下角,不管用哪套 Edit Mode 布局都在)
2. 现在 WoW 默认 Edit Mode「Modern」布局会在右下角额外摆一份同样内容的状态框副本

这两个区域是按一次 2560×1410 的实际截图量出来的,已经留了余量,但如果你的分辨率/UI 缩放/
状态框摆放位置跟这次实测的不一样,不排除还是会被别的 UI 元素撞上——遇到"明明画面里有鱼漂,
点击却落在自己头像/血条附近"这种情况,可以打开 `DEBUG_SNAPSHOTS`(见上面"调试模式"),从存
下来的 `debug/fallback_*.png`(带绿框+红点)或者干脆一张普通截图里量出那个 UI 元素的像素范围,
换算成宽高百分比后加进 `UI_EXCLUDE_REGIONS` 里(格式是 `((x0, x1), (y0, y1))`,数值都是
0~1 之间的屏幕宽/高占比)。

## 运行测试

```bash
pytest
```

`tests/test_find_float.py`、`tests/test_audio_listener.py` 是自动化回归测试,不需要开游戏或
真实麦克风,改动识别/音频逻辑之后应该先跑一遍确认没有回归。

`tests/test_click.py`、`tests/test_listen.py` 是需要人工确认结果的手动诊断脚本,**当成普通
Python 脚本运行**(`python tests/test_click.py`),不是自动化测试,里面没有 `test_` 开头的
用例函数。用 PyCharm 右键运行的话,注意选"运行 'test_click'"这个选项,不要选成"运行 pytest
in test_click.py"——PyCharm 有时候看到 `tests/` 目录会默认建议用 pytest 方式运行,选错了会
显示"collected 0 items",什么都不会执行,不是脚本本身有问题。直接运行查看输出即可:

- `test_click.py`:验证模拟右键点击能不能实际传到别的窗口/程序,主要用来排查 macOS 辅助功能
  权限有没有生效——如果这个脚本点了没反应,先去查权限设置,不用怀疑 `fishing.py` 的逻辑
- `test_listen.py`:验证音频回环设备能不能正常收到游戏声音、峰值/均值大概多少,配置
  BlackHole / VB-Cable 之后应该先跑这个,再去调 `listen()` 的 `threshold`
