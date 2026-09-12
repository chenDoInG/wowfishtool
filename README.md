# wowfishtool

魔兽世界自动钓鱼脚本(Python)。原理:抛竿后截图,用模板匹配找到鱼漂位置并把鼠标移过去;
监听系统音频,检测到持续的"上钩"声音后自动右键拉杆。

## 安装

### 1. 环境准备

需要 Python 3.9+。

**macOS**

`pyaudio` 依赖系统的 `portaudio` 库,先装好再 `pip install`:

```bash
brew install portaudio
```

**Windows**

`pyaudio` 在 Windows 上一般直接 `pip install` 就能装上预编译好的 wheel,不需要额外装 `portaudio`。

### 2. 安装 Python 依赖

建议用虚拟环境:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows 用 .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 配置音频回环设备(用来"听"上钩声音)

脚本靠监听系统音频里的上钩声音来判断什么时候拉杆,所以需要一个能把游戏声音路由进来的
虚拟回环录音设备,而不是对着麦克风收环境噪音。

**macOS:安装 [BlackHole](https://github.com/ExistentialAudio/BlackHole)**

```bash
brew install blackhole-2ch
```

装完之后:

1. 打开「音频 MIDI 设置」,新建一个「多输出设备」,勾选 BlackHole 2ch 和你平时用的扬声器/耳机
   (这样声音会同时发到两边,你自己也能听到游戏声音)
2. 系统设置 → 声音 → 输出,选中这个多输出设备
3. `audio_listener.py` 里默认会自动找名字里带 "blackhole" 的输入设备,不用改代码

**Windows:BlackHole 不支持 Windows**,需要装一个功能类似的虚拟声卡,推荐二选一:

> ⚠️ 下面的 Windows 部分只是按原理给出的配置说明,**没有在真实 Windows 设备上测试过**
> (作者手头没有 Windows 机器)。`fishing.py` 里截图/找窗口/模拟点击这些逻辑本身用的都是
> 跨平台库,理论上能跑,但没实测验证过,如果遇到问题欢迎反馈。

- [VB-Audio Virtual Cable](https://vb-audio.com/Cable/)(免费,最简单)
- [VoiceMeeter](https://vb-audio.com/Voicemeeter/)(功能更全,免费)

装好并把 WoW 的输出路由过去之后,还要把 `audio_listener.py` 里的 `LOOPBACK_NAME_HINT`
改成对应设备的名字关键字,比如用 VB-Cable 就改成:

```python
LOOPBACK_NAME_HINT = 'cable'
```

配置完用 `tests/test_listen.py` 实测一下,确认游戏声音能被正常录到、峰值/均值大概是多少,
再照实际数值调 `fishing.py` 里 `listen()` 的 `threshold`。

## 使用

1. 打开魔兽世界,把钓鱼快捷键放在快捷栏第 1 格
2. 运行:

   ```bash
   python fishing.py
   ```

3. 切回游戏窗口,按 **F10** 开始自动钓鱼,按 **F11** 停止

### 自动上鱼饵(可选)

如果用的鱼饵有持续时间/次数限制,需要定期重新上饵,可以在游戏里做一个宏绑在快捷键
**2** 上(鱼竿在主手栏位的话):

```
/use 鱼饵的名字
/use 16
```

脚本会在开始钓鱼时先按一次 `2`,之后每隔 `fishing.py` 里
`BAIT_REAPPLY_INTERVAL_SECONDS`(默认 10 分钟)自动再按一次,不用手动操作。如果你的鱼竿在
副手栏位,把宏里的 `/use 16` 改成 `/use 17`;如果快捷键不是 2,改 `fishing.py` 里的
`BAIT_KEY`。

## 鱼漂识别不准怎么办

不同天气、水域、光照下鱼漂的颜色/水面观感差别很大,`var/` 目录下自带的模板不一定每种场景都
适用。`find_float()` 会自动加载 `var/fishing_float_*.png` 下所有模板并取匹配分最高的那个,
所以遇到识别不准(找不到鱼漂,或者鼠标老是点偏)时,可以自己截一张当前场景的样本补进去:

1. 正常钓一竿,抛竿后脚本会把截图存到 `var/fishing_session.png`(每次都会覆盖)
2. 用任意看图/截图工具打开这张图,把鱼漂那一小块区域(浮标本体,不用太大,参考现有的
   `var/fishing_float_1.png` / `fishing_float_2.png` 裁多大)单独裁出来另存
3. 存成 `var/fishing_float_3.png`(数字接着现有的往后排,不冲突就行)
4. 重新跑一次,不用改代码,`find_float()` 会自动把这张新模板也纳入比较

注意鱼漂只会出现在角色前方的水面这一片区域(`fishing.py` 里
`FLOAT_SEARCH_X_RANGE`/`FLOAT_SEARCH_Y_RANGE` 定义的范围),截图时保证摄像机角度和平时钓鱼
差不多就行,不用整张图都很干净。

## 运行测试

```bash
pytest
```

`tests/test_find_float.py`、`tests/test_audio_listener.py` 是自动化回归测试,不需要开游戏或真实麦克风。
`tests/test_click.py`、`tests/test_listen.py` 是需要人工确认结果的手动诊断脚本,直接运行查看输出即可。
