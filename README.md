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

## 运行测试

```bash
pytest
```

`tests/test_find_float.py`、`tests/test_audio_listener.py` 是自动化回归测试,不需要开游戏或真实麦克风。
`tests/test_click.py`、`tests/test_listen.py` 是需要人工确认结果的手动诊断脚本,直接运行查看输出即可。
