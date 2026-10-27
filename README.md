# 进点双子星 · 本地语音同传工作台 V2

面向中文单人演讲的本地原型：麦克风或 WAV 输入，逐段识别中文、生成英文翻译和英文语音，支持对应片段回听与结果保存。

**当前状态：发布候选工程，按 Waysyou 的要求先作为私有仓库保存，尚未公开发布。队友已同意项目公开，队员署名与账号核验状态见 [CONTRIBUTORS.md](CONTRIBUTORS.md)；公开范围与项目代码许可证仍待确认。** 这不是一个已经获得任意使用授权的开源发行版，见 [LICENSE_STATUS.md](LICENSE_STATUS.md)。

## 版本关系

本工程来自参赛第一版之后继续开发的第二版，保留当前 V4 接入前端和后端。原比赛提交、审核和获奖对应第一版，不能用本工程代替原提交版进行成绩复现。本工程不包含官方数据、标注、比赛结果、演示视频、模型权重或其他电脑的虚拟环境。

此候选版只整理安装、路径、文档和测试素材，不重训模型，不改主识别/翻译/播放逻辑和前端布局。开发集场景专用纠错配置不随包提供，翻译词表改为空模板。因配置边界不同，不承诺与原工程逐字、逐段或时延一致。

## 能做什么

- 麦克风持续输入，或把用户自有 WAV 按时间逐段送入。
- 中文识别和英文翻译逐段显示。
- 对已生成语音的英文片段进行回听，连续语音按队列播放。
- 保存运行产生的翻译文本和英文语音。
- 使用当前界面的主题、语言等设置。

中文识别使用 Sherpa-ONNX 的 Streaming Zipformer，翻译使用 Qwen2.5-1.5B-Instruct，英文语音合成使用 Piper。基础模型不是本团队训练；模型来源与第三方许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [docs/MODELS.md](docs/MODELS.md)。

## 安装前先了解

- 当前验证基线：Windows x64、Python 3.13 x64。其他系统未验证。
- 请完整解压到当前用户可写的目录，不要在压缩软件内直接运行，也不需要管理员身份。
- 建议使用较短的英文路径，例如 `D:\Projects\SimulAgentV2`。首次安装需要联网下载依赖，准备足够磁盘空间。
- 已有开发环境使用过 RTX 4060 Laptop 8GB；这只是已用设备，不是保证所有笔记本都能实时运行的最低配置。
- 默认安装 CPU PyTorch 是为了提供明确的安装路径，不保证 CPU 能达到实时处理速度。NVIDIA GPU 用户可以按驱动条件选择 CUDA 12.8 对应安装渠道。
- 模型、依赖及授权音频由使用者准备。本候选工程不会在启动时自动下载模型。

## 1. 安装依赖

在本目录打开 PowerShell，任选一种方式。

CPU 路径：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\setup_runtime.ps1 -TorchBackend cpu
```

NVIDIA CUDA 12.8 路径：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\setup_runtime.ps1 -TorchBackend cuda128
```

也可以运行 `Setup.cmd`，默认选择 CPU 路径。脚本在**本目录**建立 `.venv`，安装命令失败会立即停止，不借用上级目录环境。依赖和临时下载缓存写入本目录 `.cache`。依赖版本来自既有 V2 环境，尚不表示在任意新电脑上均完成安装验证。GPU 驱动兼容性按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 核对。

## 2. 准备模型

按 [docs/MODELS.md](docs/MODELS.md) 准备已有授权的模型，默认位置在 `.cache/models`。也可以复制配置模板并填写自己电脑上的模型路径：

```powershell
Copy-Item .\config\local_settings.example.json .\config\local_settings.json
```

`local_settings.json` 不应提交到仓库。优先级为：显式环境变量、此本地 JSON、工程默认值。相对模型路径以工程根目录为准，不自动搜索父目录。Piper eSpeak 数据优先读取显式配置或本工程资源，否则检测当前已安装 Piper 包内的数据目录。

资源检查不加载模型：

```powershell
.\.venv\Scripts\python.exe -X utf8 run_local.py --check
```

缺少模型时会以退出码 `2` 列出缺失项，这是纯源码工程的预期情况。检查通过只说明必要文件存在，不证明权重有效或质量达标。

## 3. 启动

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\start.ps1
```

或者运行 `Start.cmd`。默认只监听 `127.0.0.1`，从 7861–7869 选空闲端口，打开脚本打印的地址。先启动模型，再选择麦克风或上传自有 WAV。不要直接双击前端 HTML 来测试后端；文件协议下保留的是样板行为，不代表真实推理。

如果只检查前端和 API 连接、暂时没有模型：

```powershell
.\.venv\Scripts\python.exe -X utf8 run_local.py --ui-only --no-browser
```

此模式可以展示界面与缺失模型状态，不能进行真实同传。正常退出按一次 `Ctrl+C`。

## 测试

无需加载权重的单元测试：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

测试使用临时文件、人工编写的短文本、合成静音和替身组件，不使用赛事音频或参考答案。单元测试通过不等于识别准确率、翻译质量或真实延迟测试通过。

自有音频的结果生成入口：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/generate_outputs.py your_own_speech.wav --output-dir outputs/own-demo
```

输入内容的使用与公开须由使用者获得授权。`outputs`、缓存和日志默认被忽略，但 `.gitignore` 不能清除已经提交的历史。

## 已知限制

- 口音、噪声、专有词、断句以及长时间运行可能影响质量。
- 0.4 秒为音频块输入节奏，不是端到端翻译延迟。
- 稳定前缀策略有助于控制提交时机，不保证识别内容正确。
- 播报队列有容量限制，积压时可能缺少实时语音，不能承诺任意语速下完整实时播报。
- 原 V2 在翻译多次失败后包含保守英文兜底句，此类输出不应当作准确译文。
- 没有提供生产部署所需的账号认证、远程访问保护和严格质量评估，不要直接作为公网服务部署，也不要用于医疗、法律等高风险决策。

## 团队与发布

项目团队：进点双子星队。当前已确认的分工与其余成员署名状态见 [CONTRIBUTORS.md](CONTRIBUTORS.md)。

本候选工程已保存到 [Waysyou/JDXZX-SimulAgent](https://github.com/Waysyou/JDXZX-SimulAgent) 私有仓库，尚未公开发布。真正公开前，请完成 [docs/RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md) 并确认正式许可证。暂不提供官方比赛数据或结果的公开下载链接。
