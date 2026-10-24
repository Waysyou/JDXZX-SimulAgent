# 本地模型获取与路径

这里只给出原始来源和结构要求，不包含权重或自动下载脚本。请先核对来源、版本及你实际使用场景的许可。模型在自己的电脑准备，不要把官方赛事数据上传到下载或云推理服务。

## Sherpa Streaming Zipformer

兼容目录名：`sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30`。

来源：https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models 。对应包名为 `sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30.tar.bz2`。

实际推理需要 `encoder.int8.onnx`、`decoder.onnx`、`joiner.int8.onnx` 和 `tokens.txt`，不要拿其他架构文件混用。保留原包 README 和许可说明。上游模型信息：https://huggingface.co/yuekai/icefall-asr-multi-zh-hans-zipformer-large 。

## Qwen

来源：https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct 。下载该版本完整的配置、tokenizer、权重及 LICENSE。至少需要 `config.json`、`tokenizer.json`、`tokenizer_config.json` 和全部 safetensors 分片及其索引（如果采用分片版本）。不要只下载权重而遗漏 tokenizer。

Qwen 在本机通过 Transformers 推理，不调用在线翻译 API。默认使用本地文件，缺少资源时不会自动下载。

## Piper 音色与 eSpeak

兼容的原版音色为 `en_US-lessac-medium.onnx` 和 `en_US-lessac-medium.onnx.json`，来源目录：https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium 。本候选工程不分发它们。模型卡引用的独立数据条款：https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/license.html 。不要从软件 GPLv3 推导权重或训练数据也有完全相同许可，亦不要据本说明推定任何用途都已获准。

可以通过 `PIPER_VOICE_MODEL` 指定自己有权使用的兼容英文音色 ONNX，保留对应 JSON 并核对采样率。更换音色后需要另行听测质量和播报速度。

启动入口可以检测**当前环境安装的 Piper 包**内 `espeak-ng-data`。如你的安装渠道未包含这些资源，需按 Piper 上游文档取得合规资源，配置 `PIPER_ESPEAK_DATA_DIR`，目录中应有 `phontab` 等完整数据。不要只为通过检查而创建空的 phontab 文件。

## 默认结构

```text
.cache/models/
  sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30/
    encoder.int8.onnx
    decoder.onnx
    joiner.int8.onnx
    tokens.txt
  qwen2.5-1.5b-instruct/
    config.json
    tokenizer.json
    tokenizer_config.json
    model.safetensors
    ...原始模型附属文件...
  piper-en_US-lessac-medium/
    en_US-lessac-medium.onnx
    en_US-lessac-medium.onnx.json
```

也可通过 `config/local_settings.json` 或同名环境变量使用显式绝对路径，不必把模型复制到工程里。普通启动没有“借用父工程模型”的隐藏回退。

`run_local.py --check` 仅做文件存在和可写性检查，不读取、加载、联网校验或验证权重哈希。正式使用时需要再检查文件来源、完整性，并用自己有权使用的音频做真实推理。
