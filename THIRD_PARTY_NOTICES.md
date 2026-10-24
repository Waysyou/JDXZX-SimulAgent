# 第三方组件与来源

该候选工程只包含团队编排代码与界面源码，不包含模型权重、音色权重、Python 环境、官方比赛数据或其他项目的整套代码。

| 组件 | 在本工程中的作用 | 原始来源与许可边界 |
|---|---|---|
| Sherpa-ONNX 1.13.4 | 运行原生流式中文识别 | https://github.com/k2-fsa/sherpa-onnx ，引擎代码 Apache-2.0，副本见 docs/licenses/SHERPA_ONNX_CODE_LICENSE.txt |
| 中文 Streaming Zipformer INT8 模型 | 音频到中文 | 来源见 docs/MODELS.md，不能用引擎许可替代权重及上游数据许可 |
| Qwen2.5-1.5B-Instruct | 中文片段到英文 | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct ，仓库标示 Apache-2.0，取得权重时保留对应 LICENSE 和模型卡 |
| Piper TTS 1.5.0 | 英文语音合成 | https://github.com/OHF-Voice/piper1-gpl ，软件 GPLv3 副本见 docs/licenses/PIPER_SOFTWARE_COPYING.txt；音色许可独立核对 |
| en_US-lessac-medium（兼容配置示例） | 英文音色 | https://huggingface.co/rhasspy/piper-voices ，模型卡副本见 docs/licenses/PIPER_LESSAC_MODEL_CARD.txt，不随本工程分发权重 |
| Transformers、PyTorch、Gradio、FastAPI、Uvicorn、NumPy、SciPy、SoundFile、OpenCC、SentencePiece、Safetensors 等 | 模型运行、界面与处理 | 具体安装版本见 requirements 文件，按上游各自许可使用，安装依赖时保留其许可文件 |

稳定前缀策略借鉴公开的 WhisperStreaming / LocalAgreement 思路：https://github.com/ufal/whisper_streaming 。该方法不宣称由本队首创；本工程使用自己的流式状态与提交实现。

当前项目代码许可仍未选定，见 LICENSE_STATUS.md。此表是来源清单，不构成对第三方权重、训练语料或任意商业用途的额外授权。
