(() => {
  "use strict";

  const STORAGE_KEY = "jdxzx-simulagent-v4-settings";
  const defaults = Object.freeze({
    theme: "system",
    language: "zh-CN",
    defaultInput: "mic",
    sourceVolume: 100,
    ttsVolume: 100,
    autoplay: true,
    subtitleScale: 100,
  });

  const messages = {
    "zh-CN": {
      "nav.live": "实时工作台",
      "nav.results": "结果回放",
      "nav.system": "系统状态",
      "nav.settings": "设置",
      "settings.title": "设置",
      "settings.subtitle": "调整界面、输入方式与本地播放偏好。",
      "settings.appearance": "外观与语言",
      "settings.appearanceHint": "选择显示主题和工作台界面语言。",
      "settings.theme": "显示主题",
      "settings.themeLight": "浅色",
      "settings.themeDark": "深色",
      "settings.themeSystem": "跟随系统",
      "settings.language": "界面语言",
      "settings.languageHint": "只改变界面文字，不改变中文译英文的任务方向。",
      "settings.audio": "声音与播放",
      "settings.audioHint": "音量只作用于本浏览器，不修改导出的 WAV 文件。",
      "settings.sourceVolume": "源演讲音量",
      "settings.ttsVolume": "英文同传音量",
      "settings.autoplay": "自动播放英文语音",
      "settings.autoplayHint": "稳定片段生成后自动进入连续播放队列。",
      "settings.workflow": "工作台偏好",
      "settings.workflowHint": "设置每次打开页面时采用的默认状态。",
      "settings.defaultInput": "默认输入方式",
      "settings.inputMic": "麦克风",
      "settings.inputFile": "WAV 文件",
      "settings.subtitleSize": "字幕字号",
      "settings.sizeCompact": "紧凑",
      "settings.sizeStandard": "标准",
      "settings.sizeLarge": "较大",
      "settings.local": "所有设置仅保存在当前浏览器",
      "settings.localHint": "不会上传账号、音频或比赛数据，也不会改变模型参数。",
      "settings.reset": "恢复默认设置",
      "settings.saved": "设置已保存",
      "settings.resetDone": "已恢复默认设置",
      "settings.current": "当前生效",
      "hero.title": "让英文，跟上演讲。",
      "hero.description": "选择麦克风或 WAV 流式推送；中文稳定片段确认后，才进入翻译和英文语音队列。",
      "input.label": "输入方式",
      "input.mic": "麦克风实时输入",
      "input.file": "上传 WAV 流式推送",
      "metrics.latency": "当前延迟",
      "metrics.stable": "稳定片段",
      "metrics.queue": "语音队列",
      "metrics.input": "输入状态",
      "metrics.latencyHint": "音频到英文首响",
      "metrics.stableHint": "已进入翻译队列",
      "metrics.queueHint": "有界队列，不覆盖",
      "metrics.inputHint": "麦克风尚未启动",
      "semantic.title": "实时语义对照",
      "semantic.hint": "中文稳定后才进入英文输出；点击已有英文行可单句回放。",
      "semantic.waiting": "等待输入",
      "semantic.time": "时间",
      "semantic.chinese": "中文 · 已确认",
      "semantic.english": "English · 点击本行播放",
      "semantic.segmentId": "片段 ID",
      "semantic.empty": "启动同传后，稳定的中英文片段会显示在这里",
      "semantic.details": "识别详情",
      "audio.title": "连续英文同传语音",
      "audio.hint": "播放片段与上方英文字幕同步高亮。",
      "audio.waiting": "等待播放",
      "audio.current": "当前英文：",
      "audio.none": "尚无英文语音",
      "queue.title": "播放队列",
      "queue.waiting": "等待英文语音",
      "queue.idle": "空闲",
      "results.title": "结果回放",
      "system.title": "系统状态",
      "phase.idle.button": "启动模型",
      "phase.idle.note": "依次载入 ASR · 翻译 · TTS",
      "phase.idle.status": "等待输入 · 请先启动模型",
      "phase.idle.state": "待机",
      "phase.loading.button": "正在启动",
      "phase.loading.note": "正在载入本地模型……",
      "phase.loading.status": "模型预热中 · 请稍候",
      "phase.loading.state": "启动中",
      "phase.ready.button": "开始同传",
      "phase.ready.note": "模型就绪 · 点击开始",
      "phase.ready.status": "模型已就绪 · 可以开始同传",
      "phase.ready.state": "就绪",
      "phase.live.button": "停止同传",
      "phase.live.note": "正在接收并处理音频",
      "phase.live.status": "同传进行中 · 中文稳定后输出英文",
      "phase.live.state": "监听中",
      "phase.stopping.button": "正在结束",
      "phase.stopping.note": "正在提交最后的稳定片段……",
      "phase.stopping.status": "正在结束本次同传",
      "phase.stopping.state": "收尾中",
      "phase.error.button": "重试启动",
      "phase.error.note": "本地服务发生错误",
      "phase.error.status": "连接异常 · 请检查终端",
      "phase.error.state": "异常",
    },
    en: {
      "nav.live": "Live workspace",
      "nav.results": "Results",
      "nav.system": "System status",
      "nav.settings": "Settings",
      "settings.title": "Settings",
      "settings.subtitle": "Adjust the interface, default input, and local playback preferences.",
      "settings.appearance": "Appearance & language",
      "settings.appearanceHint": "Choose the display theme and interface language.",
      "settings.theme": "Theme",
      "settings.themeLight": "Light",
      "settings.themeDark": "Dark",
      "settings.themeSystem": "System",
      "settings.language": "Interface language",
      "settings.languageHint": "This changes interface text only. Translation remains Chinese to English.",
      "settings.audio": "Audio & playback",
      "settings.audioHint": "Volume applies only to this browser and does not alter exported WAV files.",
      "settings.sourceVolume": "Source speech volume",
      "settings.ttsVolume": "English interpretation volume",
      "settings.autoplay": "Autoplay English speech",
      "settings.autoplayHint": "Play stable translated segments continuously as they are generated.",
      "settings.workflow": "Workspace preferences",
      "settings.workflowHint": "Choose the default state used whenever the page opens.",
      "settings.defaultInput": "Default input",
      "settings.inputMic": "Microphone",
      "settings.inputFile": "WAV file",
      "settings.subtitleSize": "Subtitle size",
      "settings.sizeCompact": "Compact",
      "settings.sizeStandard": "Standard",
      "settings.sizeLarge": "Large",
      "settings.local": "All settings stay in this browser",
      "settings.localHint": "No account data, audio, or competition material is uploaded, and model parameters are unchanged.",
      "settings.reset": "Reset defaults",
      "settings.saved": "Settings saved",
      "settings.resetDone": "Default settings restored",
      "settings.current": "Active",
      "hero.title": "Let English keep pace.",
      "hero.description": "Use a microphone or stream a WAV file. Only stable Chinese segments enter translation and the English speech queue.",
      "input.label": "Input",
      "input.mic": "Live microphone",
      "input.file": "Stream WAV file",
      "metrics.latency": "Current latency",
      "metrics.stable": "Stable segments",
      "metrics.queue": "Speech queue",
      "metrics.input": "Input state",
      "metrics.latencyHint": "Audio-to-English first response",
      "metrics.stableHint": "Committed to translation",
      "metrics.queueHint": "Bounded queue, no overlap",
      "metrics.inputHint": "Microphone not started",
      "semantic.title": "Live semantic alignment",
      "semantic.hint": "English is emitted only after Chinese becomes stable. Click a row to replay it.",
      "semantic.waiting": "Waiting for input",
      "semantic.time": "Time",
      "semantic.chinese": "Chinese · confirmed",
      "semantic.english": "English · click row to play",
      "semantic.segmentId": "Segment ID",
      "semantic.empty": "Stable Chinese and English segments will appear here after interpretation starts.",
      "semantic.details": "Recognition details",
      "audio.title": "Continuous English speech",
      "audio.hint": "Playback stays highlighted with the matching English subtitle.",
      "audio.waiting": "Waiting",
      "audio.current": "Current English:",
      "audio.none": "No English speech yet",
      "queue.title": "Playback queue",
      "queue.waiting": "Waiting for English speech",
      "queue.idle": "Idle",
      "results.title": "Results",
      "system.title": "System status",
      "phase.idle.button": "Start models",
      "phase.idle.note": "Load ASR · translation · TTS",
      "phase.idle.status": "Waiting · start the models first",
      "phase.idle.state": "Standby",
      "phase.loading.button": "Starting",
      "phase.loading.note": "Loading local models…",
      "phase.loading.status": "Warming up models · please wait",
      "phase.loading.state": "Starting",
      "phase.ready.button": "Start interpreting",
      "phase.ready.note": "Models ready · click to begin",
      "phase.ready.status": "Models ready · interpretation can begin",
      "phase.ready.state": "Ready",
      "phase.live.button": "Stop interpreting",
      "phase.live.note": "Receiving and processing audio",
      "phase.live.status": "Live · English follows stable Chinese",
      "phase.live.state": "Listening",
      "phase.stopping.button": "Finishing",
      "phase.stopping.note": "Committing the final stable segment…",
      "phase.stopping.status": "Finishing this session",
      "phase.stopping.state": "Finishing",
      "phase.error.button": "Retry startup",
      "phase.error.note": "The local service reported an error",
      "phase.error.status": "Connection error · check the terminal",
      "phase.error.state": "Error",
    },
  };

  function readSettings() {
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
      return { ...defaults, ...saved };
    } catch {
      return { ...defaults };
    }
  }

  let settings = readSettings();
  const systemTheme = window.matchMedia("(prefers-color-scheme: dark)");

  function resolvedTheme(theme = settings.theme) {
    return theme === "system" ? (systemTheme.matches ? "dark" : "light") : theme;
  }

  function t(key, fallback = "") {
    return messages[settings.language]?.[key] || messages["zh-CN"][key] || fallback || key;
  }

  function persist() {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(settings)); } catch { /* 浏览器禁用存储时仍允许临时使用。 */ }
  }

  function applyTheme() {
    document.documentElement.dataset.theme = resolvedTheme();
    document.documentElement.dataset.themePreference = settings.theme;
  }

  function applyLanguage() {
    document.documentElement.lang = settings.language;
    document.querySelectorAll("[data-i18n]").forEach((node) => {
      node.textContent = t(node.dataset.i18n, node.textContent);
    });
  }

  function applyAudio() {
    const sourceAudio = document.querySelector("#sourceAudio");
    const ttsAudio = document.querySelector("#ttsAudio");
    if (sourceAudio) sourceAudio.volume = Number(settings.sourceVolume) / 100;
    if (ttsAudio) ttsAudio.volume = Number(settings.ttsVolume) / 100;
  }

  function applySubtitleScale() {
    document.documentElement.style.setProperty("--subtitle-scale", String(Number(settings.subtitleScale) / 100));
  }

  function syncControls() {
    document.querySelectorAll("[data-theme-choice]").forEach((button) => {
      const active = button.dataset.themeChoice === settings.theme;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    document.querySelectorAll("[data-language-choice]").forEach((button) => {
      const active = button.dataset.languageChoice === settings.language;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    document.querySelectorAll("[data-default-input]").forEach((button) => {
      const active = button.dataset.defaultInput === settings.defaultInput;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    document.querySelectorAll("[data-subtitle-size]").forEach((button) => {
      const active = Number(button.dataset.subtitleSize) === Number(settings.subtitleScale);
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });

    const source = document.querySelector("#sourceVolume");
    const tts = document.querySelector("#ttsVolume");
    const autoplay = document.querySelector("#autoplayToggle");
    if (source) source.value = String(settings.sourceVolume);
    if (tts) tts.value = String(settings.ttsVolume);
    if (autoplay) autoplay.checked = Boolean(settings.autoplay);
    const sourceValue = document.querySelector("#sourceVolumeValue");
    const ttsValue = document.querySelector("#ttsVolumeValue");
    if (sourceValue) sourceValue.textContent = `${settings.sourceVolume}%`;
    if (ttsValue) ttsValue.textContent = `${settings.ttsVolume}%`;
  }

  function announce(message) {
    const status = document.querySelector("#settingsSaveStatus");
    if (!status) return;
    status.textContent = message;
    status.classList.add("visible");
    window.clearTimeout(announce.timer);
    announce.timer = window.setTimeout(() => status.classList.remove("visible"), 1800);
  }

  function apply({ notify = false, initial = false } = {}) {
    applyTheme();
    applyLanguage();
    applyAudio();
    applySubtitleScale();
    syncControls();

    if (initial) {
      const mode = document.querySelector(`.segmented button[data-mode="${settings.defaultInput}"]`);
      if (mode && !mode.classList.contains("active")) mode.click();
    }

    window.dispatchEvent(new CustomEvent("v4settingschange", { detail: { ...settings } }));
    if (notify) announce(t("settings.saved"));
  }

  function update(patch, { notify = true } = {}) {
    settings = { ...settings, ...patch };
    persist();
    apply({ notify });
  }

  function bindControls() {
    document.querySelectorAll("[data-theme-choice]").forEach((button) => button.addEventListener("click", () => update({ theme: button.dataset.themeChoice })));
    document.querySelectorAll("[data-language-choice]").forEach((button) => button.addEventListener("click", () => update({ language: button.dataset.languageChoice })));
    document.querySelectorAll("[data-default-input]").forEach((button) => button.addEventListener("click", () => update({ defaultInput: button.dataset.defaultInput })));
    document.querySelectorAll("[data-subtitle-size]").forEach((button) => button.addEventListener("click", () => update({ subtitleScale: Number(button.dataset.subtitleSize) })));

    document.querySelector("#sourceVolume")?.addEventListener("input", (event) => update({ sourceVolume: Number(event.target.value) }, { notify: false }));
    document.querySelector("#sourceVolume")?.addEventListener("change", () => announce(t("settings.saved")));
    document.querySelector("#ttsVolume")?.addEventListener("input", (event) => update({ ttsVolume: Number(event.target.value) }, { notify: false }));
    document.querySelector("#ttsVolume")?.addEventListener("change", () => announce(t("settings.saved")));
    document.querySelector("#autoplayToggle")?.addEventListener("change", (event) => update({ autoplay: event.target.checked }));
    document.querySelector("#resetSettings")?.addEventListener("click", () => {
      settings = { ...defaults };
      persist();
      apply({ initial: true });
      announce(t("settings.resetDone"));
    });
  }

  systemTheme.addEventListener?.("change", () => {
    if (settings.theme === "system") applyTheme();
  });

  // 在首屏绘制前尽早确定主题，避免深色模式闪白。
  applyTheme();

  window.V4Settings = {
    get: () => ({ ...settings }),
    t,
    update,
    apply,
  };

  document.addEventListener("DOMContentLoaded", () => {
    bindControls();
    apply({ initial: true });
  });
})();
