"""V3 acrylic presentation layer for the existing Gradio pipeline.

The UI deliberately keeps Gradio's native audio inputs and event system.  The
CSS/JS here only changes presentation and adds a local audio activity monitor;
it never performs ASR, translation, or TTS in the browser.
"""

from __future__ import annotations


APP_CSS = r"""
:root {
  --v3-ink: #182037;
  --v3-ink-soft: #303a5a;
  --v3-muted: #72798f;
  --v3-accent: #7279e6;
  --v3-accent-deep: #565fcf;
  --v3-lilac: #b49aef;
  --v3-peach: #ffb795;
  --v3-canvas: #edf1fa;
  --v3-glass: rgba(255,255,255,.62);
  --v3-line: rgba(109,120,158,.17);
  --v3-shadow: 0 22px 70px rgba(53,61,104,.12);
}

html, body, #root {
  width: 100%;
  min-width: 0;
  min-height: 100%;
}

body, .gradio-container {
  color: var(--v3-ink) !important;
  background:
    radial-gradient(circle at 91% 4%, rgba(157,174,255,.46), transparent 28rem),
    radial-gradient(circle at 4% 92%, rgba(213,180,246,.35), transparent 30rem),
    linear-gradient(145deg, rgba(255,255,255,.58), transparent 44%),
    var(--v3-canvas) !important;
  font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif !important;
}

body { margin:0; overflow-x:hidden; }
.gradio-container {
  width:100% !important;
  max-width:none !important;
  margin:0 !important;
  padding:clamp(16px,1.55vw,32px) clamp(18px,3vw,64px) 48px !important;
  box-sizing:border-box;
}
.contain { width:100% !important; max-width:none !important; }

#v3-topbar {
  display: grid;
  grid-template-columns: 1fr auto;
  align-items: center;
  min-height: 70px;
  margin-bottom: 20px;
  padding: 13px 20px;
  background: rgba(255,255,255,.61);
  border: 1px solid rgba(255,255,255,.72);
  border-radius: 20px;
  box-shadow: 0 12px 38px rgba(53,61,104,.08);
  backdrop-filter: blur(25px) saturate(145%);
}
.v3-brand strong { display:block; font-size:17px; letter-spacing:.01em; }
.v3-brand span { display:block; margin-top:3px; color:var(--v3-muted); font-size:10px; }
.v3-local { display:flex; align-items:center; gap:9px; color:var(--v3-muted); font-size:11px; }
.v3-local i { width:7px; height:7px; border-radius:50%; background:var(--v3-accent); box-shadow:0 0 0 5px rgba(114,121,230,.11); }
#v3-clock { min-width:58px; color:var(--v3-ink-soft); font-family:"Bahnschrift",monospace; font-variant-numeric:tabular-nums; }

.tabs > .tab-nav {
  width: fit-content;
  margin: 0 auto 20px !important;
  padding: 4px !important;
  gap: 3px !important;
  background: rgba(255,255,255,.46) !important;
  border: 1px solid rgba(255,255,255,.7) !important;
  border-radius: 13px !important;
  box-shadow: 0 8px 24px rgba(53,61,104,.06);
}
.tabs > .tab-nav button { min-width:112px; padding:9px 17px !important; border:0 !important; border-radius:9px !important; color:var(--v3-muted) !important; }
.tabs > .tab-nav button.selected { color:#fff !important; background:var(--v3-ink-soft) !important; box-shadow:0 6px 16px rgba(40,48,83,.18); }

#v3-hero {
  position: relative;
  min-height: 342px;
  margin-bottom: 22px;
  overflow: hidden;
  color: #f7f8ff;
  background: linear-gradient(125deg, rgba(29,35,69,.97), rgba(71,76,133,.9));
  border: 1px solid rgba(255,255,255,.25);
  border-radius: 27px;
  box-shadow: var(--v3-shadow);
}
#v3-hero::before { content:""; position:absolute; inset:0; pointer-events:none; background:linear-gradient(120deg,rgba(255,255,255,.08),transparent 38%,rgba(190,174,255,.08)); }
#v3-hero-copy { position:relative; z-index:1; padding:34px 34px 27px; }
#v3-hero-copy h1 { margin:15px 0 8px; color:#fff; font-size:clamp(36px,4.4vw,62px); line-height:1.04; letter-spacing:-.06em; font-weight:730; }
#v3-hero-copy p, #v3-hero-copy .prose { color:#c5c9dd !important; font-size:13px; }
#v3-hero-copy > .form { width:fit-content !important; padding:0 !important; background:transparent !important; border:0 !important; }
.v3-live-line { display:flex; align-items:center; gap:9px; color:#e3e6ff !important; font-size:11px; }
.v3-live-line span { color:#e3e6ff !important; opacity:.92; }
.v3-live-line i { width:8px; height:8px; border-radius:50%; background:var(--v3-peach); box-shadow:0 0 0 5px rgba(255,183,149,.13); animation:v3-pulse 2s ease-in-out infinite; }

#v3-wave { display:flex; align-items:center; gap:3px; width:min(640px,100%); height:48px; margin:19px 0 17px; }
#v3-wave i { width:3px; height:var(--h,8px); min-height:5px; border-radius:99px; opacity:.66; background:linear-gradient(#f6f7ff,#9da7ee); transition:height 75ms linear,opacity .16s ease; }
#v3-wave.v3-listening i { opacity:.95; }

#v3-input-mode { max-width:430px; padding:0 !important; overflow:visible !important; background:transparent !important; border:0 !important; }
#v3-input-mode > span { color:#bfc4da !important; background:transparent !important; }
#v3-input-mode > .wrap:not(.hide) { display:inline-flex !important; width:auto !important; padding:4px !important; gap:3px !important; background:rgba(255,255,255,.08) !important; border:1px solid rgba(255,255,255,.12) !important; border-radius:11px !important; }
#v3-input-mode label { padding:8px 12px !important; color:#bfc4da !important; background:transparent !important; border-radius:8px !important; }
#v3-input-mode label.selected { color:#fff !important; background:rgba(255,255,255,.14) !important; }
#v3-input-mode input { display:none !important; }
#v3-input-mode input:checked + span { color:#fff !important; }

#v3-ignition-panel {
  position:relative;
  display:flex;
  min-width:300px;
  min-height:342px;
  align-items:center;
  justify-content:center;
  border-left:1px solid rgba(255,255,255,.11);
  background:rgba(255,255,255,.035);
}
.v3-rings {
  position:absolute;
  /* The caption below the ignition control shifts the button's flex group
     18px upward; keep the acoustic rings locked to the actual button center. */
  top:calc(50% - 18px) !important; left:50% !important;
  width:220px; height:220px;
  transform:translate(-50%,-50%) scale(1);
  transform-origin:center;
  border:1px solid rgba(218,224,255,.22);
  border-radius:50%;
  pointer-events:none;
  transition:transform 80ms linear;
}
#v3-rings-layer {
  position:absolute !important;
  inset:0 !important;
  width:100% !important;
  height:100% !important;
  min-height:100% !important;
  padding:0 !important;
  margin:0 !important;
  pointer-events:none;
}
#v3-rings-layer .html-container,
#v3-rings-layer .prose {
  width:100% !important;
  height:100% !important;
  padding:0 !important;
  margin:0 !important;
}
.v3-rings::before,.v3-rings::after { content:""; position:absolute; inset:14%; border:1px solid rgba(218,224,255,.29); border-radius:50%; }
.v3-rings::after { inset:31%; border-color:rgba(255,183,149,.45); }
#v3-warmup {
  position:relative; z-index:2; min-width:0 !important; flex:none !important;
  width:112px !important; height:112px !important; padding:10px !important;
  white-space:pre-line !important; word-break:keep-all !important; line-height:1.28 !important;
  color:#222844 !important; font-weight:760 !important;
  font-size:14px !important; letter-spacing:.04em !important;
  background:radial-gradient(circle at 34% 28%,#fbfcff,#d1d6f7 68%,#aeb6e7) !important;
  border:1px solid rgba(255,255,255,.86) !important; border-radius:50% !important;
  box-shadow:0 16px 38px rgba(6,10,35,.3),inset 0 2px 3px rgba(255,255,255,.9),inset 0 -5px 10px rgba(81,91,154,.2),0 0 0 8px rgba(216,221,255,.09) !important;
  transition:transform .2s ease,box-shadow .25s ease !important;
}
#v3-warmup:hover { transform:translateY(-2px) scale(1.025); }
#v3-warmup.v3-warming { color:#6d4e3d !important; background:radial-gradient(circle at 34% 28%,#fff8f3,#ffd4be 68%,#eea37f) !important; animation:v3-breathe 1s ease-in-out infinite; }
#v3-warmup.v3-ready { color:#353d83 !important; background:radial-gradient(circle at 34% 28%,#fff,#dce1ff 68%,#abb5f1) !important; box-shadow:0 16px 38px rgba(6,10,35,.3),inset 0 2px 3px rgba(255,255,255,.9),0 0 0 9px rgba(148,158,239,.14),0 0 28px rgba(157,167,255,.35) !important; }
#v3-warmup.v3-running { color:#6f3f35 !important; background:radial-gradient(circle at 34% 28%,#fff8f4,#ffd1bd 68%,#ee9f7f) !important; box-shadow:0 16px 38px rgba(6,10,35,.3),inset 0 2px 3px rgba(255,255,255,.9),0 0 0 9px rgba(255,183,149,.14),0 0 30px rgba(255,183,149,.3) !important; }
.v3-ignition-caption { position:absolute; right:24px; bottom:23px; left:24px; color:#bbc0d6; text-align:center; font-size:10px; }

.v3-glass, #v3-status-card, #v3-semantic-card, #v3-timeline-card, #v3-results-card, #v3-system-card {
  background:var(--v3-glass) !important;
  border:1px solid rgba(255,255,255,.72) !important;
  border-radius:24px !important;
  box-shadow:0 12px 38px rgba(53,61,104,.08) !important;
  backdrop-filter:blur(25px) saturate(145%);
}
#v3-status-card .styler,
#v3-source-card-file .styler,
#v3-semantic-card .styler,
#v3-timeline-card .styler,
#v3-results-card .styler,
#v3-system-card .styler {
  background:transparent !important;
  border-color:transparent !important;
  box-shadow:none !important;
}
#v3-status-card { margin-bottom:20px; padding:10px 14px; }
#v3-status-card textarea { font-variant-numeric:tabular-nums; }
#v3-status-card .html-container { padding:0 !important; background:transparent !important; border:0 !important; }
#v3-mode-help {
  padding:9px 12px !important;
  color:var(--v3-ink-soft) !important;
  background:linear-gradient(90deg,rgba(114,121,230,.09),rgba(180,154,239,.06)) !important;
  border:1px solid rgba(114,121,230,.12) !important;
  border-radius:11px !important;
}
#v3-mode-help p { margin:0 !important; color:var(--v3-ink-soft) !important; font-size:12px !important; }
#v3-semantic-card, #v3-timeline-card, #v3-results-card, #v3-system-card { margin-bottom:22px; padding:18px 20px 20px; overflow:hidden; }
.v3-section-title { position:relative; }
.v3-section-title h2 { margin:0; color:var(--v3-ink); font-size:17px; letter-spacing:-.02em; }
.v3-section-title p { margin:4px 0 13px; color:var(--v3-muted); font-size:10px; }
.v3-section-status {
  position:absolute; top:4px; right:2px;
  display:inline-flex; align-items:center; gap:7px;
  color:var(--v3-accent-deep); font-size:10px;
}
.v3-section-status::before { content:""; width:7px; height:7px; border-radius:50%; background:var(--v3-accent); box-shadow:0 0 0 4px rgba(114,121,230,.1); }

/* Native transport controls stay mounted but never participate in layout. */
#v3-mic-transport, #v3-ui-transport {
  position:fixed !important;
  top:0 !important;
  left:-12000px !important;
  width:4px !important;
  height:4px !important;
  min-width:0 !important;
  min-height:0 !important;
  overflow:hidden !important;
  opacity:.001 !important;
  pointer-events:none !important;
}

#v3-source-card-file {
  margin-bottom:22px !important;
  padding:18px !important;
  background:var(--v3-glass) !important;
  border:1px solid rgba(255,255,255,.72) !important;
  border-radius:24px !important;
  box-shadow:0 12px 38px rgba(53,61,104,.08) !important;
  backdrop-filter:blur(25px) saturate(145%);
  overflow:hidden;
}
#v3-file-audio-grid { gap:14px !important; align-items:stretch !important; }
#v3-upload-audio, #v3-source-playback {
  min-height:168px !important;
  background:rgba(255,255,255,.44) !important;
  border:1px solid var(--v3-line) !important;
  border-radius:17px !important;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.7) !important;
  overflow:hidden !important;
}
#v3-upload-audio > .wrap.default.full,
#v3-source-playback > .wrap.default.full { height:168px !important; min-height:168px !important; }
#v3-upload-audio .audio-container,
#v3-source-playback .audio-container { height:144px !important; min-height:144px !important; }
#v3-upload-audio .audio-container > button,
#v3-source-playback .audio-container > button { height:144px !important; min-height:144px !important; }
#v3-upload-audio .audio-container .wrap,
#v3-source-playback .audio-container .wrap { height:168px !important; min-height:168px !important; }
#v3-upload-audio button { color:var(--v3-muted) !important; }
#v3-file-controls {
  justify-content:flex-end !important;
  align-items:center !important;
  gap:10px !important;
  margin-top:14px !important;
}
#v3-file-start, #v3-file-stop {
  flex:0 0 auto !important;
  width:auto !important;
  min-width:176px !important;
  min-height:42px !important;
  border-radius:11px !important;
  font-weight:700 !important;
  transition:transform .16s ease,box-shadow .16s ease,background .16s ease !important;
}
#v3-file-start {
  color:#fff !important;
  background:linear-gradient(135deg,var(--v3-accent-deep),var(--v3-accent)) !important;
  border:1px solid rgba(255,255,255,.3) !important;
  box-shadow:0 8px 20px rgba(86,95,207,.22) !important;
}
#v3-file-stop {
  min-width:108px !important;
  color:#8d5364 !important;
  background:rgba(255,255,255,.56) !important;
  border:1px solid rgba(141,83,100,.22) !important;
  box-shadow:none !important;
}
#v3-file-start:hover, #v3-file-stop:hover { transform:translateY(-1px); }

#v3-semantic-card .html-container,
#v3-timeline-card .html-container,
#v3-results-card .html-container,
#v3-system-card .html-container {
  padding:0 !important;
  background:transparent !important;
  border:0 !important;
}
#live-translation {
  min-height:148px !important;
  padding:0 !important;
  background:rgba(255,255,255,.30) !important;
  border:1px solid var(--v3-line) !important;
  border-radius:17px !important;
  overflow:hidden !important;
}
#live-translation > .html-container,
#live-translation > .html-container > .prose {
  width:100% !important;
  max-width:none !important;
  padding:0 !important;
  margin:0 !important;
  background:transparent !important;
}
.v3-unit-empty {
  display:grid;
  min-height:148px;
  place-content:center;
  gap:7px;
  padding:24px;
  color:var(--v3-muted);
  text-align:center;
}
.v3-unit-empty strong { color:var(--v3-ink-soft); font-size:13px; }
.v3-unit-empty span { font-size:10px; }
.v3-unit-list { display:grid; gap:8px; padding:8px; }
.v3-unit-head, .v3-unit-row {
  display:grid;
  grid-template-columns:78px minmax(180px,.9fr) minmax(230px,1.1fr) 54px;
  align-items:stretch;
}
.v3-unit-head {
  padding:4px 3px;
  color:var(--v3-muted);
  font-size:9px;
  letter-spacing:.04em;
}
.v3-unit-head span { padding:5px 10px; }
.v3-unit-row {
  color:var(--v3-ink-soft);
  background:rgba(255,255,255,.54);
  border:1px solid rgba(255,255,255,.78);
  border-left:3px solid transparent;
  border-radius:12px;
  box-shadow:0 1px 0 rgba(255,255,255,.8);
  overflow:hidden;
  transition:transform .16s ease,background .16s ease,box-shadow .16s ease;
}
.v3-unit-row:hover {
  transform:translateY(-1px);
  background:rgba(237,239,255,.76);
  box-shadow:0 8px 22px rgba(56,64,104,.08);
}
.v3-unit-row.is-playing {
  background:rgba(226,229,255,.84);
  border-left-color:var(--v3-accent);
}
.v3-unit-row > time,
.v3-unit-row > p,
.v3-unit-id,
.v3-unit-play { margin:0; padding:12px 11px; }
.v3-unit-row > time {
  color:var(--v3-muted);
  font:11px/1.55 "Bahnschrift",monospace;
  font-variant-numeric:tabular-nums;
}
.v3-unit-row > p { line-height:1.55; border-left:1px solid var(--v3-line); }
.v3-unit-play {
  display:flex;
  min-width:0;
  align-items:center;
  justify-content:space-between;
  gap:12px;
  color:var(--v3-ink) !important;
  text-align:left;
  font:620 12px/1.55 inherit;
  background:transparent !important;
  border:0 !important;
  border-left:1px solid var(--v3-line) !important;
  border-radius:0 !important;
  box-shadow:none !important;
  cursor:pointer;
}
.v3-unit-play:hover { color:var(--v3-accent-deep) !important; }
.v3-unit-play i {
  flex:0 0 auto;
  display:grid;
  width:28px;
  height:28px;
  place-items:center;
  color:var(--v3-accent-deep);
  background:rgba(255,255,255,.7);
  border:1px solid rgba(114,121,230,.15);
  border-radius:50%;
  font-size:9px;
  font-style:normal;
}
.v3-unit-row.is-pending .v3-unit-play i { opacity:.36; }
.v3-unit-id {
  display:grid;
  place-items:center;
  color:var(--v3-muted);
  border-left:1px solid var(--v3-line);
  font:10px "Bahnschrift",monospace;
}
#live-stable { background:rgba(255,255,255,.28) !important; }
#live-stable textarea { min-height:62px !important; max-height:110px !important; font-size:12px !important; line-height:1.6 !important; background:rgba(255,255,255,.2) !important; }
#live-stable label > span { color:var(--v3-muted) !important; font-size:9px !important; }
#live-subtitles { min-height:96px; padding:10px; background:rgba(255,255,255,.44); border:1px solid var(--v3-line); border-radius:13px; }
.live-subtitle-list { display:flex; flex-direction:column; gap:7px; }
.live-subtitle-row { display:flex; gap:12px; align-items:baseline; padding:8px 10px; border-left:4px solid transparent; border-radius:7px; transition:background .14s ease,border-color .14s ease; }
.live-subtitle-row.playing { background:rgba(114,121,230,.14); border-left-color:var(--v3-accent); font-weight:700; }
.live-subtitle-row.queued { background:rgba(114,121,230,.06); }
.live-subtitle-row.partial { opacity:.66; border-left-style:dashed; }
.live-subtitle-time { min-width:55px; color:var(--v3-muted); font-family:"Bahnschrift",monospace; font-variant-numeric:tabular-nums; }
.live-subtitle-en { flex:1; }
.live-subtitles-empty { padding:22px 8px; color:var(--v3-muted); text-align:center; }
#v3-timeline-card { padding:16px !important; background:rgba(255,255,255,.54) !important; }
.v3-player-title { padding:1px 5px 8px; }
#live-subtitles { min-height:0 !important; margin-bottom:10px; padding:7px !important; background:transparent !important; border:0 !important; }
.live-subtitle-list { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:7px; }
.live-subtitle-row { min-width:0; background:rgba(255,255,255,.4); border:1px solid rgba(255,255,255,.72); border-left:3px solid transparent; border-radius:11px; }
.live-subtitle-row.playing { background:rgba(226,229,255,.76); border-color:rgba(114,121,230,.14); border-left-color:var(--v3-accent); }
.live-subtitle-en { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
#live-audio-player { width:100%; min-height:74px !important; padding:10px 12px !important; color:#fff !important; background:rgba(29,35,69,.94) !important; border:1px solid rgba(255,255,255,.17) !important; border-radius:15px !important; box-shadow:0 12px 30px rgba(28,34,67,.15) !important; overflow:hidden !important; }
#live-audio-player label, #live-audio-player span { color:#e8eaff !important; }
#live-audio-player .audio-container, #live-audio-player .wrap { background:transparent !important; border-color:transparent !important; }
#live-audio-player [aria-label="Empty value"] { min-height:48px !important; opacity:.42; }
#v3-metrics-details { margin-bottom:0 !important; }

#v3-source-playback ::part(wrapper) { margin-bottom:22px !important; }

#v3-recognition-details, #v3-metrics-details { margin-top:12px; background:rgba(255,255,255,.34); border:1px solid var(--v3-line); border-radius:14px; }
#v3-footer { display:flex; justify-content:space-between; gap:20px; padding:8px 4px; color:var(--v3-muted); font-size:10px; }
.gradio-container footer:not(#v3-footer) { display:none !important; }

@keyframes v3-pulse { 50% { opacity:.5; transform:scale(.78); } }
@keyframes v3-breathe { 50% { transform:scale(.965); filter:saturate(1.18); } }
@media (min-width: 1700px) {
  #v3-topbar { min-height:76px; padding-right:26px; padding-left:26px; }
  #v3-hero { min-height:366px; }
  #v3-hero-copy { padding:40px 44px 31px; }
  #v3-hero-copy h1 { max-width:980px; font-size:66px; }
  #v3-wave { width:min(820px,100%); }
  #v3-ignition-panel { min-height:366px; }
  #v3-semantic-card, #v3-timeline-card, #v3-results-card, #v3-system-card { padding-right:26px; padding-left:26px; }
}
@media (max-width: 900px) {
  .gradio-container { padding:12px 12px 34px !important; }
  #v3-topbar { grid-template-columns:1fr; gap:8px; }
  .v3-local { justify-content:flex-start; }
  #v3-hero { flex-direction:column !important; }
  #v3-ignition-panel { min-width:0; min-height:230px; border-top:1px solid rgba(255,255,255,.11); border-left:0; }
  #v3-file-audio-grid { flex-direction:column !important; }
  #v3-file-controls { justify-content:stretch !important; }
  #v3-file-start, #v3-file-stop { flex:1 1 0 !important; min-width:0 !important; }
  .tabs > .tab-nav { width:100%; overflow-x:auto; }
  .tabs > .tab-nav button { min-width:100px; }
  .v3-section-status { position:static; margin-bottom:12px; }
  .v3-unit-head, .v3-unit-row { grid-template-columns:64px minmax(150px,.9fr) minmax(190px,1.1fr); }
  .v3-unit-head span:last-child, .v3-unit-id { display:none; }
}
@media (max-width: 620px) {
  .v3-unit-head { display:none; }
  .v3-unit-row { grid-template-columns:60px minmax(0,1fr); }
  .v3-unit-row > p { border-left:1px solid var(--v3-line); }
  .v3-unit-play { grid-column:1 / -1; border-top:1px solid var(--v3-line) !important; border-left:0 !important; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior:auto !important; animation-duration:.001ms !important; animation-iteration-count:1 !important; transition-duration:.001ms !important; }
}
"""


APP_JS = r"""
() => {
  const monitor = { stream: null, context: null, analyser: null, data: null, frame: 0, fileNode: null };
  const sessionControl = { ready: false, running: false };
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  function ensureWave() {
    const wave = $('#v3-wave');
    if (!wave || wave.dataset.ready) return;
    wave.dataset.ready = '1';
    for (let i = 0; i < 64; i += 1) {
      const bar = document.createElement('i');
      bar.style.setProperty('--h', `${6 + (i % 7)}px`);
      wave.appendChild(bar);
    }
  }

  function setVisual(level, bins = null) {
    const wave = $('#v3-wave');
    if (!wave) return;
    const bars = $$('i', wave);
    bars.forEach((bar, index) => {
      const envelope = .48 + .52 * Math.sin(Math.PI * (index + 1) / (bars.length + 1));
      const value = bins ? bins[Math.min(bins.length - 1, Math.floor(index / bars.length * bins.length))] / 255 : level;
      bar.style.setProperty('--h', `${5 + Math.min(1, value * 1.35) * 36 * envelope}px`);
    });
    const rings = $('.v3-rings');
    if (rings) rings.style.transform = `translate(-50%, -50%) scale(${1 + level * .10})`;
  }

  function animate() {
    if (!monitor.analyser || !monitor.data) return;
    monitor.analyser.getByteFrequencyData(monitor.data);
    const average = monitor.data.reduce((sum, value) => sum + value, 0) / Math.max(1, monitor.data.length);
    setVisual(Math.min(1, average / 92), monitor.data);
    monitor.frame = requestAnimationFrame(animate);
  }

  async function startMicMonitor() {
    if (monitor.stream || !navigator.mediaDevices?.getUserMedia) return;
    try {
      monitor.stream = await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true}});
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      monitor.context = new AudioContext();
      const source = monitor.context.createMediaStreamSource(monitor.stream);
      monitor.analyser = monitor.context.createAnalyser();
      monitor.analyser.fftSize = 128;
      monitor.analyser.smoothingTimeConstant = .72;
      monitor.data = new Uint8Array(monitor.analyser.frequencyBinCount);
      source.connect(monitor.analyser);
      $('#v3-wave')?.classList.add('v3-listening');
      cancelAnimationFrame(monitor.frame);
      animate();
    } catch (_) { /* Native Gradio recording still reports permission errors. */ }
  }

  async function stopMicMonitor() {
    cancelAnimationFrame(monitor.frame);
    monitor.stream?.getTracks().forEach(track => track.stop());
    monitor.stream = null;
    monitor.analyser = null;
    monitor.data = null;
    await monitor.context?.close().catch(() => {});
    monitor.context = null;
    $('#v3-wave')?.classList.remove('v3-listening');
    setVisual(0);
  }

  function bindMicControls() {
    if (document.body.dataset.v3MicBound) return;
    document.body.dataset.v3MicBound = '1';
    document.addEventListener('click', event => {
      const button = event.target.closest('#v3-mic-transport button');
      if (!button) return;
      const label = `${button.textContent} ${button.title || ''} ${button.getAttribute('aria-label') || ''}`.toLowerCase();
      if (!/(录制|停止|record|stop)/.test(label)) return;
      if (monitor.stream) stopMicMonitor(); else setTimeout(startMicMonitor, 180);
    });
  }

  function watchWarmup() {
    const button = $('#v3-warmup');
    if (!button || button.dataset.watched) return;
    button.dataset.watched = '1';
    const sync = () => {
      const text = button.textContent || '';
      const warming = text.includes('正在启动');
      const ready = text.includes('开始同声传译') || text.includes('停止同声传译');
      sessionControl.ready = ready;
      button.classList.toggle('v3-warming', warming);
      button.classList.toggle('v3-ready', ready);
      const line = $('.v3-live-line span');
      if (line && !sessionControl.running) {
        line.textContent = warming
          ? '正在启动本地模型'
          : ready ? '模型就绪 · 可开始同传' : '等待输入 · 先点火预热模型';
      }
      const caption = $('.v3-ignition-caption');
      if (caption && !sessionControl.running) {
        caption.textContent = warming
          ? '正在依次加载 ASR · 翻译 · TTS'
          : ready ? 'ASR · 翻译 · TTS 已就绪' : '启动本地 ASR · 翻译 · TTS';
      }
    };
    new MutationObserver(sync).observe(button, {subtree:true,childList:true,characterData:true});
    sync();
  }

  function currentInputMode() {
    const checked = $('#v3-input-mode input:checked');
    return checked?.value?.includes('上传') ? 'file' : 'microphone';
  }

  function nativeRecorderButton(action) {
    const buttons = $$('#v3-mic-transport button');
    return buttons.find(candidate => {
      const label = `${candidate.textContent || ''} ${candidate.title || ''} ${candidate.getAttribute('aria-label') || ''}`.toLowerCase();
      if (action === 'stop') return /(停止|stop)/.test(label);
      return /(录制|record)/.test(label) && !/(停止|stop)/.test(label);
    }) || null;
  }

  function setSessionControl(running) {
    sessionControl.running = running;
    const button = $('#v3-warmup');
    if (!button) return;
    button.textContent = running ? '■\n停止同声传译' : '▶\n开始同声传译';
    button.classList.toggle('v3-running', running);
    button.classList.add('v3-ready');
    const line = $('.v3-live-line span');
    if (line) line.textContent = running ? '正在监听 · 同传流水线运行中' : '模型就绪 · 可开始同传';
    const caption = $('.v3-ignition-caption');
    if (caption) caption.textContent = running ? '再次点击可停止同声传译' : 'ASR · 翻译 · TTS 已就绪';
  }

  function bindSourcePlaybackEnd() {
    const audio = $('#v3-source-playback audio');
    if (!audio || audio.dataset.v3EndBound) return;
    audio.dataset.v3EndBound = '1';
    audio.addEventListener('ended', () => setSessionControl(false));
  }

  function bindUnifiedSessionControl() {
    if (document.body.dataset.v3UnifiedControlBound) return;
    document.body.dataset.v3UnifiedControlBound = '1';
    document.addEventListener('click', event => {
      const button = event.target.closest('#v3-warmup');
      if (!button || !sessionControl.ready) return;
      event.preventDefault();
      event.stopImmediatePropagation();

      const nextRunning = !sessionControl.running;
      const mode = currentInputMode();
      let target = null;
      if (mode === 'microphone') {
        target = nativeRecorderButton(nextRunning ? 'start' : 'stop');
      } else {
        target = nextRunning ? $('#v3-file-start') : $('#v3-file-stop');
      }
      if (!target || target.disabled) {
        const status = $('#v3-status-card textarea');
        if (status) status.value = mode === 'file' ? '请先上传 WAV 音频' : '麦克风暂不可用，请检查浏览器权限';
        return;
      }
      setSessionControl(nextRunning);
      target.click();
    }, true);
  }

  function bindUnitPlayback() {
    if (document.body.dataset.v3UnitPlaybackBound) return;
    document.body.dataset.v3UnitPlaybackBound = '1';
    document.addEventListener('click', event => {
      const unit = event.target.closest('.v3-unit-play');
      if (!unit) return;
      const segmentId = unit.dataset.segmentId;
      const row = unit.closest('.v3-unit-row');
      if (!row?.classList.contains('has-audio')) return;
      const input = $('#v3-selected-segment input, #v3-selected-segment textarea');
      const trigger = $('#v3-play-segment button, button#v3-play-segment');
      if (!segmentId || !input || !trigger) return;

      const prototype = input instanceof HTMLTextAreaElement
        ? HTMLTextAreaElement.prototype
        : HTMLInputElement.prototype;
      const setter = Object.getOwnPropertyDescriptor(prototype, 'value')?.set;
      if (setter) setter.call(input, segmentId); else input.value = segmentId;
      input.dispatchEvent(new Event('input', {bubbles:true}));
      input.dispatchEvent(new Event('change', {bubbles:true}));
      $$('.v3-unit-row.is-playing').forEach(row => row.classList.remove('is-playing'));
      row.classList.add('is-playing');
      requestAnimationFrame(() => trigger.click());
    });
  }

  function syncPlaceholders() {
    $$('#live-stable textarea').forEach(area => {
      area.placeholder = area.value.trim() ? '' : '等待稳定中文片段……';
    });
  }

  function markTransportControls() {
    ['#v3-mic-transport', '#v3-ui-transport'].forEach(selector => {
      const host = $(selector);
      if (host) host.setAttribute('aria-hidden', 'true');
    });
  }

  function bindClock() {
    if (window.__v3Clock) return;
    const tick = () => {
      const node = $('#v3-clock');
      const next = new Date().toLocaleTimeString('zh-CN',{hour12:false});
      if (node && node.textContent !== next) node.textContent = next;
    };
    tick();
    window.__v3Clock = setInterval(tick, 1000);
  }

  function boot() {
    ensureWave();
    bindMicControls();
    watchWarmup();
    bindUnifiedSessionControl();
    bindUnitPlayback();
    bindSourcePlaybackEnd();
    bindClock();
    syncPlaceholders();
    markTransportControls();
  }
  boot();
  new MutationObserver(boot).observe(document.body, {childList:true,subtree:true});
}
"""
