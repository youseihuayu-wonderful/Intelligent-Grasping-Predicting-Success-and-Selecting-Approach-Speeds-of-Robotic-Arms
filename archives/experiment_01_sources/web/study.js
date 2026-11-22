"use strict";

const byId = (id) => document.getElementById(id);
const names = {bowl: "碗 · Bowl", bottle: "瓶 · Bottle", spoon: "勺子 · Spoon", glass: "玻璃杯 · Glass", pen: "笔 · Pen"};
const phases = {settling: "初始物体沉降", prepare: "准备", approach: "接近", descend: "下降", close: "闭合夹爪", lift: "抬升", hold: "稳定保持", done: "控制器完成，检查连续保持", failed: "实际抓取失败", idle: "等待执行"};
const predicateNames = {bilateral_grip: "双侧接触", clear_of_table: "脱离桌面", target_lifted: "抬升达标", relative_pose_stable: "相对稳定", verified_hold: "控制器保持"};
let latest = null;
let pendingSelection = null;
let requestedRunSerial = 0;
const socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);

function showError(message) {
  byId("error").hidden = false;
  byId("error").textContent = message;
}
function send(message) {
  if (socket.readyState !== WebSocket.OPEN) {
    showError("仿真连接未就绪；请检查服务后重新加载页面。");
    return false;
  }
  socket.send(JSON.stringify(message));
  return true;
}
function requestRun(target, message) {
  if (!send(message)) return;
  requestedRunSerial = Math.max(requestedRunSerial, latest?.run_serial ?? 0) + 1;
  pendingSelection = {target, minimumRunSerial: requestedRunSerial};
  updateSelection();
}
function selectTarget(target) {
  requestRun(target, {action: "select", target});
}
function updateSelection() {
  const intendedTarget = pendingSelection?.target ?? latest?.target;
  for (const button of document.querySelectorAll("[data-target]")) {
    button.setAttribute("aria-pressed", String(button.dataset.target === intendedTarget));
  }
  byId("selection-note").textContent = pendingSelection
    ? `正在请求抓取 ${names[intendedTarget]}；等待服务端确认全场重置。画面仍显示实际执行状态。`
    : `当前抓取目标：${names[intendedTarget]}。切换目标会重置全场，其他物体不会被替换。`;
}
function updateObjects(state) {
  byId("object-count").textContent = `实测 ${state.objects.length} 个动态物体`;
  byId("object-states").replaceChildren(...state.objects.map((object) => {
    const item = document.createElement("li");
    item.dataset.active = String(object.target === state.target);
    const title = document.createElement("strong");
    title.textContent = `${names[object.target] ?? object.target}${object.target === state.target ? " · 当前目标" : ""}`;
    const bodyName = document.createElement("small");
    bodyName.textContent = object.name;
    const position = document.createElement("span");
    position.textContent = `位置 x / y / z：${object.position_m.map((value) => value.toFixed(3)).join(" / ")} m`;
    const clearance = document.createElement("span");
    clearance.textContent = `实测离桌间隙：${object.table_clearance_m.toFixed(4)} m`;
    item.append(title, bodyName, position, clearance);
    return item;
  }));
}
function update(state) {
  if (state.runtime_error) {
    showError(`物理或渲染已停止：${state.runtime_error}`);
    byId("connection").textContent = "仿真已停止，不会用旧画面伪装继续执行";
    for (const control of document.querySelectorAll("button,select,input")) control.disabled = true;
    return;
  }
  latest = state;
  requestedRunSerial = Math.max(requestedRunSerial, state.run_serial);
  if (pendingSelection && pendingSelection.target === state.target && state.run_serial >= pendingSelection.minimumRunSerial) pendingSelection = null;
  updateSelection();
  updateObjects(state);
  byId("object-name").textContent = names[state.target];
  byId("phase").textContent = phases[state.phase] || state.phase;
  byId("connection").textContent = state.paused ? "已暂停 · Physics paused" : "实时 MuJoCo 接触计算 · Live physics";
  byId("frame-count").textContent = `运行 ${state.run_serial} · 帧 ${state.frame_id}`;
  byId("time").textContent = `${state.time_s.toFixed(2)} s`;
  byId("hold").textContent = `${state.verified_hold_s.toFixed(2)} s`;
  byId("lift").textContent = state.lift_m === null ? "—" : `${state.lift_m.toFixed(3)} m`;
  byId("opening").textContent = `${(state.gripper_opening_m * 1000).toFixed(1)} mm`;
  const result = state.invalid_reason ? `无效初始状态：${state.invalid_reason}` : state.outcome === true ? "SUCCESS · 双侧接触、离桌、抬升、稳定和连续 2 秒保持通过" : state.outcome === false ? `FAILURE · ${state.failure_reason || "物理验收未通过"}` : "执行中，尚未判定";
  byId("current-outcome").textContent = result;
  const completionNote = state.paused ? "已暂停" : state.auto_advance ? "终态显示后重置全场并切换目标" : "自动切换已关闭";
  const finalResult = state.invalid_reason ? "本次初始状态无效" : state.outcome === true ? "本次物理抓取通过" : state.outcome === false ? "本次未通过 · 保留真实失败" : "本次结束，尚无验收结果";
  byId("outcome").textContent = state.finished ? `${finalResult} · ${completionNote}` : state.paused ? "已暂停" : `${phases[state.phase] || state.phase} · 正在实际执行`;
  byId("frame").alt = `五物体同桌 · 当前抓取 ${names[state.target]} · MuJoCo 实时物理画面 · ${phases[state.phase] || state.phase}`;
  byId("outcome").dataset.kind = state.invalid_reason || state.outcome === false ? "failure" : state.outcome === true ? "success" : "running";
  byId("pause").textContent = state.paused ? "继续" : "暂停";
  if (document.activeElement !== byId("view")) byId("view").value = state.view;
  if (document.activeElement !== byId("auto")) byId("auto").checked = state.auto_advance;
  const p = state.settings;
  byId("parameters").textContent = `尺寸 ${p.scale}× · 质量 ${p.mass_kg} kg · 摩擦 ${p.sliding_friction} · 下降 ${p.descent_speed_m_s} m/s · 开口 ${p.opening_m} m · 闭合指令 ${p.closing_force_n} N · 触发距离 ${p.close_trigger_m} m`;
  for (const item of document.querySelectorAll("[data-phase]")) {
    item.classList.toggle("active", item.dataset.phase === state.phase && !state.finished);
  }
  byId("predicates").replaceChildren(...Object.entries(predicateNames).map(([key, label]) => {
    const tag = document.createElement("span");
    tag.textContent = label;
    tag.dataset.ok = String(Boolean(state.predicates[key]));
    return tag;
  }));
  if (state.warnings.length) showError(`MuJoCo 数值警告：${state.warnings.join(", ")}`);
}
socket.addEventListener("message", (event) => {
  const packet = JSON.parse(event.data);
  if (packet.error) {
    showError(packet.error);
    return;
  }
  if (packet.image) byId("frame").src = `data:image/jpeg;base64,${packet.image}`;
  if (packet.state) update(packet.state);
});
socket.addEventListener("error", () => showError("无法连接本地 MuJoCo 服务，请检查终端服务日志。"));
socket.addEventListener("close", () => {
  byId("connection").textContent = "连接中断 · 最后一帧，不是继续执行";
  for (const control of document.querySelectorAll("button,select,input")) control.disabled = true;
});
for (const button of document.querySelectorAll("[data-target]")) button.addEventListener("click", () => selectTarget(button.dataset.target));
byId("pause").addEventListener("click", () => send({action: "pause"}));
byId("restart").addEventListener("click", () => {
  const target = pendingSelection?.target ?? latest?.target;
  if (target) requestRun(target, {action: "restart"});
});
byId("next").addEventListener("click", () => {
  if (!latest) return;
  const index = latest.targets.indexOf(pendingSelection?.target ?? latest.target);
  selectTarget(latest.targets[(index + 1) % latest.targets.length]);
});
byId("view").addEventListener("change", () => send({action: "view", view: byId("view").value}));
byId("auto").addEventListener("change", () => send({action: "auto", enabled: byId("auto").checked}));
