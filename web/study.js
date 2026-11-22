"use strict";

const byId = (id) => document.getElementById(id);
const names = {bowl: "碗 · Bowl", bottle: "瓶 · Bottle", spoon: "勺子 · Spoon", glass: "玻璃杯 · Glass", pen: "笔 · Pen"};
const phases = {
  settling: "初始物体沉降", prepare: "准备", approach: "接近", descend: "下降",
  close: "闭合夹爪", lift: "抬升", hold: "抓取保持", raise: "提高搬运高度",
  rotate: "调整姿态", transfer: "搬运至抽屉", entry: "进入抽屉",
  insert: "放入抽屉", lower: "降低物体", release: "松开物体",
  withdraw: "退出抽屉", verify_release: "撤手后验证真实松开", clear: "离开柜体", return_home: "返回准备位置", push_approach: "接近抽屉把手",
  prepush_home: "推关前安全归位", push_drawer: "接触推闭抽屉", push_withdraw: "推闭后退离",
  verify: "最终连续物理验收", done: "任务结束，保留终态", failed: "任务失败，保留终态",
};
const factNames = {inside: "完全在抽屉内", supported: "底板真实支撑", detached: "与手爪脱离", stable: "速度稳定", stored: "收纳通过"};
const controllerNames = {idle: "等待稳定后自动开始", running: "正在执行", succeeded: "服务端报告成功", failed: "服务端报告失败"};
const socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);

function showError(message) {
  byId("error").hidden = false;
  byId("error").textContent = message;
}

function send(message) {
  if (socket.readyState !== WebSocket.OPEN) {
    showError("仿真连接未就绪；请检查服务后重新加载页面。");
    return;
  }
  socket.send(JSON.stringify(message));
}

function number(value, digits = 2) {
  return Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function measurement(value, unit, digits = 2) {
  return Number.isFinite(value) ? `${number(value, digits)} ${unit}` : "—";
}

function factTag(label, value) {
  const tag = document.createElement("span");
  tag.textContent = `${label}：${value === true ? "是" : value === false ? "否" : "未知"}`;
  tag.dataset.ok = String(value === true);
  return tag;
}

function updateObjects(state) {
  byId("object-count").textContent = `服务端实测 ${state.objects.length} 件`;
  byId("mission-order").replaceChildren(...state.targets.map((target) => {
    const item = document.createElement("li");
    const object = state.objects.find((entry) => entry.target === target);
    item.textContent = `${names[target] ?? target}${object?.stored === true ? " · 已收纳" : ""}`;
    item.dataset.active = String(target === state.target && !state.finished);
    return item;
  }));
  byId("object-states").replaceChildren(...state.targets.map((target) => {
    const object = state.objects.find((entry) => entry.target === target);
    const item = document.createElement("li");
    item.dataset.active = String(target === state.target && !state.finished);
    const title = document.createElement("strong");
    title.textContent = `${names[target] ?? target}${target === state.target && !state.finished && state.current_index < state.targets.length ? " · 当前目标" : ""}`;
    const position = document.createElement("span");
    position.className = "position";
    position.textContent = object?.position_m
      ? `位置 x / y / z：${object.position_m.map((value) => number(value, 3)).join(" / ")} m`
      : "尚无实测位置";
    const facts = document.createElement("div");
    facts.className = "predicates";
    facts.append(...Object.entries(factNames).map(([key, label]) => factTag(label, object?.[key])));
    item.append(title, position, facts);
    if (Number.isFinite(object?.support_force_n)) {
      const support = document.createElement("span");
      support.className = "position";
      support.textContent = `抽屉底板支撑力：${measurement(object.support_force_n, "N")}`;
      item.append(support);
    }
    return item;
  }));
}

function updateCabinet(state) {
  const cabinet = state.cabinet;
  const position = Number.isFinite(cabinet.drawer_position_m)
    ? `${number(cabinet.drawer_position_m * 1000, 1)} mm`
    : "位置未知";
  const closure = state.drawer_closed === true ? "已关闭" : state.drawer_closed === false ? "未关闭" : "未知";
  byId("drawer-position").textContent = `${closure} · ${position}`;
  const interior = cabinet.interior_size_m
    ? ` · 内腔 ${cabinet.interior_size_m.map((value) => number(value * 1000, 0)).join(" × ")} mm`
    : "";
  byId("cabinet-info").textContent = `右侧真实抽屉 · ${closure} · 滑动关节位置 ${position}${interior}。关闭判定来自服务端物理状态。`;
}

function update(state) {
  if (state.runtime_error) {
    showError(`物理或渲染已停止：${state.runtime_error}`);
    byId("connection").textContent = "运行错误 · 只保留最后真实画面";
    byId("phase").textContent = "物理或渲染已停止";
    byId("outcome").textContent = `未完成 · ${state.runtime_error}`;
    byId("outcome").dataset.kind = "failure";
    byId("current-outcome").textContent = "运行中止，不确认成功";
    for (const control of document.querySelectorAll("button,select")) control.disabled = true;
    return;
  }
  updateObjects(state);
  updateCabinet(state);
  const targetName = state.current_index >= state.targets.length
    ? "抽屉柜 · Cabinet" : names[state.target] ?? state.target ?? "无当前目标";
  const phase = phases[state.phase] ?? state.phase;
  const allStored = state.targets.length === 5 && new Set(state.targets).size === 5
    && state.objects.length === 5 && state.stored_count === 5
    && state.targets.every((target) => state.objects.some((object) => object.target === target
      && Object.keys(factNames).every((key) => object[key] === true)));
  const succeeded = state.finished === true && state.controller_status === "succeeded"
    && state.outcome === true && allStored && state.drawer_closed === true
    && state.hand_empty === true && !state.runtime_error;
  const failed = state.controller_status === "failed" || state.outcome === false || Boolean(state.runtime_error);
  let result = state.paused ? "已暂停，尚未确认完成" : "实际执行中，尚未确认完成";
  if (succeeded) result = "完成 · 五物体实测收纳、抽屉关闭、手爪空置";
  else if (failed) result = `未完成 · ${state.runtime_error || state.failure_reason || "物理验收未通过"}`;
  else if (state.controller_status === "succeeded" || state.outcome === true) result = "成功报告尚未满足全部实测判据，不确认完成";
  else if (state.finished) result = "任务已结束，但未确认成功";

  byId("object-name").textContent = `当前目标：${targetName}`;
  byId("phase").textContent = phase;
  byId("connection").textContent = state.runtime_error
    ? "物理或渲染已停止 · 最后实测状态"
    : state.paused ? "已暂停 · Physics paused" : "实时 MuJoCo 接触计算 · Live physics";
  byId("frame-count").textContent = `整场运行 ${state.run_serial} · 帧 ${state.frame_id}`;
  byId("mission-note").textContent = state.finished
    ? "任务终态保留；不会自动重启。仅手动重新开始才重置全部五物体和抽屉。"
    : state.phase === "settling" || state.controller_status === "idle"
      ? "等待物体稳定后自动开始；不需要启动按钮。物体之间不重置。"
      : state.current_index < state.targets.length
        ? `连续收纳任务 · 服务端当前序号 ${state.current_index + 1} / ${state.targets.length} · 物体之间不重置；之后真实推闭抽屉。`
        : "五物体搬运阶段结束；正在真实推闭抽屉或执行最终物理验收，不重置场景。";
  byId("stored-count").textContent = `${state.stored_count} / ${state.targets.length}`;
  byId("time").textContent = measurement(state.time_s, "s");
  byId("grasp-hold").textContent = measurement(state.grasp_hold_s, "s");
  byId("verified-hold").textContent = measurement(state.verified_hold_s, "s");
  byId("lift").textContent = measurement(state.lift_m, "m", 3);
  byId("current-outcome").textContent = result;
  byId("controller-status").textContent = `${controllerNames[state.controller_status] ?? state.controller_status} · ${phase}`;
  byId("outcome").textContent = state.finished || failed ? `${result} · 保留终态，不自动重启` : state.paused ? "已暂停" : `${phase} · 正在实际执行`;
  byId("outcome").dataset.kind = succeeded ? "success" : failed ? "failure" : "running";
  byId("frame").alt = `五物体连续收纳 · 当前目标 ${targetName} · ${phase} · MuJoCo 实时物理画面`;
  byId("pause").textContent = state.paused ? "继续" : "暂停";
  for (const control of document.querySelectorAll("button,select")) control.disabled = false;
  if (document.activeElement !== byId("view")) byId("view").value = state.view;
  const settings = state.settings;
  byId("parameters").textContent = settings
    ? `尺寸 ${settings.scale}× · 质量 ${settings.mass_kg} kg · 摩擦 ${settings.sliding_friction} · 下降指令 ${settings.descent_speed_m_s} m/s · 开口指令 ${settings.opening_m} m · 闭合指令 ${settings.closing_force_n} N · 触发距离 ${settings.close_trigger_m} m`
    : "无当前目标设置";
  byId("predicates").replaceChildren(
    factTag("五物体全部实测收纳", allStored),
    factTag("抽屉实测关闭", state.drawer_closed),
    factTag("手爪实测空置", state.hand_empty),
  );
  if (state.runtime_error) showError(`物理或渲染已停止：${state.runtime_error}`);
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
  byId("connection").textContent = "连接中断 · 保留最后一帧，不代表继续执行";
  for (const control of document.querySelectorAll("button,select")) control.disabled = true;
});
byId("pause").addEventListener("click", () => send({action: "pause"}));
byId("restart").addEventListener("click", () => send({action: "restart"}));
byId("view").addEventListener("change", () => send({action: "view", view: byId("view").value}));
