"use strict";

// 视频转谱的页面：新建转谱（链接或拖入视频）、进行中的任务、我的谱（历史作业）、作业详情（有问题的小节）、B 站扫码登录。
const TOKEN = document.querySelector('meta[name="token"]').content;
const $ = (selector) => document.querySelector(selector);

// ---------------------------------------------------------------- 工具

async function api(path, options = {}) {
  const init = { method: options.method || "GET", headers: { "X-Token": TOKEN } };
  if (options.json !== undefined) {
    init.method = options.method || "POST";
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(options.json);
  }
  const response = await fetch("/api/" + path, init);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `出错了（${response.status}）`);
  return data;
}

function fileUrl(job, path, key, download) {
  const query = new URLSearchParams({ t: TOKEN, path });
  if (key) query.set("key", key);
  if (download) query.set("download", "1");
  return `/api/jobs/${job}/file?${query}`;
}

// 建一个元素：h("div", {class: "x", onclick: fn}, 子元素或文字…)
function h(tag, attrs = {}, ...children) {
  const element = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (name.startsWith("on")) element.addEventListener(name.slice(2), value);
    else if (name === "class") element.className = value;
    else element.setAttribute(name, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === undefined || child === null || child === false) continue;
    element.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return element;
}

let toastTimer = null;
function toast(text) {
  const box = $("#toast");
  box.textContent = text;
  box.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (box.hidden = true), 3500);
}

async function guarded(action) {
  try {
    return await action();
  } catch (error) {
    toast(error.message);
  }
}

// ---------------------------------------------------------------- 状态和任务

let state = { tasks: [] };
let lastActive = new Set();

async function refreshState() {
  try {
    state = await api("state");
  } catch {
    return;
  }
  renderNotices();
  renderLogin();
  renderTasks();
  const active = new Set(state.tasks.filter((task) => task.state === "queued" || task.state === "running").map((task) => task.id));
  // 有任务结束了：刷新"我的谱"。
  if ([...lastActive].some((id) => !active.has(id))) refreshJobs();
  lastActive = active;
}

function renderNotices() {
  const box = $("#notices");
  box.replaceChildren();
  if (state.translocated) {
    box.append(h("div", { class: "notice" },
      "macOS 把程序放在了临时的只读位置运行，转出的谱没法存进 video2score/data。请退出后把整个 video2score 文件夹挪到别的位置（比如\"应用程序\"或\"文稿\"）再打开；或者在终端运行：",
      h("br"), h("code", {}, `xattr -dr com.apple.quarantine "${state.app_root}"`)));
  }
  if (!state.browser) {
    box.append(h("div", { class: "notice" }, "没找到打印用的浏览器：排版 PDF 暂时不能导出（打包好的程序自带，从仓库运行时请用 python packaging/build.py --browser 下载，或安装 Chrome）。其他功能不受影响。"));
  }
  if (!state.node) {
    box.append(h("div", { class: "notice" }, "没找到 Node.js：排版 PDF 暂时不能导出。"));
  }
  const pdf = document.querySelector('input[name="export"][value="pdf"]');
  pdf.disabled = !state.browser || !state.node;
  if (pdf.disabled) pdf.checked = false;
  $("#data-dir").textContent = state.data_dir ? `文件都在 ${state.data_dir}` : "";
}

function renderLogin() {
  $("#login-button").textContent = state.logged_in ? "B 站已登录 · 退出登录" : "B 站登录";
}

const STATES = { queued: "排队中", running: "进行中", done: "完成", failed: "失败", cancelled: "已取消" };

function renderTasks() {
  const list = $("#tasks");
  const tasks = state.tasks.filter((task) => task.state === "queued" || task.state === "running" || task.state === "failed" || task.error);
  $("#tasks-card").hidden = tasks.length === 0;
  list.replaceChildren(...tasks.map((task) => {
    const live = task.state === "queued" || task.state === "running";
    return h("li", { class: "task" },
      h("div", { class: "row space" },
        h("strong", {}, (task.kind === "export" ? "导出：" : "") + task.label),
        live ? h("button", { class: "small", onclick: () => guarded(async () => { await api(`tasks/${task.id}/cancel`, { json: {} }); refreshState(); }) }, "取消") : null),
      live ? h("div", { class: "progress" }, h("div", { style: `width:${Math.round(task.progress * 100)}%` })) : null,
      h("div", { class: "small muted" }, task.state === "running" ? `${task.stage}（${Math.round(task.progress * 100)}%）` : STATES[task.state]),
      task.error ? h("div", { class: "error" }, task.error) : null);
  }));
}

// ---------------------------------------------------------------- 我的谱

async function refreshJobs() {
  const jobs = await guarded(() => api("jobs"));
  if (!jobs) return;
  $("#jobs-empty").hidden = jobs.length > 0;
  $("#jobs").replaceChildren(...jobs.map(renderJob));
}

function renderJob(job) {
  const tags = [];
  if (job.status === "failed") tags.push(h("span", { class: "tag bad" }, "失败"));
  else if (job.status === "captured") tags.push(h("span", { class: "tag" }, "只有截图谱"));
  if (job.issues) tags.push(h("span", { class: "tag warn" }, `${job.issues} 处要核对`));
  const exportSelect = h("select", { "aria-label": "补导出的格式" },
    h("option", { value: "pdf" }, "排版 PDF"), h("option", { value: "gp" }, "GP（.gp）"),
    h("option", { value: "gpx" }, "GP6（.gpx）"), h("option", { value: "gp5" }, "GP5（.gp5）"));
  return h("li", { class: "job" },
    h("div", { class: "job-title", onclick: () => openDetail(job.id) }, job.title, ...tags),
    h("div", { class: "job-meta" }, [job.artist, job.created, job.measures ? `${job.measures} 小节` : null].filter(Boolean).join(" · ")),
    job.error ? h("div", { class: "error" }, job.error) : null,
    h("div", { class: "files" }, job.files.map((file) => h("a", {
      href: fileUrl(job.id, file.path, file.key, !file.path.endsWith(".pdf")),
      target: file.path.endsWith(".pdf") ? "_blank" : null,
      title: file.path.endsWith(".pdf") ? "在浏览器里打开" : "下载",
    }, file.label))),
    h("div", { class: "actions" },
      h("button", { class: "small", onclick: () => openDetail(job.id) }, job.issues ? "看要核对的小节" : "详情"),
      job.can_export ? [exportSelect, h("button", {
        class: "small",
        onclick: () => guarded(async () => { await api(`jobs/${job.id}/export`, { json: { exports: [exportSelect.value] } }); toast("已加入队列"); refreshState(); }),
      }, "补导出")] : null,
      h("button", { class: "small", onclick: () => guarded(() => api(`jobs/${job.id}/open`, { json: {} })) }, "打开文件夹"),
      h("button", {
        class: "small danger",
        onclick: () => guarded(async () => {
          if (!confirm(`删除"${job.title}"？会移到 data/trash 里，需要时可以拿回来。`)) return;
          await api(`jobs/${job.id}/remove`, { json: {} });
          toast("已移到回收站");
          refreshJobs();
        }),
      }, "删除")));
}

async function openDetail(id) {
  const job = await guarded(() => api(`jobs/${id}`));
  if (!job) return;
  $("#detail-title").textContent = job.title;
  const body = $("#detail-body");
  body.replaceChildren(h("div", {},
    h("p", { class: "muted small" }, [job.artist, job.created, job.status_text].filter(Boolean).join(" · ")),
    job.link ? h("p", { class: "small" }, "出处：", h("a", { href: job.link, target: "_blank", rel: "noreferrer" }, job.credit || job.link)) : null,
    job.error ? h("p", { class: "error" }, job.error) : null,
    h("h2", {}, job.issue_list.length ? `要核对的地方（${job.issue_list.length}）` : "没有要核对的地方"),
    job.issue_list.length ? h("p", { class: "muted small" }, "这些小节识别时没把握，建议在 Guitar Pro 里对照视频看一下。") : null,
    ...job.issue_list.map((item) => h("div", { class: "issue" },
      h("strong", {}, item.number ? `第 ${item.number} 小节` : "全曲"),
      h("ul", {}, item.issues.map((text) => h("li", {}, text))),
      item.image ? h("img", { src: fileUrl(job.id, item.image), alt: `第 ${item.number} 小节` }) : null))));
  $("#detail").showModal();
}

// ---------------------------------------------------------------- 新建转谱

let chosenFile = null;

function chooseFile(file) {
  chosenFile = file || null;
  $("#drop").classList.toggle("chosen", Boolean(chosenFile));
  $("#drop-text").textContent = chosenFile ? `已选：${chosenFile.name}（点这里换一个）` : "或者把视频文件拖到这里（也可以点这里选择）";
  $("#link-row").hidden = !chosenFile;
  if (chosenFile) $("#source").value = "";
}

function upload(file) {
  // 用 XMLHttpRequest 才能看到上传进度（视频可能很大）。
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", `/api/upload?name=${encodeURIComponent(file.name)}`);
    request.setRequestHeader("X-Token", TOKEN);
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) $("#upload-status").textContent = `正在载入视频 ${Math.round((event.loaded / event.total) * 100)}%`;
    };
    request.onload = () => {
      const data = JSON.parse(request.responseText || "{}");
      if (request.status === 200) resolve(data);
      else reject(new Error(data.error || "上传失败"));
    };
    request.onerror = () => reject(new Error("上传失败"));
    request.send(file);
  });
}

async function startConvert(event) {
  event.preventDefault();
  const exports = [...document.querySelectorAll('input[name="export"]:checked')].map((box) => box.value);
  const button = $("#start");
  button.disabled = true;
  try {
    let source = $("#source").value.trim();
    let label = source;
    if (chosenFile) {
      const saved = await upload(chosenFile);
      source = saved.path;
      label = chosenFile.name;
    }
    if (!source) throw new Error("请粘贴视频链接，或拖入一个视频文件。");
    await api("convert", { json: { source, label, exports, link: chosenFile ? $("#link").value.trim() : null } });
    $("#source").value = "";
    $("#link").value = "";
    chooseFile(null);
    toast("已开始，进度在下面");
    refreshState();
  } catch (error) {
    toast(error.message);
  } finally {
    button.disabled = false;
    $("#upload-status").textContent = "";
  }
}

// ---------------------------------------------------------------- 登录、退出

let loginKey = null;
let loginTimer = null;

async function openLogin() {
  if (state.logged_in) {
    if (!confirm("退出 B 站登录？")) return;
    await guarded(() => api("logout", { json: {} }));
    return refreshState();
  }
  $("#login-qr").removeAttribute("src");
  $("#login-status").textContent = "正在获取二维码……";
  $("#login").showModal();
  const data = await guarded(() => api("login", { json: {} }));
  if (!data) return $("#login").close();
  loginKey = data.key;
  $("#login-qr").src = data.qr;
  $("#login-status").textContent = "用哔哩哔哩手机客户端扫码";
  clearInterval(loginTimer);
  loginTimer = setInterval(pollLogin, 2000);
}

async function pollLogin() {
  if (!$("#login").open) return clearInterval(loginTimer);
  const data = await api(`login?key=${encodeURIComponent(loginKey)}`).catch((error) => ({ status: "error", error: error.message }));
  const text = { waiting: "用哔哩哔哩手机客户端扫码", scanned: "已扫码，请在手机上确认", expired: "二维码已失效，请关掉重新打开", error: data.error };
  if (data.status === "logged_in") {
    clearInterval(loginTimer);
    $("#login").close();
    toast("登录成功");
    return refreshState();
  }
  $("#login-status").textContent = text[data.status] || "";
  if (data.status === "expired") clearInterval(loginTimer);
}

async function quit() {
  const busy = state.tasks.some((task) => task.state === "queued" || task.state === "running");
  if (!confirm(busy ? "还有任务没做完，退出会中断它们。确定退出？" : "退出视频转谱？")) return;
  await api("quit", { json: {} }).catch(() => {});
  document.body.replaceChildren(h("main", {}, h("section", { class: "card" }, h("h2", {}, "视频转谱已退出"), h("p", { class: "muted" }, "可以关闭这个页面了。再用时重新打开 video2score。"))));
}

// ---------------------------------------------------------------- 启动

function init() {
  $("#convert-form").addEventListener("submit", startConvert);
  $("#source").addEventListener("input", () => { if ($("#source").value.trim()) chooseFile(null); });
  const drop = $("#drop");
  drop.addEventListener("click", () => $("#file").click());
  drop.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") $("#file").click(); });
  $("#file").addEventListener("change", () => chooseFile($("#file").files[0]));
  drop.addEventListener("dragover", (event) => { event.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (event) => {
    event.preventDefault();
    drop.classList.remove("over");
    if (event.dataTransfer.files.length) chooseFile(event.dataTransfer.files[0]);
  });
  // 拖到页面别处也不要让浏览器直接打开视频。
  window.addEventListener("dragover", (event) => event.preventDefault());
  window.addEventListener("drop", (event) => event.preventDefault());
  $("#login-button").addEventListener("click", openLogin);
  $("#quit-button").addEventListener("click", quit);
  for (const button of document.querySelectorAll("[data-close]")) button.addEventListener("click", () => button.closest("dialog").close());
  refreshState();
  refreshJobs();
  // 有任务在跑时每秒刷新，空闲时每 5 秒。
  let tick = 0;
  setInterval(() => {
    tick += 1;
    if (lastActive.size > 0 || tick % 5 === 0) refreshState();
  }, 1000);
}

init();
