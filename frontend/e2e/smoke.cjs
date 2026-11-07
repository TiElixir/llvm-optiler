// Browser smoke test: compile a template -> findings visible -> apply the fix -> verified -> compare opens.
// Uses the globally installed Playwright (NODE_PATH=/usr/lib/node_modules) and starts its own server.
//   NODE_PATH=/usr/lib/node_modules node frontend/e2e/smoke.cjs   (PW_CHANNEL=chromium to use Playwright's own build)
const { chromium } = require("playwright");
const { spawn } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const ROOT = path.resolve(__dirname, "..", "..");
const PORT = process.env.PORT || "8777";
const URL = `http://127.0.0.1:${PORT}/`;
const runs = fs.mkdtempSync(path.join(os.tmpdir(), "optiler-e2e-"));

function ok(cond, msg) { if (!cond) throw new Error("FAILED: " + msg); console.log("ok  " + msg); }

(async () => {
  const server = spawn(path.join(ROOT, ".venv/bin/uvicorn"), ["backend.api.app:app", "--port", PORT, "--log-level", "warning"],
    { cwd: ROOT, env: { ...process.env, OPTILER_RUNS_DIR: runs }, stdio: "inherit", detached: true });
  const stop = () => { try { process.kill(-server.pid, "SIGTERM"); } catch { /* already gone */ } };
  process.on("exit", stop);
  let browser;
  try {
    for (let i = 0; i < 40; i++) { try { if ((await fetch(URL)).ok) break; } catch { /* not up yet */ } await new Promise((r) => setTimeout(r, 500)); }
    browser = await chromium.launch({ channel: process.env.PW_CHANNEL || "chrome", headless: true });
    const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
    const errors = [];
    page.on("pageerror", (e) => errors.push(String(e)));
    await page.goto(URL);

    // settings: fewer iterations keeps the test short
    await page.click("#advanced");
    await page.fill("#set-iters", "30");
    ok((await page.locator("#pipeline").inputValue()).includes("stage: linalg_tensor"), "advanced drawer shows the default pipeline");
    await page.click("#drawer-close");

    await page.click("#compile");
    await page.waitForSelector(".finding", { timeout: 120000 });
    const finding = await page.locator(".finding").first().innerText();
    ok(/not vectorized/i.test(finding) && /est/.test(finding), "a finding with an estimate is listed: " + finding.split("\n")[1]);
    ok(await page.locator("#metrics .metric").count() >= 2, "result bar shows runtime and ratio");
    ok(await page.locator(".grp").count() > 20, "pipeline map shows lineage groups");

    await page.locator(".finding").first().click();
    ok(await page.locator(".grp.hit").count() >= 4, "selecting a finding traces its lines through the stages");
    ok(await page.locator(".fixbtn").count() === 1, "the selected finding offers a fix");
    await page.keyboard.press("f");
    await page.waitForSelector(".run >> nth=1", { timeout: 60000 });
    await page.waitForFunction(() => !document.getElementById("compile").disabled && document.querySelector("#metrics .metric"), null, { timeout: 120000 });
    ok(await page.locator(".run").count() === 2, "the child run joins the history chain");

    await page.locator(".run").first().click();
    await page.waitForSelector(".finding", { timeout: 30000 });
    await page.locator(".finding").first().click();   // the accent marks the selected finding only
    await page.waitForSelector(".fv.verified", { timeout: 30000 });
    ok(true, "the parent finding is verified: " + (await page.locator(".fv.verified").innerText()));

    await page.locator(".run").nth(1).click({ modifiers: ["Shift"] });
    await page.waitForSelector(".compare table");
    ok(/faster/.test(await page.locator(".compare table").innerText()), "compare shows the measured speedup");
    await page.keyboard.press("Escape");

    await page.click("#advanced");
    ok(await page.locator("#pass-table tr").count() > 100, "advanced drawer lists every pass with timings");
    ok(((await page.locator("#repro").getAttribute("href")) || "").endsWith("/reproducer"), "reproducer download is offered");
    ok(errors.length === 0, "no page errors" + (errors.length ? ": " + errors.join("; ") : ""));
    console.log("\nsmoke test passed");
  } finally {
    await browser?.close();
    stop();
    await new Promise((r) => setTimeout(r, 1500));  // let the worker exit before its runs directory goes away
    fs.rmSync(runs, { recursive: true, force: true });
  }
})().catch((e) => { console.error(e.message); process.exit(1); });
