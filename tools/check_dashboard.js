/* Render the dashboard's script against a real state.json, outside a browser.
 *
 * `node --check` proves the script parses; it does not prove the render path works.
 * A missing property or a bad call inside render() throws at runtime and leaves the
 * page as empty as a syntax error does. This runs the actual render function against
 * actual run state with a minimal DOM stub, and reports what it produced.
 *
 *     node tools/check_dashboard.js runs/garage-ep0-.../state.json
 *
 * Exits non-zero if the script throws or produces no scenario markup.
 */
"use strict";

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const statePath = process.argv[2];
if (!statePath) {
  console.error("usage: node tools/check_dashboard.js <state.json>");
  process.exit(2);
}

const html = fs.readFileSync(
  path.join(__dirname, "..", "src", "penumbra", "ui", "dashboard.html"), "utf8");
const script = html.split("<script>")[1].split("</script>")[0];
const state = JSON.parse(fs.readFileSync(statePath, "utf8"));

/* A DOM thin enough to be obviously honest: elements remember what was assigned to
 * them and nothing else. Anything the script reads back has to be something it wrote. */
const nodes = {};
function makeNode(id) {
  return {
    id: id,
    innerHTML: "",
    textContent: "",
    style: {},
    classList: { toggle: function () {}, add: function () {}, remove: function () {} },
    querySelector: function () { return { src: "" }; }
  };
}
["sub", "cams", "phase", "updated", "dot", "livetext", "rail", "stats", "headline", "control",
 "correction", "chips", "scenarios", "crash", "lightbox"].forEach(function (id) {
  nodes[id] = makeNode(id);
});

const sandbox = {
  console: console,
  Number: Number,
  JSON: JSON,
  Math: Math,
  Object: Object,
  String: String,
  document: {
    getElementById: function (id) { return nodes[id] || makeNode(id); },
    addEventListener: function () {}
  },
  window: { addEventListener: function () {} },
  EventSource: function () { return { onmessage: null, onerror: null }; }
};
sandbox.globalThis = sandbox;

let failed = null;
try {
  vm.createContext(sandbox);
  vm.runInContext(script, sandbox, { filename: "dashboard.js" });
  sandbox.render(state);
} catch (err) {
  failed = err;
}

if (failed) {
  console.error("render threw: " + (failed && failed.stack ? failed.stack : failed));
  process.exit(1);
}
if (nodes.crash.style.display === "block") {
  console.error("script reported a crash: " + nodes.crash.textContent);
  process.exit(1);
}

const cards = (nodes.scenarios.innerHTML.match(/class="card /g) || []).length;
const imgs = (nodes.scenarios.innerHTML.match(/<img /g) || []).length;
const stats = (nodes.stats.innerHTML.match(/class="cell/g) || []).length;
console.log("scenario cards : " + cards);
console.log("generation imgs: " + imgs);
console.log("stat cells     : " + stats);
console.log("headline       : " + (nodes.headline.innerHTML ? "present" : "empty"));
console.log("control card   : " + (nodes.control.innerHTML.indexOf("card") >= 0 ? "present" : "empty"));
console.log("phase          : " + nodes.phase.textContent);

if ((state.scenarios || []).length > 0 && cards === 0) {
  console.error("state has scenarios but no cards were produced");
  process.exit(1);
}
