// Starts the Vite dev server and the FastAPI server together. Neither one builds an
// index: the API only opens a saved one, and `python rag_core.py --rebuild` builds it.
import { spawn } from "node:child_process";
import console from "node:console";
import { existsSync } from "node:fs";
import { connect } from "node:net";
import { dirname, resolve } from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const API_PORT = 8000;
const frontend = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const root = resolve(frontend, "..");
const backend = resolve(root, "backend");
const children = [];

function findPython() {
  const candidates = [resolve(root, ".venv/Scripts/python.exe"), resolve(root, ".venv/bin/python")];
  return candidates.find((path) => existsSync(path)) ?? "python";
}

function isPortInUse(port) {
  return new Promise((done) => {
    const socket = connect({ port, host: "127.0.0.1" });
    socket.once("connect", () => {
      socket.destroy();
      done(true);
    });
    socket.once("error", () => done(false));
  });
}

function start(command, args, cwd) {
  const child = spawn(command, args, { cwd, stdio: "inherit" });
  children.push(child);
  return child;
}

function stopAll(exitCode = 0) {
  for (const child of children) child.kill();
  process.exit(exitCode);
}

process.on("SIGINT", () => stopAll());
process.on("SIGTERM", () => stopAll());

if (await isPortInUse(API_PORT)) {
  console.log(`[dev] Port ${API_PORT} is already in use, so the API there will be used.`);
} else {
  const api = start(findPython(), ["-m", "uvicorn", "api:app", "--port", String(API_PORT)], backend);
  api.on("error", (error) => console.error(`[dev] The API couldn't start: ${error.message}`));
  api.on("exit", (code) => {
    if (code) console.error(`[dev] The API stopped with exit code ${code}.`);
  });
}

const web = start(process.execPath, [resolve(frontend, "node_modules/vite/bin/vite.js")], frontend);
web.on("exit", (code) => stopAll(code ?? 0));
