// The log explorer bundle (public/logs.js + logs.css, built by build-ui.mjs):
// loaded by app.js on the Logs page.
import "./logs.css";
import { mountLogs } from "./explorer";
import * as model from "./model";

(window as any).FVLogs = { mount: mountLogs, model };
