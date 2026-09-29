// The editor bundle (public/editor.js, built by build-ui.mjs): loaded on
// demand by app.js on the pages that edit or view JSON.
import { createJsonEditor } from "./editor";
import { renderForm } from "./form";
import { openDocPanel } from "./panel";
import { validate } from "./schema";
import { diffLines, renderDiff, renderTree } from "./view";

(window as any).FVEditor = { createJsonEditor, renderForm, openDocPanel, renderDiff, renderTree, diffLines, validate };
