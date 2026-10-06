// The editor bundle (public/editor.js, built by build-ui.mjs): loaded on
// demand by app.js on the pages that edit or view JSON.
import { EditorView } from "@codemirror/view";
import { openClusterConfig } from "./cluster/config";
import { createJsonEditor } from "./editor";
import { renderForm } from "./form";
import { openDocPanel } from "./panel";
import { validate } from "./schema";
import { diffLines, renderDiff, renderTree } from "./view";

(window as any).FVEditor = { openClusterConfig, createJsonEditor, renderForm, openDocPanel, renderDiff, renderTree, diffLines, validate, viewOf: (dom: HTMLElement) => EditorView.findFromDOM(dom) };
