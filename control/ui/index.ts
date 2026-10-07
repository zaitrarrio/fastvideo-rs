// The editor bundle (public/editor.js, built by build-ui.mjs): loaded on
// demand by app.js on the pages that edit or view JSON or configuration.
import { EditorView } from "@codemirror/view";
import { openClusterConfig } from "./cluster/config";
import { createJsonEditor } from "./editor";
import { auditForms, controlKind, createForm, formDialog } from "./fields";
import { renderForm } from "./form";
import { askCancelQueued, askExtend, askJobCancel, askKeyName, askPurge, askRoll, askScale, askSlsCancel } from "./forms/dialogs";
import { mountEndpointForm, mountScaleForm, mountSpecEditor } from "./forms/serverless";
import { mountInvokePicker, servesPanel } from "./forms/serves";
import { mountReleaseForm, mountTokenForm } from "./forms/settings";
import { mountLaunchForm } from "./forms/standalone";
import { openDocPanel } from "./panel";
import { schemaAt, validate } from "./schema";
import { diffLines, renderDiff, renderTree } from "./view";

(window as any).FVEditor = {
  openClusterConfig,
  createJsonEditor,
  renderForm,
  openDocPanel,
  renderDiff,
  renderTree,
  diffLines,
  validate,
  schemaAt,
  controlKind,
  createForm,
  formDialog,
  auditForms,
  mountLaunchForm,
  mountEndpointForm,
  mountSpecEditor,
  mountScaleForm,
  mountInvokePicker,
  servesPanel,
  mountTokenForm,
  mountReleaseForm,
  askExtend,
  askScale,
  askRoll,
  askKeyName,
  askSlsCancel,
  askPurge,
  askJobCancel,
  askCancelQueued,
  viewOf: (dom: HTMLElement) => EditorView.findFromDOM(dom),
};
