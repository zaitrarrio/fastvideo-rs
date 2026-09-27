//! The model's OpenAPI 3.1 document: `GET /schema` and `ModelSchema.openapi`
//! (reactor §6.3, `RT:interface/model/schema.py:to_openapi`).
//!
//! - each command is `paths["/events/<name>"].post` with a JSON body schema,
//!   answering 200 with a `$ref` to its reply message, or 202 (bodyless ack);
//! - model messages are `webhooks` and `components.schemas`, next to
//!   `ReactorUploadReference`;
//! - `x-reactor.tracks[{name, kind, direction}]` in **model** perspective
//!   (`out` for what we send): `[main_video, main_audio]` or `[main_video]`,
//!   identical to the descriptor and `track_map` (design §5.3);
//! - each field carries `x-reactor-moderate`.

use fastvideo_protocol::TrackSet;
use serde_json::{json, Map, Value};

use crate::commands::CommandTable;

/// `x-reactor.tracks` entries, model perspective.
pub fn model_tracks(tracks: &TrackSet) -> Vec<Value> {
    let mut v = vec![json!({"name": tracks.video.name, "kind": "video", "direction": "out"})];
    if let Some(a) = &tracks.audio {
        v.push(json!({"name": a.name, "kind": "audio", "direction": "out"}));
    }
    v
}

/// PascalCase component name of a snake_case message type
/// (`state_update` → `StateUpdate`, RT's `pascal_to_snake` inverted).
pub fn component_name(msg: &str) -> String {
    msg.split('_')
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                Some(f) => f.to_uppercase().chain(c).collect::<String>(),
                None => String::new(),
            }
        })
        .collect()
}

/// The OpenAPI document for `table`.
pub fn openapi(title: &str, version: &str, table: &CommandTable, tracks: &TrackSet) -> Value {
    let mut paths = Map::new();
    for c in &table.commands {
        let mut props = Map::new();
        let mut required = Vec::new();
        for p in &c.params {
            props.insert(p.name.to_owned(), p.schema());
            if p.required {
                required.push(json!(p.name));
            }
        }
        let body = json!({"type": "object", "properties": props, "required": required, "additionalProperties": false});
        let responses = match c.reply {
            Some(r) => json!({"200": {
                "description": format!("Replies with `{r}`."),
                "content": {"application/json": {"schema": {"$ref": format!("#/components/schemas/{}", component_name(r))}}}
            }}),
            None => json!({"202": {"description": "Accepted (bodyless ack)."}}),
        };
        paths.insert(
            format!("/events/{}", c.name),
            json!({"post": {
                "operationId": c.name,
                "summary": c.description,
                "requestBody": {"required": true, "content": {"application/json": {"schema": body}}},
                "responses": responses
            }}),
        );
    }
    let mut schemas = Map::new();
    let mut webhooks = Map::new();
    for m in &table.messages {
        let name = component_name(m.name);
        let mut s = m.schema.clone();
        s["title"] = json!(name);
        s["description"] = json!(m.description);
        s["x-reactor-message-type"] = json!(m.name);
        schemas.insert(name.clone(), s);
        webhooks.insert(
            m.name.to_owned(),
            json!({"post": {
                "summary": m.description,
                "requestBody": {"content": {"application/json": {"schema": {"$ref": format!("#/components/schemas/{name}")}}}},
                "responses": {"200": {"description": "Delivered."}}
            }}),
        );
    }
    schemas.insert(
        "ReactorUploadReference".into(),
        json!({"type": "object", "title": "ReactorUploadReference", "properties": {
            "upload_id": {"type": "string"}, "name": {"type": "string"},
            "mime_type": {"type": "string"}, "size": {"type": "integer"}
        }, "required": ["upload_id"]}),
    );
    json!({
        "openapi": "3.1.0",
        "info": {"title": title, "version": version},
        "paths": paths,
        "webhooks": webhooks,
        "components": {"schemas": schemas},
        "x-reactor": {"tracks": model_tracks(tracks), "mode": table.mode}
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::ClipBounds;
    use crate::engine::Mode;
    use fastvideo_protocol::{AudioTrack, VideoTrack};

    fn tracks(audio: bool) -> TrackSet {
        TrackSet {
            video: VideoTrack { name: "main_video".into(), width: 1344, height: 768, fps: 24 },
            audio: audio.then(|| AudioTrack { name: "main_audio".into(), rate: 48_000, channels: 1 }),
        }
    }

    #[test]
    fn document_shape() {
        let t = CommandTable::for_mode(Mode::Clip, ClipBounds { min_s: 5.167, max_s: 14.375, default_s: 5.167 });
        let d = openapi("fasth3", "1", &t, &tracks(true));
        assert_eq!(d["openapi"], "3.1.0");
        // fast-h3's own test pins these two outbound tracks.
        assert_eq!(
            d["x-reactor"]["tracks"],
            json!([{"name":"main_video","kind":"video","direction":"out"},{"name":"main_audio","kind":"audio","direction":"out"}])
        );
        let e = &d["paths"]["/events/enqueue"]["post"];
        assert_eq!(e["requestBody"]["content"]["application/json"]["schema"]["required"], json!(["prompt"]));
        assert_eq!(e["requestBody"]["content"]["application/json"]["schema"]["properties"]["prompt"]["x-reactor-moderate"], true);
        assert_eq!(e["responses"]["200"]["content"]["application/json"]["schema"]["$ref"], "#/components/schemas/ClipQueued");
        assert!(d["paths"]["/events/play"]["post"]["responses"]["202"].is_object());
        assert!(d["webhooks"]["state_update"].is_object());
        assert!(d["components"]["schemas"]["StateUpdate"].is_object());
        assert!(d["components"]["schemas"]["ReactorUploadReference"].is_object());

        let v = openapi("sfwan", "1", &CommandTable::for_mode(Mode::Causal, ClipBounds { min_s: 0.0, max_s: 0.0, default_s: 0.0 }), &tracks(false));
        assert_eq!(v["x-reactor"]["tracks"], json!([{"name":"main_video","kind":"video","direction":"out"}]));
        assert!(v["paths"]["/events/set_prompt"].is_object());
    }

    #[test]
    fn component_names() {
        assert_eq!(component_name("state_update"), "StateUpdate");
        assert_eq!(component_name("command_error"), "CommandError");
    }
}
