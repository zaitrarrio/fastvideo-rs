//! `reactor_wire.v1` (prost) and the legacy v0 JSON codec, sniffed per
//! connection (reactor §2, design §5.7).
//!
//! Both encodings share one vocabulary: [`ClientMsg`] in, [`ServerMsg`] out.
//!
//! - **v1**: binary protobuf. `Data*Message` on the `"data"` channel,
//!   `Control*Message` on `"control"`.
//! - **v0**: JSON text. Every platform message rides `"data"` under
//!   `{"scope":"runtime"}`, commands under `{"scope":"application"}`; the
//!   `"control"` channel carries only the track verbs. Commands are not
//!   correlated, so bodyless acks and command errors are never sent (RT's
//!   `send_command_ack` skips v0).
//!
//! The first inbound frame on either channel latches the version for the
//! life of the connection ([`sniff`]): a text frame, or a binary frame whose
//! first non-space byte is `{` or `[`, is v0; anything else is v1.

use std::collections::BTreeMap;

use bytes::Bytes;
use fastvideo_webrtc::channel::{ChannelMessage, REACTOR_CONTROL, REACTOR_DATA};
use prost::Message as _;
use prost_types::value::Kind;
use serde_json::{json, Map, Value};

use crate::pb;

/// Wire encoding of one connection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum WireVersion {
    V0,
    V1,
}

impl WireVersion {
    /// The codec the `Reactor-WebRTC-Version` offer header seeds: `"1.0"`,
    /// absent and unknown values all map to v0 (RT `transport/webrtc/version.py`).
    pub fn from_header(_h: Option<&str>) -> Self {
        WireVersion::V0
    }
}

/// Which channel a frame arrived on or is sent on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    Data,
    Control,
}

impl Channel {
    pub fn from_label(label: &str) -> Option<Self> {
        match label {
            REACTOR_DATA => Some(Channel::Data),
            REACTOR_CONTROL => Some(Channel::Control),
            _ => None,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Channel::Data => REACTOR_DATA,
            Channel::Control => REACTOR_CONTROL,
        }
    }
}

/// The version a first inbound frame implies.
pub fn sniff(msg: &ChannelMessage) -> WireVersion {
    if !msg.binary {
        return WireVersion::V0;
    }
    match msg.data.iter().find(|b| !b.is_ascii_whitespace()) {
        Some(b'{') | Some(b'[') => WireVersion::V0,
        _ => WireVersion::V1,
    }
}

/// An upload reference inside a command (`Command.uploads[param]`).
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct UploadRef {
    #[serde(default)]
    pub upload_id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub mime_type: String,
    #[serde(default)]
    pub size: i64,
}

/// Every client → runtime message, in either encoding.
#[derive(Clone, Debug, PartialEq)]
pub enum ClientMsg {
    Ping,
    RequestSchema { request_id: String },
    FileUploaded { upload: UploadRef },
    RequestClip { request_id: String, duration_seconds: f64 },
    RequestRecording { request_id: String },
    PublishTrack { request_id: String, name: String },
    PauseTrack { name: String },
    ResumeTrack { name: String },
    UnpublishTrack { name: String },
    /// A client-sent `Error` payload (decoded, never routed).
    Error { code: String, message: String },
    /// A model command. `request_id` is `None` for v0 (uncorrelated) and for
    /// a v1 command that carried none.
    Command {
        request_id: Option<String>,
        name: String,
        data: Map<String, Value>,
        uploads: BTreeMap<String, UploadRef>,
    },
}

/// Every runtime → client message.
#[derive(Clone, Debug, PartialEq)]
pub enum ServerMsg {
    ModelSchema { request_id: String, openapi: Value },
    ClipFailed { request_id: String, reason: String },
    Moderation { action: String, message: String },
    SessionEnded { reason: String },
    PublishTrackOk { request_id: String },
    PublishTrackError { request_id: String, code: String, message: String },
    /// A model message: the correlated reply to a command when `request_id`
    /// is set, else a notification/broadcast.
    Model { request_id: Option<String>, kind: String, data: Value },
    /// Bodyless ack of a command whose handler returned nothing (v1 only).
    CommandAck { request_id: String },
    /// A failed command (v1 only).
    CommandError { request_id: String, code: String, message: String },
}

impl ServerMsg {
    /// A notification-style model message.
    pub fn broadcast(kind: impl Into<String>, data: Value) -> Self {
        ServerMsg::Model { request_id: None, kind: kind.into(), data }
    }
}

/// Why a frame could not be decoded (it is dropped with a warning, as RT).
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum WireError {
    #[error("frame on unknown channel {0:?}")]
    UnknownChannel(String),
    #[error("protobuf: {0}")]
    Proto(String),
    #[error("json: {0}")]
    Json(String),
    #[error("message has no payload")]
    Empty,
    #[error("unknown message type {0:?}")]
    UnknownType(String),
}

/// Decode one inbound frame with the connection's latched version.
pub fn decode(version: WireVersion, msg: &ChannelMessage) -> Result<ClientMsg, WireError> {
    let ch = Channel::from_label(&msg.label)
        .ok_or_else(|| WireError::UnknownChannel(msg.label.clone()))?;
    match version {
        WireVersion::V1 => decode_v1(ch, &msg.data),
        WireVersion::V0 => {
            let v: Value =
                serde_json::from_slice(&msg.data).map_err(|e| WireError::Json(e.to_string()))?;
            decode_v0(ch, &v)
        }
    }
}

/// Encode one outbound message. `None`: the message does not exist in this
/// version (v0 has no command acks or command errors).
pub fn encode(version: WireVersion, msg: &ServerMsg) -> Option<ChannelMessage> {
    match version {
        WireVersion::V1 => {
            let (ch, bytes) = encode_v1(msg);
            Some(ChannelMessage::binary(ch.label(), Bytes::from(bytes)))
        }
        WireVersion::V0 => {
            let (ch, v) = encode_v0(msg)?;
            Some(ChannelMessage::text(ch.label(), v.to_string()))
        }
    }
}

// ---- v1 --------------------------------------------------------------------

fn decode_v1(ch: Channel, data: &[u8]) -> Result<ClientMsg, WireError> {
    use pb::control_client_message::Payload as C;
    use pb::data_client_message::Payload as D;
    match ch {
        Channel::Control => {
            let m = pb::ControlClientMessage::decode(data)
                .map_err(|e| WireError::Proto(e.to_string()))?;
            let rid = m.request_id;
            Ok(match m.payload.ok_or(WireError::Empty)? {
                C::Ping(_) => ClientMsg::Ping,
                C::RequestSchema(_) => ClientMsg::RequestSchema { request_id: rid },
                C::FileUploaded(f) => ClientMsg::FileUploaded {
                    upload: UploadRef {
                        upload_id: f.upload_id,
                        name: f.name,
                        mime_type: f.mime_type,
                        size: f.size,
                    },
                },
                C::RequestClip(c) => ClientMsg::RequestClip {
                    request_id: rid,
                    duration_seconds: c.duration_seconds,
                },
                C::RequestRecording(_) => ClientMsg::RequestRecording { request_id: rid },
                C::PublishTrack(t) => ClientMsg::PublishTrack { request_id: rid, name: t.name },
                C::PauseTrack(t) => ClientMsg::PauseTrack { name: t.name },
                C::ResumeTrack(t) => ClientMsg::ResumeTrack { name: t.name },
                C::UnpublishTrack(t) => ClientMsg::UnpublishTrack { name: t.name },
                C::Error(e) => ClientMsg::Error { code: e.code, message: e.message },
            })
        }
        Channel::Data => {
            let m = pb::DataClientMessage::decode(data)
                .map_err(|e| WireError::Proto(e.to_string()))?;
            match m.payload.ok_or(WireError::Empty)? {
                D::Command(c) => Ok(ClientMsg::Command {
                    request_id: (!m.request_id.is_empty()).then_some(m.request_id),
                    name: c.r#type,
                    data: c.data.map(struct_to_map).unwrap_or_default(),
                    uploads: c
                        .uploads
                        .into_iter()
                        .map(|(k, u)| {
                            (
                                k,
                                UploadRef {
                                    upload_id: u.upload_id,
                                    name: u.name,
                                    mime_type: u.mime_type,
                                    size: u.size,
                                },
                            )
                        })
                        .collect(),
                }),
                D::Error(e) => Ok(ClientMsg::Error { code: e.code, message: e.message }),
            }
        }
    }
}

fn kind(k: pb::MessageKind) -> i32 {
    k as i32
}

fn control(request_id: String, k: pb::MessageKind, p: pb::control_server_message::Payload) -> Vec<u8> {
    pb::ControlServerMessage { request_id, kind: kind(k), payload: Some(p) }.encode_to_vec()
}

fn encode_v1(msg: &ServerMsg) -> (Channel, Vec<u8>) {
    use pb::control_server_message::Payload as C;
    use pb::data_server_message::Payload as D;
    use pb::MessageKind as K;
    match msg.clone() {
        ServerMsg::ModelSchema { request_id, openapi } => (
            Channel::Control,
            control(
                request_id,
                K::Response,
                C::ModelSchema(pb::ModelSchema { openapi: Some(json_to_struct(&openapi)) }),
            ),
        ),
        ServerMsg::ClipFailed { request_id, reason } => (
            Channel::Control,
            control(request_id, K::Response, C::ClipFailed(pb::ClipFailed { reason })),
        ),
        ServerMsg::Moderation { action, message } => (
            Channel::Control,
            control(
                String::new(),
                K::Notification,
                C::Moderation(pb::Moderation { action, message, ..Default::default() }),
            ),
        ),
        ServerMsg::SessionEnded { reason } => (
            Channel::Control,
            control(String::new(), K::Notification, C::SessionEnded(pb::SessionEnded { reason })),
        ),
        // RT's encode_publish_response leaves `name` empty (reactor §2.1).
        ServerMsg::PublishTrackOk { request_id } => (
            Channel::Control,
            control(request_id, K::Response, C::PublishTrack(pb::PublishTrackResponse::default())),
        ),
        ServerMsg::PublishTrackError { request_id, code, message } => (
            Channel::Control,
            control(request_id, K::Response, C::Error(pb::Error { code, message })),
        ),
        ServerMsg::Model { request_id, kind: t, data } => {
            let (rid, k) = match request_id {
                Some(r) => (r, K::Response),
                None => (String::new(), K::Notification),
            };
            let m = pb::DataServerMessage {
                request_id: rid,
                kind: kind(k),
                payload: Some(D::Message(pb::ModelMessage {
                    r#type: t,
                    data: Some(json_to_struct(&data)),
                })),
            };
            (Channel::Data, m.encode_to_vec())
        }
        ServerMsg::CommandAck { request_id } => (
            Channel::Data,
            pb::DataServerMessage { request_id, kind: kind(K::Response), payload: None }
                .encode_to_vec(),
        ),
        ServerMsg::CommandError { request_id, code, message } => (
            Channel::Data,
            pb::DataServerMessage {
                request_id,
                kind: kind(K::Response),
                payload: Some(D::Error(pb::Error { code, message })),
            }
            .encode_to_vec(),
        ),
    }
}

// ---- v0 --------------------------------------------------------------------

fn obj(v: &Value) -> Map<String, Value> {
    v.as_object().cloned().unwrap_or_default()
}

fn str_of(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::as_str).unwrap_or_default().to_owned()
}

fn decode_v0(ch: Channel, v: &Value) -> Result<ClientMsg, WireError> {
    match ch {
        Channel::Data => {
            let scope = v.get("scope").and_then(Value::as_str).unwrap_or_default();
            let inner = v.get("data").cloned().unwrap_or(Value::Null);
            let t = str_of(&inner, "type");
            let body = inner.get("data").cloned().unwrap_or(Value::Null);
            match scope {
                "application" => Ok(ClientMsg::Command {
                    request_id: None,
                    name: t,
                    data: obj(&body),
                    uploads: inner
                        .get("uploads")
                        .and_then(|u| serde_json::from_value(u.clone()).ok())
                        .unwrap_or_default(),
                }),
                "runtime" => match t.as_str() {
                    "ping" => Ok(ClientMsg::Ping),
                    "requestSchema" => Ok(ClientMsg::RequestSchema { request_id: String::new() }),
                    "fileUploaded" => Ok(ClientMsg::FileUploaded {
                        upload: serde_json::from_value(body).unwrap_or_default(),
                    }),
                    "requestClip" => Ok(ClientMsg::RequestClip {
                        request_id: String::new(),
                        duration_seconds: body
                            .get("duration_seconds")
                            .and_then(Value::as_f64)
                            .unwrap_or_default(),
                    }),
                    "requestRecording" => {
                        Ok(ClientMsg::RequestRecording { request_id: String::new() })
                    }
                    other => Err(WireError::UnknownType(other.to_owned())),
                },
                other => Err(WireError::UnknownType(format!("scope {other}"))),
            }
        }
        Channel::Control => {
            let name = v.get("data").map(|d| str_of(d, "name")).unwrap_or_default();
            let rid = str_of(v, "request_id");
            match (str_of(v, "type").as_str(), v.get("method"), v.get("event")) {
                ("request", Some(Value::String(m)), _) if m == "publish_track" => {
                    Ok(ClientMsg::PublishTrack { request_id: rid, name })
                }
                ("notification", _, Some(Value::String(e))) => match e.as_str() {
                    "pause_track" => Ok(ClientMsg::PauseTrack { name }),
                    "resume_track" => Ok(ClientMsg::ResumeTrack { name }),
                    "unpublish_track" => Ok(ClientMsg::UnpublishTrack { name }),
                    other => Err(WireError::UnknownType(other.to_owned())),
                },
                (t, _, _) => Err(WireError::UnknownType(t.to_owned())),
            }
        }
    }
}

fn runtime(t: &str, data: Value) -> Value {
    json!({"scope": "runtime", "data": {"type": t, "data": data}})
}

fn encode_v0(msg: &ServerMsg) -> Option<(Channel, Value)> {
    Some(match msg.clone() {
        ServerMsg::ModelSchema { openapi, .. } => (Channel::Data, runtime("modelSchema", openapi)),
        ServerMsg::ClipFailed { reason, .. } => {
            (Channel::Data, runtime("clipFailed", json!({"reason": reason})))
        }
        ServerMsg::Moderation { action, message } => (
            Channel::Data,
            runtime(
                "moderation",
                json!({"action": action, "input_kind": "", "command": "", "categories": [], "message": message}),
            ),
        ),
        ServerMsg::SessionEnded { reason } => {
            (Channel::Data, runtime("sessionEnded", json!({"reason": reason})))
        }
        ServerMsg::PublishTrackOk { request_id } => (
            Channel::Control,
            json!({"type": "response", "method": "publish_track", "request_id": request_id, "data": {}}),
        ),
        ServerMsg::PublishTrackError { request_id, code, message } => (
            Channel::Control,
            json!({"type": "response", "method": "publish_track", "request_id": request_id,
                   "error": {"code": code, "message": message}}),
        ),
        // request_id is dropped in v0.
        ServerMsg::Model { kind, data, .. } => (
            Channel::Data,
            json!({"scope": "application", "data": {"type": kind, "data": data}}),
        ),
        ServerMsg::CommandAck { .. } | ServerMsg::CommandError { .. } => return None,
    })
}

// ---- google.protobuf.Struct <-> JSON ---------------------------------------

/// JSON → `Struct`. A non-object becomes an empty struct. Numbers travel as
/// `double` (as RT's `json_format`; integers above 2^53 lose precision).
pub fn json_to_struct(v: &Value) -> prost_types::Struct {
    prost_types::Struct {
        fields: v
            .as_object()
            .map(|m| m.iter().map(|(k, v)| (k.clone(), json_to_value(v))).collect())
            .unwrap_or_default(),
    }
}

fn json_to_value(v: &Value) -> prost_types::Value {
    let kind = match v {
        Value::Null => Kind::NullValue(0),
        Value::Bool(b) => Kind::BoolValue(*b),
        Value::Number(n) => Kind::NumberValue(n.as_f64().unwrap_or_default()),
        Value::String(s) => Kind::StringValue(s.clone()),
        Value::Array(a) => Kind::ListValue(prost_types::ListValue {
            values: a.iter().map(json_to_value).collect(),
        }),
        Value::Object(_) => Kind::StructValue(json_to_struct(v)),
    };
    prost_types::Value { kind: Some(kind) }
}

/// `Struct` → JSON object. Whole-number doubles come back as integers so
/// `{"seed": 7}` round-trips as `7`, not `7.0`.
pub fn struct_to_map(s: prost_types::Struct) -> Map<String, Value> {
    s.fields.into_iter().map(|(k, v)| (k, value_to_json(v))).collect()
}

fn value_to_json(v: prost_types::Value) -> Value {
    match v.kind {
        None | Some(Kind::NullValue(_)) => Value::Null,
        Some(Kind::BoolValue(b)) => Value::Bool(b),
        Some(Kind::NumberValue(n)) => {
            if n.fract() == 0.0 && n.abs() < 9.007_199_254_740_992e15 {
                Value::from(n as i64)
            } else {
                serde_json::Number::from_f64(n).map(Value::Number).unwrap_or(Value::Null)
            }
        }
        Some(Kind::StringValue(s)) => Value::String(s),
        Some(Kind::ListValue(l)) => Value::Array(l.values.into_iter().map(value_to_json).collect()),
        Some(Kind::StructValue(s)) => Value::Object(struct_to_map(s)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pb::control_client_message::Payload as C;

    fn v1_control(p: C, rid: &str) -> ChannelMessage {
        let m = pb::ControlClientMessage {
            request_id: rid.into(),
            kind: pb::MessageKind::Request as i32,
            payload: Some(p),
        };
        ChannelMessage::binary(REACTOR_CONTROL, m.encode_to_vec())
    }

    #[test]
    fn sniff_rules() {
        assert_eq!(sniff(&ChannelMessage::text("data", "{}")), WireVersion::V0);
        assert_eq!(sniff(&ChannelMessage::binary("data", b"  {\"a\":1}".to_vec())), WireVersion::V0);
        assert_eq!(sniff(&ChannelMessage::binary("data", b"[1]".to_vec())), WireVersion::V0);
        assert_eq!(sniff(&v1_control(C::Ping(pb::Ping {}), "")), WireVersion::V1);
        // An empty protobuf (all defaults) is v1 too.
        assert_eq!(sniff(&ChannelMessage::binary("control", Vec::<u8>::new())), WireVersion::V1);
        assert_eq!(WireVersion::from_header(Some("1.0")), WireVersion::V0);
        assert_eq!(WireVersion::from_header(None), WireVersion::V0);
    }

    /// Every client oneof arm decodes (v1).
    #[test]
    fn v1_client_arms() {
        let cases: Vec<(C, ClientMsg)> = vec![
            (C::Ping(pb::Ping {}), ClientMsg::Ping),
            (C::RequestSchema(pb::RequestSchema {}), ClientMsg::RequestSchema { request_id: "ctrl_1".into() }),
            (
                C::FileUploaded(pb::FileUploaded { upload_id: "u".into(), name: "n".into(), mime_type: "image/png".into(), size: 3 }),
                ClientMsg::FileUploaded { upload: UploadRef { upload_id: "u".into(), name: "n".into(), mime_type: "image/png".into(), size: 3 } },
            ),
            (C::RequestClip(pb::RequestClip { duration_seconds: 2.5 }), ClientMsg::RequestClip { request_id: "ctrl_1".into(), duration_seconds: 2.5 }),
            (C::RequestRecording(pb::RequestRecording {}), ClientMsg::RequestRecording { request_id: "ctrl_1".into() }),
            (C::PublishTrack(pb::PublishTrack { name: "cam".into() }), ClientMsg::PublishTrack { request_id: "ctrl_1".into(), name: "cam".into() }),
            (C::PauseTrack(pb::PauseTrack { name: "main_video".into() }), ClientMsg::PauseTrack { name: "main_video".into() }),
            (C::ResumeTrack(pb::ResumeTrack { name: "main_video".into() }), ClientMsg::ResumeTrack { name: "main_video".into() }),
            (C::UnpublishTrack(pb::UnpublishTrack { name: "cam".into() }), ClientMsg::UnpublishTrack { name: "cam".into() }),
            (C::Error(pb::Error { code: "x".into(), message: "y".into() }), ClientMsg::Error { code: "x".into(), message: "y".into() }),
        ];
        for (p, want) in cases {
            assert_eq!(decode(WireVersion::V1, &v1_control(p, "ctrl_1")).unwrap(), want);
        }
        let cmd = pb::DataClientMessage {
            request_id: "data_3".into(),
            kind: pb::MessageKind::Request as i32,
            payload: Some(pb::data_client_message::Payload::Command(pb::Command {
                r#type: "enqueue".into(),
                data: Some(json_to_struct(&json!({"prompt": "a cat", "seed": 7, "seconds": 5.5, "x": [true, null]}))),
                uploads: [("image".to_owned(), pb::UploadReference { upload_id: "u1".into(), ..Default::default() })].into(),
            })),
        };
        let got = decode(WireVersion::V1, &ChannelMessage::binary(REACTOR_DATA, cmd.encode_to_vec())).unwrap();
        let ClientMsg::Command { request_id, name, data, uploads } = got else { panic!() };
        assert_eq!(request_id.as_deref(), Some("data_3"));
        assert_eq!(name, "enqueue");
        assert_eq!(Value::Object(data), json!({"prompt": "a cat", "seed": 7, "seconds": 5.5, "x": [true, null]}));
        assert_eq!(uploads["image"].upload_id, "u1");
        // An undecodable frame is an error (dropped by the gateway).
        assert!(decode(WireVersion::V1, &ChannelMessage::binary(REACTOR_DATA, vec![0xff, 0xff])).is_err());
        assert!(decode(WireVersion::V1, &ChannelMessage::binary("other", vec![])).is_err());
    }

    /// Every server arm encodes (v1) onto the right channel and round-trips.
    #[test]
    fn v1_server_arms() {
        use pb::control_server_message::Payload as CS;
        use pb::data_server_message::Payload as DS;
        let ctl = |m: &ServerMsg| {
            let c = encode(WireVersion::V1, m).unwrap();
            assert_eq!(c.label, REACTOR_CONTROL);
            assert!(c.binary);
            pb::ControlServerMessage::decode(&c.data[..]).unwrap()
        };
        let data = |m: &ServerMsg| {
            let c = encode(WireVersion::V1, m).unwrap();
            assert_eq!(c.label, REACTOR_DATA);
            pb::DataServerMessage::decode(&c.data[..]).unwrap()
        };
        let m = ctl(&ServerMsg::ModelSchema { request_id: "ctrl_1".into(), openapi: json!({"openapi": "3.1.0"}) });
        assert_eq!(m.request_id, "ctrl_1");
        assert_eq!(m.kind, pb::MessageKind::Response as i32);
        let Some(CS::ModelSchema(s)) = m.payload else { panic!() };
        assert_eq!(Value::Object(struct_to_map(s.openapi.unwrap())), json!({"openapi": "3.1.0"}));
        assert!(matches!(ctl(&ServerMsg::ClipFailed { request_id: "r".into(), reason: "recording disabled".into() }).payload, Some(CS::ClipFailed(c)) if c.reason == "recording disabled"));
        let m = ctl(&ServerMsg::SessionEnded { reason: "bye".into() });
        assert_eq!(m.kind, pb::MessageKind::Notification as i32);
        assert!(matches!(m.payload, Some(CS::SessionEnded(s)) if s.reason == "bye"));
        assert!(matches!(ctl(&ServerMsg::Moderation { action: "terminate".into(), message: "m".into() }).payload, Some(CS::Moderation(x)) if x.action == "terminate"));
        assert!(matches!(ctl(&ServerMsg::PublishTrackOk { request_id: "r".into() }).payload, Some(CS::PublishTrack(p)) if p.name.is_empty()));
        assert!(matches!(ctl(&ServerMsg::PublishTrackError { request_id: "r".into(), code: "publish_refused".into(), message: "no".into() }).payload, Some(CS::Error(e)) if e.code == "publish_refused"));

        let m = data(&ServerMsg::Model { request_id: Some("data_1".into()), kind: "clip_queued".into(), data: json!({"clip": {"frames": 125}}) });
        assert_eq!((m.request_id.as_str(), m.kind), ("data_1", pb::MessageKind::Response as i32));
        let Some(DS::Message(mm)) = m.payload else { panic!() };
        assert_eq!(mm.r#type, "clip_queued");
        assert_eq!(Value::Object(struct_to_map(mm.data.unwrap())), json!({"clip": {"frames": 125}}));
        let m = data(&ServerMsg::broadcast("state_update", json!({})));
        assert_eq!((m.request_id.as_str(), m.kind), ("", pb::MessageKind::Notification as i32));
        let m = data(&ServerMsg::CommandAck { request_id: "data_2".into() });
        assert_eq!((m.request_id.as_str(), m.payload), ("data_2", None));
        assert!(matches!(data(&ServerMsg::CommandError { request_id: "d".into(), code: "invalid_command".into(), message: "x".into() }).payload, Some(DS::Error(e)) if e.code == "invalid_command"));
    }

    /// The v0 envelopes of reactor §2.2.
    #[test]
    fn v0_envelopes() {
        let d = |s: &str| decode(WireVersion::V0, &ChannelMessage::text(REACTOR_DATA, s)).unwrap();
        let c = |s: &str| decode(WireVersion::V0, &ChannelMessage::text(REACTOR_CONTROL, s)).unwrap();
        let ClientMsg::Command { request_id, name, data, uploads } = d(r#"{"scope":"application","data":{"type":"set_prompt","data":{"prompt":"hi"},"uploads":{"image":{"upload_id":"u","name":"a.png","mime_type":"image/png","size":4}}}}"#) else { panic!() };
        assert_eq!((request_id, name.as_str()), (None, "set_prompt"));
        assert_eq!(Value::Object(data), json!({"prompt": "hi"}));
        assert_eq!(uploads["image"].size, 4);
        assert_eq!(d(r#"{"scope":"runtime","data":{"type":"ping","data":{}}}"#), ClientMsg::Ping);
        assert_eq!(d(r#"{"scope":"runtime","data":{"type":"requestSchema","data":{}}}"#), ClientMsg::RequestSchema { request_id: String::new() });
        assert_eq!(d(r#"{"scope":"runtime","data":{"type":"requestClip","data":{"duration_seconds":3}}}"#), ClientMsg::RequestClip { request_id: String::new(), duration_seconds: 3.0 });
        assert_eq!(d(r#"{"scope":"runtime","data":{"type":"requestRecording"}}"#), ClientMsg::RequestRecording { request_id: String::new() });
        assert!(matches!(d(r#"{"scope":"runtime","data":{"type":"fileUploaded","data":{"upload_id":"u","name":"n","mime_type":"m","size":1}}}"#), ClientMsg::FileUploaded { .. }));
        assert_eq!(c(r#"{"type":"request","method":"publish_track","request_id":"r1","data":{"name":"cam"}}"#), ClientMsg::PublishTrack { request_id: "r1".into(), name: "cam".into() });
        assert_eq!(c(r#"{"type":"notification","event":"resume_track","data":{"name":"main_video"}}"#), ClientMsg::ResumeTrack { name: "main_video".into() });
        assert_eq!(c(r#"{"type":"notification","event":"pause_track","data":{"name":"main_audio"}}"#), ClientMsg::PauseTrack { name: "main_audio".into() });
        assert_eq!(c(r#"{"type":"notification","event":"unpublish_track","data":{"name":"cam"}}"#), ClientMsg::UnpublishTrack { name: "cam".into() });
        assert!(decode(WireVersion::V0, &ChannelMessage::text(REACTOR_DATA, "not json")).is_err());

        let e = |m: ServerMsg| {
            let c = encode(WireVersion::V0, &m).unwrap();
            assert!(!c.binary);
            (c.label.clone(), serde_json::from_slice::<Value>(&c.data).unwrap())
        };
        assert_eq!(e(ServerMsg::Model { request_id: Some("x".into()), kind: "state_update".into(), data: json!({"a": 1}) }),
            (REACTOR_DATA.to_owned(), json!({"scope":"application","data":{"type":"state_update","data":{"a":1}}})));
        assert_eq!(e(ServerMsg::ModelSchema { request_id: String::new(), openapi: json!({"openapi":"3.1.0"}) }).1,
            json!({"scope":"runtime","data":{"type":"modelSchema","data":{"openapi":"3.1.0"}}}));
        assert_eq!(e(ServerMsg::ClipFailed { request_id: String::new(), reason: "r".into() }).1,
            json!({"scope":"runtime","data":{"type":"clipFailed","data":{"reason":"r"}}}));
        assert_eq!(e(ServerMsg::SessionEnded { reason: "r".into() }).1,
            json!({"scope":"runtime","data":{"type":"sessionEnded","data":{"reason":"r"}}}));
        assert_eq!(e(ServerMsg::Moderation { action: "terminate".into(), message: "m".into() }).1["data"]["type"], "moderation");
        assert_eq!(e(ServerMsg::PublishTrackOk { request_id: "r".into() }),
            (REACTOR_CONTROL.to_owned(), json!({"type":"response","method":"publish_track","request_id":"r","data":{}})));
        assert_eq!(e(ServerMsg::PublishTrackError { request_id: "r".into(), code: "c".into(), message: "m".into() }).1["error"],
            json!({"code":"c","message":"m"}));
        assert!(encode(WireVersion::V0, &ServerMsg::CommandAck { request_id: "r".into() }).is_none());
        assert!(encode(WireVersion::V0, &ServerMsg::CommandError { request_id: "r".into(), code: "c".into(), message: "m".into() }).is_none());
    }
}
