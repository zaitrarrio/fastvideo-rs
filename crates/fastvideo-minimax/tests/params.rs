//! Plan E2: the MiniMax create parser enforces the parse-time domains of
//! its `ParamSpec` for every MiniMax model name.

use fastvideo_minimax::create::normalize;
use fastvideo_minimax::params::create_spec;
use fastvideo_minimax::{CreateBody, MiniMaxModel};
use fastvideo_protocol::params::parse_agreement;
use fastvideo_protocol::{ApiError, ModelCaps};
use serde_json::{json, Value};

#[test]
fn create_parser_enforces_the_spec() {
    let caps = ModelCaps::h3("h3-max", true);
    for m in MiniMaxModel::ALL {
        let Value::Object(base) = json!({
            "model": m.as_str(),
            "content": [{"type": "text", "text": "a cat"}],
            "resolution": "768P",
            "duration": 5,
            "ratio": "16:9",
        }) else {
            unreachable!()
        };
        let parse = |v: Value| {
            let b: CreateBody =
                serde_json::from_value(v).map_err(|e| ApiError::invalid(e.to_string()))?;
            normalize(b).map(|_| ())
        };
        // `model` selects the rules themselves; `ratio` `adaptive` is
        // refused for text-to-video (checked with the content).
        let bad = parse_agreement(&create_spec(m, &caps), &base, &parse, &["model", "ratio"]);
        assert!(bad.is_empty(), "{m}: {bad:#?}");
    }
}
