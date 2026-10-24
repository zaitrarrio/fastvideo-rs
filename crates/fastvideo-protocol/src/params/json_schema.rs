//! [`Param`]s from a JSON Schema (draft 2020-12) object schema: the fal
//! apps publish their inputs as JSON Schema (`/fal/schema`), so their
//! [`super::ParamSpec`]s are read from the same document their parsers are
//! tested against.

use serde_json::{Map, Value};

use super::{Domain, MediaField, Param, ParamType, Support};

/// One [`Param`] per property of `schema` (in `x-fal-order-properties`
/// order when present). `served` is the schema the model is served with;
/// `wire`, when given, is the surface's full schema: a property whose
/// domain differs there carries it as [`Param::wire`]. `map(name)` gives
/// the internal field and the support of a property.
pub fn params_from_json_schema(
    served: &Value,
    wire: Option<&Value>,
    map: &dyn Fn(&str) -> (String, Support),
) -> Vec<Param> {
    let empty = Map::new();
    let props = served["properties"].as_object().unwrap_or(&empty);
    let required: Vec<&str> = served["required"]
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let order: Vec<String> = match served["x-fal-order-properties"].as_array() {
        Some(a) => a
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
        None => props.keys().cloned().collect(),
    };
    order
        .iter()
        .filter_map(|name| props.get(name).map(|p| (name, p)))
        .map(|(name, p)| {
            let (maps_to, support) = map(name);
            let (ty, nullable, domain) = describe(p);
            let mut param = Param::new(name.clone(), ty, maps_to)
                .domain(domain.clone())
                .support(support);
            param.required = required.contains(&name.as_str());
            param.nullable = nullable;
            if let Some(d) = p.get("default") {
                param.default = Some(d.clone());
            }
            if let Some(w) = wire.and_then(|w| w["properties"].get(name)) {
                param = param.wire(describe(w).2);
            }
            param
        })
        .collect()
}

/// `(type, nullable, domain)` of one property schema.
fn describe(p: &Value) -> (ParamType, bool, Domain) {
    // `anyOf: [{...}, {"type": "null"}]`: the non-null member, nullable.
    let (s, nullable) = match p["anyOf"].as_array() {
        Some(alts) => {
            let null = alts.iter().any(|a| a["type"] == "null");
            let first = alts.iter().find(|a| a["type"] != "null").unwrap_or(p);
            (first, null)
        }
        None => (p, false),
    };
    let enum_values = p["enum"].as_array().or_else(|| s["enum"].as_array());
    let ty = match s["type"].as_str() {
        Some("integer") => ParamType::Integer,
        Some("number") => ParamType::Number,
        Some("string") => ParamType::String,
        Some("boolean") => ParamType::Boolean,
        Some("array") => ParamType::Array,
        Some("object") => ParamType::Object,
        _ => match enum_values {
            Some(v) if v.iter().all(Value::is_number) => ParamType::Integer,
            Some(v) if v.iter().all(Value::is_string) => ParamType::String,
            Some(_) => ParamType::IntegerOrString,
            None => ParamType::Any,
        },
    };
    if let Some(values) = enum_values {
        return (
            ty,
            nullable,
            Domain::OneOf {
                values: values.clone(),
            },
        );
    }
    let domain = match ty {
        ParamType::Integer => Domain::Int {
            min: s["minimum"].as_i64(),
            max: s["maximum"].as_i64(),
        },
        ParamType::Number => match (s["exclusiveMinimum"].as_f64(), s["minimum"].as_f64()) {
            (Some(m), _) => Domain::Number {
                min: Some(m),
                max: s["maximum"].as_f64(),
                exclusive_min: true,
            },
            (None, min) => Domain::Number {
                min,
                max: s["maximum"].as_f64(),
                exclusive_min: false,
            },
        },
        ParamType::String => match media(p) {
            Some(m) => Domain::Media { media: m },
            None => {
                let min_chars = s["minLength"].as_u64().unwrap_or(0) as usize;
                let max_chars = s["maxLength"].as_u64().map(|n| n as usize);
                let non_blank = s["pattern"] == "\\S";
                if min_chars == 0 && max_chars.is_none() && !non_blank {
                    Domain::Any
                } else {
                    Domain::Text {
                        min_chars,
                        max_chars,
                        non_blank,
                    }
                }
            }
        },
        ParamType::Array => match s["maxItems"].as_u64() {
            Some(m) => Domain::Items {
                max: u32::try_from(m).unwrap_or(u32::MAX),
            },
            None => Domain::Any,
        },
        _ => Domain::Any,
    };
    (ty, nullable, domain)
}

/// The console's `x-fv-media` vendor key.
fn media(p: &Value) -> Option<MediaField> {
    match p["x-fv-media"].as_str()? {
        "image" => Some(MediaField::Image),
        "video" => Some(MediaField::Video),
        "audio" => Some(MediaField::Audio),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn reads_types_bounds_enums_and_nullability() {
        let schema = json!({
            "type": "object",
            "required": ["prompt"],
            "x-fal-order-properties": ["prompt", "duration", "resolution", "seed", "image_url", "refs", "ltx"],
            "properties": {
                "prompt": {"type": "string", "minLength": 1, "maxLength": 50000},
                "duration": {"type": "integer", "minimum": 5, "maximum": 15, "default": 5},
                "resolution": {"type": "string", "enum": ["480P", "768P"], "default": "768P"},
                "seed": {"anyOf": [{"type": "integer", "minimum": 0}, {"type": "null"}], "default": null},
                "image_url": {"anyOf": [{"type": "string", "minLength": 1, "pattern": "\\S"}, {"type": "null"}], "x-fv-media": "image"},
                "refs": {"type": "array", "maxItems": 9},
                "ltx": {"enum": [6, 8, "auto"]}
            }
        });
        let wire =
            json!({"properties": {"duration": {"type": "integer", "minimum": 5, "maximum": 20}}});
        let ps =
            params_from_json_schema(&schema, Some(&wire), &|n| (n.to_owned(), Support::Honoured));
        let names: Vec<&str> = ps.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "prompt",
                "duration",
                "resolution",
                "seed",
                "image_url",
                "refs",
                "ltx"
            ]
        );
        assert!(ps[0].required);
        assert_eq!(ps[0].domain, Domain::chars(50000));
        assert_eq!(
            ps[1].domain,
            Domain::int(super::super::IntRange::new(5, 15))
        );
        assert_eq!(
            ps[1].wire,
            Some(Domain::int(super::super::IntRange::new(5, 20)))
        );
        assert_eq!(ps[2].domain, Domain::strings(&["480P", "768P"]));
        assert!(ps[3].nullable);
        assert_eq!(ps[3].domain, Domain::int_min(0));
        assert_eq!(ps[4].domain, Domain::media(MediaField::Image));
        assert_eq!(ps[5].domain, Domain::items(9));
        assert_eq!(ps[6].ty, ParamType::IntegerOrString);
    }
}
