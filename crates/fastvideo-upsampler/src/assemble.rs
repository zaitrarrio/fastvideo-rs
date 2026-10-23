//! Segment prompt assembly per model family (docs/serve/prompt-upsampling.md
//! §Assembly): the showrunner's [`ExpansionOutput`] plus the registry's
//! verbatim descriptors become the prompt the DiT gets.
//!
//! - [`Layout::H3`] (H3, FastH3, Sol-H3, H3-Trim): the Context-IR layout H3
//!   was trained on and fal's H3 Max expander emits:
//!   ```text
//!   integrated_multimodal_description: <location> <Name: descriptor.> <action> <Name says <d>[English] line</d>> <camera.> <lighting.> <style.>
//!   overall_soundscape: <audio>
//!   non_diegetic_music: <music, or None.>
//!   ```
//! - [`Layout::Prose`] (Wan, LTX, LongLive, SF-Wan): the owner's
//!   `assemblePrompt` (docs/serve/upsampler-spec.ts), unchanged while the
//!   prompt fits, plus `Music: …` after the sound when the segment has a
//!   score.
//!
//! **Over the limit** ([`PromptLimit`]: the encoder's tokens by its real
//! tokenizer, else characters) parts are trimmed by priority, never by
//! slicing a string: 1. the style, 2. the music, 3. the descriptors of cast
//! members that neither speak nor act in this segment (oldest registered
//! first) are shortened to their name, 4. the lighting, 5. the camera. The
//! location descriptor, the speaking / acting cast, the action, the
//! dialogue and the sound are kept. Only if those alone are still over the
//! limit is the description cut at the last sentence (else word) that
//! fits, never inside a `<d>…</d>` pair, and the sound line is kept.
//!
//! This deviates from the TS, which drops the *last* part (the style, or the
//! `Sound:` line when the style is empty) and then slices characters.

use crate::director::{EntityKind, ExpansionOutput, Registry, WorldBible};
use crate::rewrite::PromptFamily;
use crate::tokens::{cut_to_fit, PromptLimit};

/// How a family's segment prompt is laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// The owner's prose.
    Prose,
    /// H3's Context-IR sections with `<d>` dialogue.
    H3,
}

impl Layout {
    pub fn for_family(f: PromptFamily) -> Self {
        match f {
            PromptFamily::H3 => Layout::H3,
            _ => Layout::Prose,
        }
    }
}

/// One cast member as placed in the prompt.
#[derive(Clone, Debug)]
struct Member {
    name: String,
    descriptor: String,
    /// Speaks or acts in this segment: never shortened.
    keep: bool,
    /// Registry position (smaller is older).
    age: usize,
    short: bool,
}

/// The parts of one segment prompt and what has been trimmed.
#[derive(Clone, Debug)]
struct Parts {
    location: String,
    cast: Vec<Member>,
    action: String,
    /// `(speaker name, line)`.
    dialogue: Vec<(String, String)>,
    camera: Option<String>,
    lighting: Option<String>,
    audio: String,
    music: Option<String>,
    style: Option<String>,
}

fn nonempty(s: &str) -> Option<String> {
    let t = s.trim();
    (!t.is_empty()).then(|| t.to_owned())
}

/// `s` ending in sentence punctuation (H3 layout).
fn sentence(s: &str) -> String {
    let t = s.trim();
    if t.is_empty() || t.ends_with(['.', '!', '?', '…', '"', '\'', ')']) || t.ends_with("</d>") {
        t.to_owned()
    } else {
        format!("{t}.")
    }
}

/// A music field that says "no music".
fn no_music(m: &str) -> bool {
    let t = m.trim().trim_end_matches('.').to_ascii_lowercase();
    t.is_empty() || t == "none" || t == "no music" || t == "n/a"
}

/// A dialogue line without markup the model may have added itself.
fn clean_line(l: &str) -> String {
    let t = l.replace("<d>", "").replace("</d>", "");
    let t = t.trim();
    let t = t.strip_prefix("[English]").unwrap_or(t).trim();
    t.to_owned()
}

impl Parts {
    fn new(out: &ExpansionOutput, registry: &Registry, world: &WorldBible) -> Self {
        let age = |id: &str| {
            registry
                .values()
                .iter()
                .position(|e| e.id == id)
                .unwrap_or(usize::MAX)
        };
        let cast: Vec<_> = out
            .entity_ids
            .iter()
            .filter_map(|id| registry.get(id))
            .collect();
        let location = cast
            .iter()
            .find(|e| e.kind == EntityKind::Location)
            .map_or_else(String::new, |l| l.descriptor.clone());
        let action_lc = out.action.to_lowercase();
        let speakers: Vec<&str> = out.dialogue.iter().map(|d| d.speaker_id.as_str()).collect();
        let members = cast
            .iter()
            .filter(|e| e.kind != EntityKind::Location)
            .map(|e| Member {
                name: e.name.clone(),
                descriptor: e.descriptor.clone(),
                keep: speakers.contains(&e.id.as_str())
                    || (!e.name.trim().is_empty()
                        && action_lc.contains(&e.name.trim().to_lowercase()))
                    || action_lc.contains(&e.id.to_lowercase()),
                age: age(&e.id),
                short: false,
            })
            .collect();
        let name_of = |id: &str| {
            registry
                .get(id)
                .map_or("a voice", |e| e.name.as_str())
                .to_owned()
        };
        Parts {
            location,
            cast: members,
            action: out.action.clone(),
            dialogue: out
                .dialogue
                .iter()
                .map(|d| (name_of(&d.speaker_id), d.line.clone()))
                .collect(),
            camera: nonempty(&out.camera),
            lighting: nonempty(&out.lighting),
            audio: out.audio.clone(),
            music: Some(out.music.clone()).filter(|m| !no_music(m)),
            style: nonempty(&world.style),
        }
    }

    /// The next trim by priority; `false` when none is left.
    fn trim_once(&mut self) -> bool {
        if self.style.take().is_some() || self.music.take().is_some() {
            return true;
        }
        let next = self
            .cast
            .iter_mut()
            .filter(|m| !m.keep && !m.short)
            .min_by_key(|m| m.age);
        if let Some(m) = next {
            m.short = true;
            return true;
        }
        self.lighting.take().is_some() || self.camera.take().is_some()
    }

    /// The description parts (everything but the sound and music), in order.
    fn front(&self, layout: Layout) -> Vec<String> {
        let mut v = Vec::new();
        let h3 = layout == Layout::H3;
        let fmt = |s: &str| if h3 { sentence(s) } else { s.to_owned() };
        v.push(fmt(&self.location));
        for m in &self.cast {
            v.push(if m.short {
                fmt(&m.name)
            } else {
                fmt(&format!("{}: {}", m.name, m.descriptor))
            });
        }
        v.push(fmt(&self.action));
        for (name, line) in &self.dialogue {
            v.push(if h3 {
                format!("{name} says <d>[English] {}</d>", clean_line(line))
            } else {
                format!("{name} says, \"{line}\"")
            });
        }
        if let Some(c) = &self.camera {
            v.push(if h3 {
                sentence(c)
            } else {
                format!("Camera: {c}")
            });
        }
        if let Some(l) = &self.lighting {
            v.push(if h3 {
                sentence(l)
            } else {
                format!("Lighting: {l}")
            });
        }
        v.into_iter().filter(|p| !p.trim().is_empty()).collect()
    }

    fn sound(&self) -> String {
        format!(
            "{}{}",
            self.audio.trim(),
            if self.dialogue.is_empty() {
                " No spoken words."
            } else {
                ""
            }
        )
        .trim()
        .to_owned()
    }

    /// The prompt with `front` as the description.
    fn render_with(&self, layout: Layout, front: &str) -> String {
        match layout {
            Layout::H3 => {
                let mut desc = front.to_owned();
                if let Some(s) = &self.style {
                    desc = format!("{desc} {}", sentence(s)).trim().to_owned();
                }
                let music = self
                    .music
                    .as_deref()
                    .map_or_else(|| "None.".to_owned(), sentence);
                format!(
                    "integrated_multimodal_description: {desc}\noverall_soundscape: {}\nnon_diegetic_music: {music}",
                    sentence(&self.sound())
                )
            }
            Layout::Prose => {
                let mut parts = vec![front.to_owned(), format!("Sound: {}", self.sound())];
                if let Some(m) = &self.music {
                    parts.push(format!("Music: {m}"));
                }
                if let Some(s) = &self.style {
                    parts.push(s.clone());
                }
                parts
                    .into_iter()
                    .filter(|p| !p.trim().is_empty())
                    .collect::<Vec<_>>()
                    .join(" ")
            }
        }
    }

    fn render(&self, layout: Layout) -> String {
        self.render_with(layout, &self.front(layout).join(" "))
    }
}

/// The segment prompt for `layout` within `limit` (module docs).
pub fn assemble(
    layout: Layout,
    out: &ExpansionOutput,
    registry: &Registry,
    world: &WorldBible,
    limit: &PromptLimit,
) -> String {
    let mut parts = Parts::new(out, registry, world);
    let mut p = parts.render(layout);
    while !limit.fits(&p) {
        if !parts.trim_once() {
            // Last resort: the protected description alone is too long.
            let front = parts.front(layout).join(" ");
            let fits = |f: &str| limit.fits(&parts.render_with(layout, f));
            let cut = cut_to_fit(&front, &fits);
            let p = parts.render_with(layout, &cut);
            // Degenerate: the sound line alone is over the limit. The
            // encoder's limit wins over keeping it whole.
            return if limit.fits(&p) {
                p
            } else {
                cut_to_fit(&p, &|s: &str| limit.fits(s))
            };
        }
        p = parts.render(layout);
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::director::{DialogueLine, Entity};
    use crate::tokens::tests::Words;
    use crate::tokens::TokenCounter;
    use std::sync::Arc;

    fn ent(id: &str, kind: EntityKind, name: &str, d: &str) -> Entity {
        Entity {
            id: id.into(),
            kind,
            name: name.into(),
            descriptor: d.into(),
        }
    }

    fn setup() -> (Registry, WorldBible, ExpansionOutput) {
        let world = WorldBible {
            premise: "p".into(),
            style: "35mm film, muted teal palette".into(),
            entities: vec![
                ent(
                    "gull",
                    EntityKind::Prop,
                    "Gull",
                    "a grey herring gull on the rail",
                ),
                ent(
                    "mara",
                    EntityKind::Character,
                    "Mara",
                    "a wiry woman in her sixties, yellow oilskin coat",
                ),
                ent(
                    "tower",
                    EntityKind::Location,
                    "Lighthouse",
                    "a white lighthouse on black rocks",
                ),
                ent(
                    "tom",
                    EntityKind::Character,
                    "Tom",
                    "a young sailor in a navy peacoat",
                ),
            ],
        };
        let mut r = Registry::default();
        for e in &world.entities {
            r.set(e.clone());
        }
        let out = ExpansionOutput {
            beat_summary: "s".into(),
            entity_ids: vec!["tower".into(), "mara".into(), "gull".into(), "tom".into()],
            new_entities: vec![],
            action: "Mara climbs the spiral stairs, lamp swinging".into(),
            camera: "slow tracking shot upward".into(),
            lighting: "warm lamp glow against blue dusk".into(),
            audio: "wind howling, iron steps clanging".into(),
            music: "a low cello drone".into(),
            dialogue: vec![DialogueLine {
                speaker_id: "mara".into(),
                line: "Light's out again.".into(),
            }],
        };
        (r, world, out)
    }

    #[test]
    fn h3_golden_with_dialogue() {
        let (r, w, out) = setup();
        let p = assemble(Layout::H3, &out, &r, &w, &PromptLimit::chars(10_000));
        assert_eq!(
            p,
            "integrated_multimodal_description: a white lighthouse on black rocks. \
             Mara: a wiry woman in her sixties, yellow oilskin coat. Gull: a grey herring gull on the rail. \
             Tom: a young sailor in a navy peacoat. Mara climbs the spiral stairs, lamp swinging. \
             Mara says <d>[English] Light's out again.</d> slow tracking shot upward. \
             warm lamp glow against blue dusk. 35mm film, muted teal palette.\n\
             overall_soundscape: wind howling, iron steps clanging.\n\
             non_diegetic_music: a low cello drone."
        );
        // No dialogue and no score: "No spoken words." and music None.
        let mut o = out.clone();
        o.dialogue.clear();
        o.music = "none".into();
        let p = assemble(Layout::H3, &o, &r, &w, &PromptLimit::chars(10_000));
        assert!(p.ends_with("overall_soundscape: wind howling, iron steps clanging No spoken words.\nnon_diegetic_music: None."), "{p}");
        // Markup the model added itself is not doubled.
        o.dialogue = vec![DialogueLine {
            speaker_id: "tom".into(),
            line: "<d>[English] Ahoy!</d>".into(),
        }];
        assert!(
            assemble(Layout::H3, &o, &r, &w, &PromptLimit::chars(10_000))
                .contains("Tom says <d>[English] Ahoy!</d>")
        );
    }

    /// The trim order by tokens: style, music, the non-speaking /
    /// non-acting cast (oldest first) to names, lighting, camera.
    #[test]
    fn h3_trims_by_priority_in_tokens() {
        let (r, w, out) = setup();
        let words = |n| PromptLimit {
            counter: Some(Arc::new(Words)),
            max_tokens: n,
            max_chars: 1,
        };
        let full = assemble(Layout::H3, &out, &r, &w, &PromptLimit::chars(10_000));
        let n = Words.count(&full).unwrap();
        assert_eq!(assemble(Layout::H3, &out, &r, &w, &words(n)), full);
        let steps: Vec<String> = (1..n - 12)
            .map(|k| assemble(Layout::H3, &out, &r, &w, &words(n - k)))
            .collect();
        let no_style = &steps[0];
        assert!(
            !no_style.contains("35mm") && no_style.contains("cello"),
            "{no_style}"
        );
        let at = |needle: &str| steps.iter().position(|s| !s.contains(needle)).unwrap();
        // Gull (registered first, silent, not in the action) before Tom.
        assert!(
            at("35mm") < at("cello")
                && at("cello") < at("herring")
                && at("herring") < at("peacoat")
        );
        assert!(at("peacoat") < at("warm lamp") && at("warm lamp") < at("tracking"));
        // Mara speaks and acts: her descriptor stays through all of that.
        let s = &steps[at("tracking")];
        assert!(
            s.contains("Mara: a wiry woman") && s.contains("Gull.") && s.contains("Tom."),
            "{s}"
        );
        assert!(
            s.contains("overall_soundscape: wind howling")
                && s.contains("non_diegetic_music: None."),
            "{s}"
        );
        // Beyond every trim: a cut of the description, never inside <d>…</d>,
        // the sound kept.
        for (k, s) in steps.iter().enumerate() {
            assert!(Words.count(s).unwrap() <= n - 1 - k, "{k}: {s}");
            assert_eq!(s.matches("<d>").count(), s.matches("</d>").count(), "{s}");
            assert!(s.contains("overall_soundscape: wind howling"), "{s}");
        }
        let last = steps.last().unwrap();
        assert!(
            !last.contains("<d>") && last.starts_with("integrated_multimodal_description: a white"),
            "{last}"
        );
    }

    /// The prose layout is the owner's, and the sound is never dropped
    /// (the TS dropped `Sound:` when the style was empty).
    #[test]
    fn prose_golden_and_sound_kept() {
        let (r, w, mut out) = setup();
        out.music.clear();
        out.dialogue.clear();
        out.entity_ids = vec!["tower".into(), "mara".into()];
        let p = assemble(Layout::Prose, &out, &r, &w, &PromptLimit::chars(10_000));
        assert_eq!(
            p,
            "a white lighthouse on black rocks Mara: a wiry woman in her sixties, yellow oilskin coat \
             Mara climbs the spiral stairs, lamp swinging Camera: slow tracking shot upward \
             Lighting: warm lamp glow against blue dusk Sound: wind howling, iron steps clanging No spoken words. \
             35mm film, muted teal palette"
        );
        let w2 = WorldBible {
            style: String::new(),
            ..w.clone()
        };
        let full = assemble(Layout::Prose, &out, &r, &w2, &PromptLimit::chars(10_000));
        let n = full.chars().count();
        // Down to where the sound line alone is the limit.
        for k in 1..(n - 70) {
            let p = assemble(Layout::Prose, &out, &r, &w2, &PromptLimit::chars(n - k));
            assert!(
                p.contains("Sound: wind howling, iron steps clanging No spoken words."),
                "{k}: {p}"
            );
            assert!(
                p.chars().count() <= n - k || !p.contains("Mara climbs"),
                "{k}: {p}"
            );
        }
        // Lighting goes before camera.
        let p = assemble(Layout::Prose, &out, &r, &w2, &PromptLimit::chars(n - 10));
        assert!(!p.contains("Lighting:") && p.contains("Camera:"), "{p}");
        // Music, when there is a score, follows the sound.
        out.music = "a low cello drone".into();
        assert!(
            assemble(Layout::Prose, &out, &r, &w, &PromptLimit::chars(10_000))
                .contains("No spoken words. Music: a low cello drone 35mm")
        );
    }
}
