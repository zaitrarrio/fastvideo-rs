//! LTX-2.5 stage-2 Sol contract (`models/ltx25/RTX5090/attention.py` and
//! `models/ltx2.5-refiner/GB200/refiner.toml` on NVlabs/Sana `sol-engine`).
//!
//! Three distilled refine forwards. Video self-attention on layer 0 stays
//! dense. Layers 1..=47 are Sol-Attn at tau 1.0, then 1.25, then 1.5.
//! Cross-attention and audio self-attention stay dense. The distilled LoRA
//! strength on that profile is 0.8; fusing it is separate from this route.

/// Video self-attention layers in one LTX-2.5 transformer forward.
pub const LAYERS_PER_FORWARD: usize = 48;

/// One tau per stage-2 forward. Layer 0 of that forward does not use it.
pub const STAGE2_TAUS: [f64; 3] = [1.0, 1.25, 1.5];

/// Distilled refiner LoRA multiplier from the GB200 delivery contract.
pub const LORA_STRENGTH: f64 = 0.8;

/// Four sigma endpoints: the three stage-2 updates, then 0.
pub const STAGE2_SIGMAS: [f64; 4] = [0.909375, 0.725, 0.421875, 0.0];

/// How one video self-attention call is supposed to run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Ltx25SolRoute {
    /// Layer 0, and every non-video call.
    Dense,
    /// Layers 1..=47 of a stage-2 forward.
    Sol { tau: f64 },
}

/// `route_for_call`: `(tau, forward, layer)`. `tau` is `None` on the dense layer.
pub fn route_for_call(call_index: usize) -> Result<(Option<f64>, usize, usize), String> {
    let forward = call_index / LAYERS_PER_FORWARD;
    let layer = call_index % LAYERS_PER_FORWARD;
    if forward >= STAGE2_TAUS.len() {
        return Err(format!(
            "ltx2 sol: unexpected stage-2 video attention call {call_index}"
        ));
    }
    let route = route(forward, layer)?;
    let tau = match route {
        Ltx25SolRoute::Dense => None,
        Ltx25SolRoute::Sol { tau } => Some(tau),
    };
    Ok((tau, forward, layer))
}

/// Route for one `(forward, layer)` pair. `layer` is the video block index.
pub fn route(forward: usize, layer: usize) -> Result<Ltx25SolRoute, String> {
    if forward >= STAGE2_TAUS.len() {
        return Err(format!(
            "ltx2 sol: stage-2 forward {forward} is past {} taus",
            STAGE2_TAUS.len()
        ));
    }
    if layer >= LAYERS_PER_FORWARD {
        return Err(format!(
            "ltx2 sol: layer {layer} is past {LAYERS_PER_FORWARD} video blocks"
        ));
    }
    if layer == 0 {
        Ok(Ltx25SolRoute::Dense)
    } else {
        Ok(Ltx25SolRoute::Sol {
            tau: STAGE2_TAUS[forward],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::schedule::STAGE_2_DISTILLED_SIGMA_VALUES;
    use super::*;

    #[test]
    fn sigmas_are_the_distilled_stage_2_tail() {
        assert_eq!(STAGE2_SIGMAS[..3], STAGE_2_DISTILLED_SIGMA_VALUES[..]);
        assert_eq!(*STAGE2_SIGMAS.last().unwrap(), 0.0);
    }

    #[test]
    fn layer_zero_is_dense_and_the_rest_take_that_forwards_tau() {
        assert_eq!(route(0, 0).unwrap(), Ltx25SolRoute::Dense);
        assert_eq!(route(0, 1).unwrap(), Ltx25SolRoute::Sol { tau: 1.0 });
        assert_eq!(route(0, 47).unwrap(), Ltx25SolRoute::Sol { tau: 1.0 });
        assert_eq!(route(1, 0).unwrap(), Ltx25SolRoute::Dense);
        assert_eq!(route(1, 1).unwrap(), Ltx25SolRoute::Sol { tau: 1.25 });
        assert_eq!(route(2, 0).unwrap(), Ltx25SolRoute::Dense);
        assert_eq!(route(2, 47).unwrap(), Ltx25SolRoute::Sol { tau: 1.5 });
        assert!(route(3, 0).is_err());
        assert!(route(0, 48).is_err());
    }

    #[test]
    fn call_index_matches_the_python_divmod() {
        assert_eq!(route_for_call(0).unwrap(), (None, 0, 0));
        assert_eq!(route_for_call(1).unwrap(), (Some(1.0), 0, 1));
        assert_eq!(route_for_call(47).unwrap(), (Some(1.0), 0, 47));
        assert_eq!(route_for_call(48).unwrap(), (None, 1, 0));
        assert_eq!(route_for_call(49).unwrap(), (Some(1.25), 1, 1));
        assert_eq!(route_for_call(96).unwrap(), (None, 2, 0));
        assert_eq!(route_for_call(97).unwrap(), (Some(1.5), 2, 1));
        assert_eq!(route_for_call(143).unwrap(), (Some(1.5), 2, 47));
        assert!(route_for_call(144).is_err());
    }
}
