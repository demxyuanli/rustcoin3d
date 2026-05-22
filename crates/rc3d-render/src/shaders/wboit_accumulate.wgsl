// WBOIT accumulation shader fragment output
// This is a fragment-only helper that wraps existing PBR output into
// weighted blended format:
//   RT0 (accum):     color.rgb * alpha * weight, alpha * weight
//   RT1 (revealage): alpha (as R8 — only R channel used)
//
// The weight function follows McGuire 2012:
//   weight = clamp(alpha / (1e-5 + depth³), 1e-3, 300.0)
//
// NOTE: This shader is NOT used as a standalone pipeline.
// Instead, the WBOIT logic is embedded in the PBR shader via
// `#define WBOIT_OUTPUT` which redirects the final fragment output.
// This file serves as the conceptual reference and provides the
// composite shader only.
