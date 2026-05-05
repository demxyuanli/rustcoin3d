#[derive(Clone, Debug, Default)]
pub struct FrameStats {
    pub visible_triangles: u64,
    pub visible_draw_calls: usize,
    pub culled_draw_calls: usize,
    /// Approximate GPU time per pass in microseconds: [shadow, solid, post, total]
    pub gpu_pass_times_us: Option<[f64; 4]>,
    pub diagnostics: Option<FrameDiagnostics>,
}

#[derive(Clone, Debug, Default)]
pub struct MemoryBudget {
    pub label: String,
    pub used_bytes: u64,
    pub budget_bytes: u64,
}

#[derive(Clone, Debug, Default)]
pub struct NodeTypeDrawStat {
    pub node_type: String,
    pub draw_calls: usize,
    pub triangles: u64,
}

#[derive(Clone, Debug, Default)]
pub struct BatchAnalysis {
    pub candidate_draw_calls: usize,
    pub submitted_draw_calls: usize,
    pub estimated_batch_count: usize,
    pub missed_reasons: Vec<(String, usize)>,
}

#[derive(Clone, Debug, Default)]
pub struct FrameDiagnostics {
    pub frame_index: u64,
    pub gpu_pass_timings_us: Vec<(String, f64)>,
    pub cpu_memory: Vec<MemoryBudget>,
    pub gpu_memory: Vec<MemoryBudget>,
    pub node_type_stats: Vec<NodeTypeDrawStat>,
    pub batch_analysis: BatchAnalysis,
    pub timestamp_supported: bool,
}

impl FrameDiagnostics {
    pub fn to_json_pretty(&self) -> String {
        fn esc(s: &str) -> String {
            s.replace('\\', "\\\\").replace('"', "\\\"")
        }
        let mut out = String::new();
        out.push_str("{\n");
        out.push_str(&format!("  \"frame_index\": {},\n", self.frame_index));
        out.push_str(&format!(
            "  \"timestamp_supported\": {},\n",
            if self.timestamp_supported {
                "true"
            } else {
                "false"
            }
        ));
        out.push_str("  \"gpu_pass_timings_us\": [\n");
        for (i, (name, us)) in self.gpu_pass_timings_us.iter().enumerate() {
            out.push_str(&format!(
                "    {{\"name\": \"{}\", \"us\": {:.3}}}{}\n",
                esc(name),
                us,
                if i + 1 == self.gpu_pass_timings_us.len() {
                    ""
                } else {
                    ","
                }
            ));
        }
        out.push_str("  ],\n");
        let write_budgets = |out: &mut String, key: &str, items: &[MemoryBudget]| {
            out.push_str(&format!("  \"{}\": [\n", key));
            for (i, m) in items.iter().enumerate() {
                out.push_str(&format!(
                    "    {{\"label\": \"{}\", \"used_bytes\": {}, \"budget_bytes\": {}}}{}\n",
                    esc(&m.label),
                    m.used_bytes,
                    m.budget_bytes,
                    if i + 1 == items.len() { "" } else { "," }
                ));
            }
            out.push_str("  ],\n");
        };
        write_budgets(&mut out, "cpu_memory", &self.cpu_memory);
        write_budgets(&mut out, "gpu_memory", &self.gpu_memory);
        out.push_str("  \"node_type_stats\": [\n");
        for (i, s) in self.node_type_stats.iter().enumerate() {
            out.push_str(&format!(
                "    {{\"node_type\": \"{}\", \"draw_calls\": {}, \"triangles\": {}}}{}\n",
                esc(&s.node_type),
                s.draw_calls,
                s.triangles,
                if i + 1 == self.node_type_stats.len() {
                    ""
                } else {
                    ","
                }
            ));
        }
        out.push_str("  ],\n");
        out.push_str("  \"batch_analysis\": {\n");
        out.push_str(&format!(
            "    \"candidate_draw_calls\": {},\n",
            self.batch_analysis.candidate_draw_calls
        ));
        out.push_str(&format!(
            "    \"submitted_draw_calls\": {},\n",
            self.batch_analysis.submitted_draw_calls
        ));
        out.push_str(&format!(
            "    \"estimated_batch_count\": {},\n",
            self.batch_analysis.estimated_batch_count
        ));
        out.push_str("    \"missed_reasons\": [\n");
        for (i, (reason, count)) in self.batch_analysis.missed_reasons.iter().enumerate() {
            out.push_str(&format!(
                "      {{\"reason\": \"{}\", \"count\": {}}}{}\n",
                esc(reason),
                count,
                if i + 1 == self.batch_analysis.missed_reasons.len() {
                    ""
                } else {
                    ","
                }
            ));
        }
        out.push_str("    ]\n");
        out.push_str("  }\n");
        out.push_str("}\n");
        out
    }
}
