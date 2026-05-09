use std::sync::mpsc;
use std::time::Instant;

/// GPU timestamp query wrapper — captures wgpu timestamp at begin/end of each pass.
pub struct GpuTimer {
    set: wgpu::QuerySet,
    resolve_buf: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    capacity: u32,
    next_slot: u32,
    /// Raw timestamp ticks read back from previous frame (in GPU clock ticks, not ns).
    /// Multiply by `timestamp_period_ns` to get nanoseconds.
    pub last_timestamps: Vec<u64>,
    /// Labels for each timestamp pair, in begin() order. Index = slot index.
    pub labels: Vec<&'static str>,
    /// GPU timestamp period in nanoseconds (from `queue.get_timestamp_period()`).
    pub timestamp_period_ns: f32,
}

impl GpuTimer {
    pub fn new(device: &wgpu::Device, capacity: u32, timestamp_period_ns: f32) -> Self {
        let set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Frame Timer Queries"),
            ty: wgpu::QueryType::Timestamp,
            count: capacity * 2,
        });
        let size = (capacity * 2 * 8) as u64;
        let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Timer Resolve"),
            size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Timer Staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            set,
            resolve_buf,
            staging_buf,
            capacity,
            next_slot: 0,
            last_timestamps: Vec::new(),
            labels: Vec::new(),
            timestamp_period_ns,
        }
    }

    /// Begin a timestamp span. Returns `Some(slot)` on success or `None` when at capacity.
    pub fn begin(&mut self, encoder: &mut wgpu::CommandEncoder, label: &'static str) -> Option<u32> {
        let idx = self.next_slot;
        if idx < self.capacity {
            encoder.write_timestamp(&self.set, idx * 2);
            if self.labels.len() <= idx as usize {
                self.labels.push(label);
            } else {
                self.labels[idx as usize] = label;
            }
            self.next_slot += 1;
            Some(idx)
        } else {
            None
        }
    }

    /// End a timestamp span. Pass the `Option<u32>` returned by `begin()`.
    pub fn end(&self, encoder: &mut wgpu::CommandEncoder, idx: Option<u32>) {
        if let Some(idx) = idx {
            if idx < self.capacity {
                encoder.write_timestamp(&self.set, idx * 2 + 1);
            }
        }
    }

    pub fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if self.next_slot > 0 {
            encoder.resolve_query_set(
                &self.set,
                0..(self.next_slot * 2),
                &self.resolve_buf,
                0,
            );
            encoder.copy_buffer_to_buffer(
                &self.resolve_buf,
                0,
                &self.staging_buf,
                0,
                (self.next_slot * 2 * 8) as u64,
            );
        }
        self.labels.truncate(self.next_slot as usize);
        self.next_slot = 0;
    }

    pub fn collect(&mut self, device: &wgpu::Device) {
        let buf_slice = self.staging_buf.slice(..);
        let (tx, rx) = mpsc::channel();
        buf_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        device.poll(wgpu::Maintain::Wait);
        if let Ok(Ok(())) = rx.recv() {
            let view = buf_slice.get_mapped_range();
            let timestamps: &[u64] = bytemuck::cast_slice(&view);
            let active_count = (self.labels.len() * 2).min(timestamps.len());
            self.last_timestamps = timestamps[..active_count].to_vec();
            drop(view);
        }
        self.staging_buf.unmap();
    }
}

/// CPU-side timing span — collects per-frame durations.
#[derive(Default)]
pub struct CpuSpanCollector {
    spans: Vec<(&'static str, f64)>, // (name, duration_ms)
    frame_start: Option<Instant>,
}

impl CpuSpanCollector {
    pub fn begin_frame(&mut self) {
        self.frame_start = Some(Instant::now());
        self.spans.clear();
    }

    pub fn measure<T>(&mut self, label: &'static str, f: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let result = f();
        let dur_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.spans.push((label, dur_ms));
        result
    }

    pub fn record(&mut self, label: &'static str, duration_ms: f64) {
        self.spans.push((label, duration_ms));
    }

    pub fn spans(&self) -> &[(&'static str, f64)] {
        &self.spans
    }

    pub fn total_ms(&self) -> f64 {
        self.frame_start
            .map(|s| s.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_span_measures_duration() {
        let mut collector = CpuSpanCollector::default();
        collector.begin_frame();
        let result = collector.measure("test_op", || 42);
        assert_eq!(result, 42);
        assert_eq!(collector.spans().len(), 1);
        assert_eq!(collector.spans()[0].0, "test_op");
        assert!(collector.total_ms() >= 0.0);
    }

    #[test]
    fn cpu_span_multiple_measures() {
        let mut collector = CpuSpanCollector::default();
        collector.begin_frame();
        collector.measure("op1", || std::thread::sleep(std::time::Duration::from_millis(1)));
        collector.measure("op2", || std::thread::sleep(std::time::Duration::from_millis(1)));
        assert_eq!(collector.spans().len(), 2);
    }
}
