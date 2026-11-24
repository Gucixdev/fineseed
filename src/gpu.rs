//! The chunked Huffman part (`huffman::encode_chunked`) decoded on the GPU, one chunk per invocation. Optional
//! (`--features gpu`, wgpu over Vulkan). The coded part is uploaded as it is, the table as 32-bit entries, and the
//! plane comes back in one read. This is a measurement tool for now: the archive code does not call it.

use crate::huffman::{CHUNK, chunk_ranges, decoding_table, read_lengths};
use wgpu::util::DeviceExt;

pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    pub name: String,
    max_bytes: usize,
}

fn bytes_of(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

impl Gpu {
    /// The first adapter wgpu finds, or `None` when there is no usable GPU.
    pub fn new() -> Option<Gpu> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .ok()?;
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: limits.clone(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .ok()?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("decode.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let max_bytes = (limits.max_storage_buffer_binding_size as usize).min(limits.max_buffer_size as usize);
        Some(Gpu {
            device,
            queue,
            pipeline,
            name: adapter.get_info().name,
            max_bytes,
        })
    }

    /// Exactly `n` bytes, or an error on a damaged part or a part too big for one buffer of this GPU.
    pub fn decode_chunked(&self, coded: &[u8], n: usize) -> Result<Vec<u8>, String> {
        self.decode_reps(coded, n, 1)
    }

    /// `decode_chunked` with the kernel run `reps` times in one submission (the output is the same each time), so
    /// that the difference between two timings is the kernel alone, without upload and download.
    pub fn decode_reps(&self, coded: &[u8], n: usize, reps: u32) -> Result<Vec<u8>, String> {
        let (table, top) = decoding_table(&read_lengths(coded)?)?;
        let ranges = chunk_ranges(coded, n)?;
        let out_len = ranges.len() * CHUNK;
        if coded.len() + 16 > self.max_bytes || out_len > self.max_bytes {
            return Err("the part does not fit in one buffer of this GPU".into());
        }
        let mut stream = coded.to_vec();
        stream.resize(coded.len().div_ceil(4) * 4 + 8, 0);
        let meta: Vec<u32> = ranges.iter().flat_map(|&(a, b)| [a as u32, b as u32]).collect();
        let table: Vec<u32> = table[..1 << top].iter().map(|&e| u32::from(e)).collect();
        let params = [top, ranges.len() as u32, n as u32, CHUNK as u32];

        let dev = &self.device;
        let init = |data: &[u8], usage| {
            dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: data,
                usage,
            })
        };
        let storage = wgpu::BufferUsages::STORAGE;
        let b_stream = init(&stream, storage);
        let b_meta = init(&bytes_of(&meta), storage);
        let b_table = init(&bytes_of(&table), storage);
        let b_params = init(&bytes_of(&params), wgpu::BufferUsages::UNIFORM);
        let b_err = init(&[0u8; 4], storage | wgpu::BufferUsages::COPY_SRC);
        let buffer = |size: usize, usage| {
            dev.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: size as u64,
                usage,
                mapped_at_creation: false,
            })
        };
        let b_out = buffer(out_len, storage | wgpu::BufferUsages::COPY_SRC);
        let down = buffer(out_len, wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ);
        let down_err = buffer(4, wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ);

        let layout = self.pipeline.get_bind_group_layout(0);
        let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: b_stream.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_meta.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_table.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: b_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: b_err.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: b_params.as_entire_binding() },
            ],
        });
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            for _ in 0..reps.max(1) {
                pass.dispatch_workgroups(ranges.len().div_ceil(64) as u32, 1, 1);
            }
        }
        enc.copy_buffer_to_buffer(&b_out, 0, &down, 0, out_len as u64);
        enc.copy_buffer_to_buffer(&b_err, 0, &down_err, 0, 4);
        self.queue.submit([enc.finish()]);

        down.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        down_err.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        dev.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("gpu: {e}"))?;
        let flag = {
            let view = down_err.slice(..).get_mapped_range().map_err(|e| format!("gpu: {e}"))?;
            u32::from_le_bytes(view[..4].try_into().unwrap_or([0; 4]))
        };
        if flag != 0 {
            return Err("huffman stream damaged or cut short".into());
        }
        let view = down.slice(..).get_mapped_range().map_err(|e| format!("gpu: {e}"))?;
        Ok(view[..n].to_vec())
    }
}
