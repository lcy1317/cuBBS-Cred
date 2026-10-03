use crate::config::Config;
use serde::Serialize;
use std::{fs, process::Command};

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BackendReport {
    pub requested: &'static str,
    pub active: &'static str,
    pub hardware_acceleration: bool,
    pub fallback: bool,
    pub gpu_abi_version: u32,
    pub detail: &'static str,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PreparationReport {
    pub policy: &'static str,
    pub scalar_matrix_bytes_per_stream: usize,
    pub common_point_bytes_per_stream: usize,
    pub rust_timed_loop_allocations: &'static str,
    pub correctness_validation: &'static str,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MetricReport {
    pub elapsed_ms: f64,
    pub native_calls: u64,
    pub outputs: u64,
    pub outputs_per_sec: f64,
    pub native_calls_per_sec: f64,
    pub process_user_seconds: f64,
    pub process_system_seconds: f64,
    pub equivalent_cpu_cores: f64,
    pub cpu_seconds_per_million_outputs: f64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CeilingReport {
    pub tool: &'static str,
    pub mode: &'static str,
    pub scope: &'static str,
    pub config: Config,
    pub backend: BackendReport,
    pub preparation: PreparationReport,
    pub metric: MetricReport,
    pub validation_passed: bool,
    pub gpu: Option<String>,
    pub cpu: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct ProcessCpuTime {
    pub user_seconds: f64,
    pub system_seconds: f64,
}

pub fn process_cpu_time() -> Result<ProcessCpuTime, String> {
    let stat = fs::read_to_string("/proc/self/stat")
        .map_err(|error| format!("read /proc/self/stat: {error}"))?;
    let close = stat
        .rfind(')')
        .ok_or_else(|| "invalid /proc/self/stat comm field".to_owned())?;
    let fields = stat[close + 1..].split_whitespace().collect::<Vec<_>>();
    let user_ticks = fields
        .get(11)
        .ok_or_else(|| "missing process utime".to_owned())?
        .parse::<f64>()
        .map_err(|_| "invalid process utime".to_owned())?;
    let system_ticks = fields
        .get(12)
        .ok_or_else(|| "missing process stime".to_owned())?
        .parse::<f64>()
        .map_err(|_| "invalid process stime".to_owned())?;
    let ticks_per_second = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
    if ticks_per_second <= 0 {
        return Err("sysconf(_SC_CLK_TCK) failed".to_owned());
    }
    Ok(ProcessCpuTime {
        user_seconds: user_ticks / ticks_per_second as f64,
        system_seconds: system_ticks / ticks_per_second as f64,
    })
}

pub fn gpu_identity() -> Option<String> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,uuid,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        ])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

pub fn cpu_identity() -> Option<String> {
    let cpuinfo = fs::read_to_string("/proc/cpuinfo").ok()?;
    cpuinfo
        .lines()
        .find_map(|line| {
            line.strip_prefix("model name")
                .and_then(|line| line.split_once(':'))
        })
        .map(|(_, value)| value.trim().to_owned())
}
