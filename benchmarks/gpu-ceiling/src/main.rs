use cubbs_bbs::{BbsBackend, BbsEngine};
use cubbs_gpu_ceiling::{config::Config, runner, workload};

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let config = Config::parse()?;
    BbsBackend::GpuRequestedCpuFallback
        .ensure_hardware_available()
        .map_err(|error| error.to_string())?;
    let engine = BbsEngine::new(BbsBackend::GpuRequestedCpuFallback);
    let workload = workload::prepare(&engine, &config)?;
    let report = runner::run(config, &engine, workload)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?
    );
    Ok(())
}
