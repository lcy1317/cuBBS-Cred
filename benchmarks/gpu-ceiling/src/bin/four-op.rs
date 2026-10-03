use cubbs_bbs::{BbsBackend, BbsEngine};
use cubbs_gpu_ceiling::{four_config::FourOpConfig, four_runner, four_workload};

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let config = FourOpConfig::parse()?;
    let backend = BbsBackend::parse(&config.backend).map_err(|error| error.to_string())?;
    if config.backend == "gpu" {
        four_runner::apply_gpu_throughput_preset(&config);
        backend
            .ensure_hardware_available()
            .map_err(|error| error.to_string())?;
    }
    let engine = BbsEngine::new(backend);
    let templates = four_workload::prepare(&engine, &config)?;
    let report = four_runner::run(config, &engine, templates)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?
    );
    Ok(())
}
