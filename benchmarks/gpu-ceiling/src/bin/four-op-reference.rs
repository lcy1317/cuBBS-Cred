use cubbs_bbs::reference::{Zkryptium061Reference, ZkryptiumReference};
use cubbs_gpu_ceiling::{four_config::FourOpConfig, four_runner, four_workload};

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let config = FourOpConfig::parse()?;
    if config.backend != "cpu" || config.api_mode != "single" {
        return Err("cubbs-four-op-reference requires --backend cpu --api single".to_owned());
    }
    let reference = ZkryptiumReference::V0_6_1(Zkryptium061Reference::new());
    let templates = four_workload::prepare_reference(&reference, &config)?;
    let report = four_runner::run_reference(config, &reference, templates)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?
    );
    Ok(())
}
