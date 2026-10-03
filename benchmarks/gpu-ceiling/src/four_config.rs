use serde::Serialize;
use std::env;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FourOpConfig {
    pub backend: String,
    pub api_mode: String,
    pub operation: String,
    pub message_counts: Vec<usize>,
    pub unique_issuer_per_item: bool,
    pub disclosed_ratio: f64,
    pub batch_size: usize,
    pub sign_batches: usize,
    pub verify_batches: usize,
    pub proof_gen_batches: usize,
    pub proof_verify_batches: usize,
    pub sign_streams: usize,
    pub verify_streams: usize,
    pub proof_gen_streams: usize,
    pub proof_verify_streams: usize,
}

impl Default for FourOpConfig {
    fn default() -> Self {
        Self {
            backend: "gpu".to_owned(),
            api_mode: "batch".to_owned(),
            operation: "all".to_owned(),
            message_counts: vec![16],
            unique_issuer_per_item: false,
            disclosed_ratio: 0.5,
            batch_size: 4_096,
            sign_batches: 32,
            verify_batches: 128,
            proof_gen_batches: 16,
            proof_verify_batches: 32,
            sign_streams: 4,
            verify_streams: 20,
            proof_gen_streams: 6,
            proof_verify_streams: 14,
        }
    }
}

impl FourOpConfig {
    pub fn parse() -> Result<Self, String> {
        let mut config = Self::default();
        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            let raw = |args: &mut dyn Iterator<Item = String>| {
                args.next()
                    .ok_or_else(|| format!("missing value after {arg}"))
            };
            match arg.as_str() {
                "--backend" => config.backend = raw(&mut args)?,
                "--api" => config.api_mode = raw(&mut args)?,
                "--operation" => config.operation = raw(&mut args)?,
                "--messages" => {
                    config.message_counts = raw(&mut args)?
                        .split(',')
                        .map(str::trim)
                        .filter(|item| !item.is_empty())
                        .map(|item| {
                            item.parse()
                                .map_err(|_| format!("invalid message count: {item}"))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                }
                "--unique-issuer-per-item" => config.unique_issuer_per_item = true,
                "--same-issuer" => config.unique_issuer_per_item = false,
                "--disclosed-ratio" => {
                    config.disclosed_ratio = raw(&mut args)?
                        .parse()
                        .map_err(|_| format!("invalid value for {arg}"))?;
                }
                "--batch" | "--batch-size" => {
                    config.batch_size = parse_usize(&raw(&mut args)?, &arg)?
                }
                "--sign-batches" => config.sign_batches = parse_usize(&raw(&mut args)?, &arg)?,
                "--verify-batches" => config.verify_batches = parse_usize(&raw(&mut args)?, &arg)?,
                "--proof-gen-batches" => {
                    config.proof_gen_batches = parse_usize(&raw(&mut args)?, &arg)?
                }
                "--proof-verify-batches" => {
                    config.proof_verify_batches = parse_usize(&raw(&mut args)?, &arg)?
                }
                "--sign-streams" => config.sign_streams = parse_usize(&raw(&mut args)?, &arg)?,
                "--verify-streams" => config.verify_streams = parse_usize(&raw(&mut args)?, &arg)?,
                "--proof-gen-streams" => {
                    config.proof_gen_streams = parse_usize(&raw(&mut args)?, &arg)?
                }
                "--proof-verify-streams" => {
                    config.proof_verify_streams = parse_usize(&raw(&mut args)?, &arg)?
                }
                "--help" | "-h" => return Err(Self::usage().to_owned()),
                other => return Err(format!("unknown argument: {other}\n\n{}", Self::usage())),
            }
        }
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), String> {
        if !matches!(self.backend.as_str(), "cpu" | "gpu") {
            return Err(format!("invalid --backend: {}", self.backend));
        }
        if !matches!(self.api_mode.as_str(), "single" | "batch") {
            return Err(format!("invalid --api: {}", self.api_mode));
        }
        if !matches!(
            self.operation.as_str(),
            "all" | "sign" | "verify" | "verify-signature" | "proof-gen" | "proof-verify"
        ) {
            return Err(format!("invalid --operation: {}", self.operation));
        }
        if self.message_counts.is_empty() || self.message_counts.contains(&0) {
            return Err("--messages must contain positive counts".to_owned());
        }
        if !self.disclosed_ratio.is_finite()
            || self.disclosed_ratio < 0.0
            || self.disclosed_ratio > 1.0
        {
            return Err("--disclosed-ratio must be in [0, 1]".to_owned());
        }
        if self.batch_size == 0 {
            return Err("--batch must be positive".to_owned());
        }
        for (label, batches, streams) in [
            ("sign", self.sign_batches, self.sign_streams),
            ("verify", self.verify_batches, self.verify_streams),
            ("proof-gen", self.proof_gen_batches, self.proof_gen_streams),
            (
                "proof-verify",
                self.proof_verify_batches,
                self.proof_verify_streams,
            ),
        ] {
            if batches == 0 || streams == 0 || batches < streams {
                return Err(format!(
                    "{label} requires positive batches/streams and batches >= streams"
                ));
            }
        }
        Ok(())
    }

    pub fn usage() -> &'static str {
        "cubbs-four-op-preloaded | cubbs-four-op-reference [options]\n\
         \n\
         cubbs-four-op-reference requires --backend cpu --api single\n\
         \n\
         --backend gpu              gpu|cpu\n\
         --api batch                single|batch\n\
         --operation all             all|sign|verify-signature|proof-gen|proof-verify\n\
         --messages 16               Message shapes (mixed lists supported)\n\
         --same-issuer               Reuse one issuer key across the batch (default)\n\
         --unique-issuer-per-item    Generate a distinct issuer key for every item\n\
         --disclosed-ratio 0.5       Fraction disclosed by proof operations\n\
         --batch 4096                Items per preloaded batch\n\
         --sign-batches 32           One-shot sign batches\n\
         --verify-batches 128        One-shot signature-verify batches\n\
         --proof-gen-batches 16      One-shot proof-gen batches\n\
         --proof-verify-batches 32   One-shot proof-verify batches\n\
         --sign-streams 4            Sign consumers\n\
         --verify-streams 20         Verify consumers\n\
         --proof-gen-streams 6       Proof-gen consumers\n\
         --proof-verify-streams 14   Proof-verify consumers"
    }

    pub fn runs(&self, operation: &str) -> bool {
        self.operation == "all"
            || self.operation == operation
            || (operation == "verify-signature" && self.operation == "verify")
    }
}

fn parse_usize(value: &str, label: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|_| format!("invalid value for {label}: {value}"))
}
