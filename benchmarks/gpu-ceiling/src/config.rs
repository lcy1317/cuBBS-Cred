use serde::Serialize;
use std::env;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Config {
    pub message_counts: Vec<usize>,
    pub batch_size: usize,
    pub streams: usize,
    pub warmup_secs: f64,
    pub duration_secs: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            message_counts: vec![16, 64, 128],
            batch_size: 8_192,
            streams: 4,
            warmup_secs: 1.0,
            duration_secs: 10.0,
        }
    }
}

impl Config {
    pub fn parse() -> Result<Self, String> {
        Self::parse_from(env::args().skip(1))
    }

    fn parse_from(args: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut config = Self::default();
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            let value = |args: &mut dyn Iterator<Item = String>| {
                args.next()
                    .ok_or_else(|| format!("missing value after {arg}"))
            };
            match arg.as_str() {
                "--messages" => {
                    let raw = value(&mut args)?;
                    config.message_counts = raw
                        .split(',')
                        .map(str::trim)
                        .filter(|item| !item.is_empty())
                        .map(|item| {
                            item.parse::<usize>()
                                .map_err(|_| format!("invalid message count: {item}"))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                }
                "--batch" | "--batch-size" => {
                    config.batch_size = value(&mut args)?
                        .parse()
                        .map_err(|_| "invalid batch size".to_owned())?;
                }
                "--streams" => {
                    config.streams = value(&mut args)?
                        .parse()
                        .map_err(|_| "invalid stream count".to_owned())?;
                }
                "--warmup-secs" => {
                    config.warmup_secs = value(&mut args)?
                        .parse()
                        .map_err(|_| "invalid warmup duration".to_owned())?;
                }
                "--duration-secs" => {
                    config.duration_secs = value(&mut args)?
                        .parse()
                        .map_err(|_| "invalid measurement duration".to_owned())?;
                }
                "--help" | "-h" => return Err(Self::usage().to_owned()),
                other => return Err(format!("unknown argument: {other}\n\n{}", Self::usage())),
            }
        }
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), String> {
        if self.message_counts.is_empty() || self.message_counts.contains(&0) {
            return Err("--messages must contain positive counts".to_owned());
        }
        if self.batch_size < self.message_counts.len() {
            return Err("--batch must be at least the number of message shapes".to_owned());
        }
        if self.streams == 0 {
            return Err("--streams must be positive".to_owned());
        }
        if !self.warmup_secs.is_finite() || self.warmup_secs < 0.0 {
            return Err("--warmup-secs must be finite and non-negative".to_owned());
        }
        if !self.duration_secs.is_finite() || self.duration_secs <= 0.0 {
            return Err("--duration-secs must be finite and positive".to_owned());
        }
        Ok(())
    }

    pub fn usage() -> &'static str {
        "cubbs-gpu-ceiling [options]\n\
         \n\
         --messages 16,64,128   Common-points MSM term shapes\n\
         --batch 8192           Output rows per stream and native call\n\
         --streams 4            Independent prepared plans / submitters\n\
         --warmup-secs 1        Warm-up duration per stream\n\
         --duration-secs 10     Stable measurement duration"
    }
}

#[cfg(test)]
mod tests {
    use super::Config;

    #[test]
    fn parses_explicit_configuration() {
        let config = Config::parse_from(
            [
                "--messages",
                "16,64",
                "--batch",
                "1024",
                "--streams",
                "2",
                "--warmup-secs",
                "0.5",
                "--duration-secs",
                "3",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(config.message_counts, vec![16, 64]);
        assert_eq!(config.batch_size, 1024);
        assert_eq!(config.streams, 2);
        assert_eq!(config.warmup_secs, 0.5);
        assert_eq!(config.duration_secs, 3.0);
    }

    #[test]
    fn rejects_zero_streams() {
        let error =
            Config::parse_from(["--streams", "0"].into_iter().map(str::to_owned)).unwrap_err();
        assert!(error.contains("streams"));
    }
}
