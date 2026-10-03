use crate::config::Config;
use bls12_381_plus::Scalar;
use cubbs_bbs::{BbsEngine, GpuMsmBenchGroupInput, SignRequest};
use rand::thread_rng;
use zkryptium::utils::util::bbsplus_utils::ScalarExt;

pub struct PreparedWorkload {
    pub groups: Vec<GpuMsmBenchGroupInput>,
    pub scalar_matrix_bytes: usize,
    pub point_bytes: usize,
}

pub fn prepare(engine: &BbsEngine, config: &Config) -> Result<PreparedWorkload, String> {
    let keypair = engine
        .generate_keypair(&mut thread_rng())
        .map_err(|error| error.to_string())?;
    let row_counts = rows_per_shape(config.batch_size, config.message_counts.len());
    let mut groups = Vec::with_capacity(config.message_counts.len());
    let mut scalar_matrix_bytes = 0usize;
    let mut point_bytes = 0usize;

    for (shape_index, (&message_count, &output_count)) in config
        .message_counts
        .iter()
        .zip(row_counts.iter())
        .enumerate()
    {
        let request = SignRequest {
            secret_key: keypair.secret_key.clone(),
            public_key: keypair.public_key.clone(),
            header: deterministic_header(message_count),
            messages: deterministic_messages(message_count),
        };
        let template = engine
            .prepare_b_msm_terms_for_sign(&request)
            .map_err(|error| error.to_string())?;
        if template.scalar_be.len() != message_count + 2 {
            return Err(format!(
                "unexpected MSM template width for {message_count} messages: {}",
                template.scalar_be.len()
            ));
        }

        let mut scalar_rows_be = Vec::with_capacity(output_count * template.scalar_be.len() * 32);
        for row in 0..output_count {
            scalar_rows_be.extend_from_slice(&template.scalar_be[0]);
            scalar_rows_be.extend_from_slice(&template.scalar_be[1]);
            for message_index in 0..message_count {
                let value = deterministic_scalar(shape_index, row, message_index);
                scalar_rows_be.extend_from_slice(&value.to_bytes_be());
            }
        }
        scalar_matrix_bytes += scalar_rows_be.len();
        point_bytes += template
            .compressed_points
            .iter()
            .map(Vec::len)
            .sum::<usize>();
        groups.push(GpuMsmBenchGroupInput {
            compressed_points: template.compressed_points.as_ref().clone(),
            scalar_rows_be,
            output_count,
        });
    }

    Ok(PreparedWorkload {
        groups,
        scalar_matrix_bytes,
        point_bytes,
    })
}

fn rows_per_shape(batch_size: usize, shape_count: usize) -> Vec<usize> {
    let base = batch_size / shape_count;
    let remainder = batch_size % shape_count;
    (0..shape_count)
        .map(|index| base + usize::from(index < remainder))
        .collect()
}

fn deterministic_header(seed: usize) -> Vec<u8> {
    (0..64)
        .map(|index| (seed as u8).wrapping_mul(17).wrapping_add(index as u8))
        .collect()
}

fn deterministic_messages(count: usize) -> Vec<Vec<u8>> {
    (0..count)
        .map(|index| format!("gpu-ceiling-template-{count}-{index}").into_bytes())
        .collect()
}

fn deterministic_scalar(shape: usize, row: usize, message: usize) -> Scalar {
    let value = (shape as u64 + 1)
        .wrapping_mul(0x9e37_79b9)
        .wrapping_add((row as u64 + 1).wrapping_mul(0x85eb_ca6b))
        .wrapping_add((message as u64 + 1).wrapping_mul(0xc2b2_ae35));
    Scalar::from(value.max(1))
}

#[cfg(test)]
mod tests {
    use super::rows_per_shape;

    #[test]
    fn distributes_all_rows_without_empty_shapes() {
        let rows = rows_per_shape(8192, 3);
        assert_eq!(rows, vec![2731, 2731, 2730]);
        assert_eq!(rows.iter().sum::<usize>(), 8192);
    }
}
