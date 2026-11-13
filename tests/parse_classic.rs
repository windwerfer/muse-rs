//! Classic firmware decoders: hand-built packets whose expected values
//! follow the formulas documented in `muse_rs::parse`.

use muse_rs::parse::{
    decode_eeg_samples, decode_ppg_samples, decode_unsigned_12bit, decode_unsigned_24bit,
    parse_accelerometer, parse_gyroscope, parse_ppg_reading, parse_telemetry, ControlAccumulator,
};

// ── packing helpers ──────────────────────────────────────────────────────────

/// Pack 12-bit values in the classic big-endian layout:
/// `[AA BB CC] → sample0 = (AA << 4) | (BB >> 4)`,
/// `sample1 = ((BB & 0x0F) << 8) | CC`.
fn pack_12bit_be(values: &[u16]) -> Vec<u8> {
    let mut out = Vec::new();
    for chunk in values.chunks(2) {
        let v0 = chunk[0];
        if chunk.len() == 1 {
            out.push((v0 >> 4) as u8);
            out.push(((v0 & 0x0F) << 4) as u8);
            break;
        }
        let v1 = chunk[1];
        out.push((v0 >> 4) as u8);
        out.push((((v0 & 0x0F) << 4) | ((v1 >> 8) & 0x0F)) as u8);
        out.push((v1 & 0xFF) as u8);
    }
    out
}

fn pack_24bit_be(values: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 3);
    for &v in values {
        out.push(((v >> 16) & 0xFF) as u8);
        out.push(((v >> 8) & 0xFF) as u8);
        out.push((v & 0xFF) as u8);
    }
    out
}

fn classic_imu(seq: u16, samples: [[i16; 3]; 3]) -> Vec<u8> {
    let mut data = vec![0u8; 20];
    data[0..2].copy_from_slice(&seq.to_be_bytes());
    for (i, off) in [2usize, 8, 14].into_iter().enumerate() {
        data[off..off + 2].copy_from_slice(&samples[i][0].to_be_bytes());
        data[off + 2..off + 4].copy_from_slice(&samples[i][1].to_be_bytes());
        data[off + 4..off + 6].copy_from_slice(&samples[i][2].to_be_bytes());
    }
    data
}

fn assert_f64_close(got: f64, expected: f64) {
    let tol = 1e-9_f64.max(expected.abs() * 1e-12);
    assert!(
        (got - expected).abs() <= tol,
        "f64 {got} != {expected} (tol {tol})"
    );
}

fn assert_f32_close(got: f32, expected: f32) {
    let tol = 1e-6_f32.max(expected.abs() * 1e-6);
    assert!(
        (got - expected).abs() <= tol,
        "f32 {got} != {expected} (tol {tol})"
    );
}

// ── 12-bit EEG packing ───────────────────────────────────────────────────────

#[test]
fn unsigned_12bit_decodes_documented_layout() {
    // 0xABC, 0xDEF → AA=0xAB, BB=0xCD, CC=0xEF
    assert_eq!(
        decode_unsigned_12bit(&[0xAB, 0xCD, 0xEF]),
        vec![0xABC, 0xDEF]
    );
}

#[test]
fn unsigned_12bit_roundtrips_minmax_and_midscale() {
    let values = [0x000, 0x800, 0xFFF, 0x001];
    assert_eq!(decode_unsigned_12bit(&pack_12bit_be(&values)), values);
}

#[test]
fn unsigned_12bit_ignores_trailing_incomplete_group_and_empty() {
    assert!(decode_unsigned_12bit(&[]).is_empty());
    // A single leftover byte cannot form a sample.
    assert!(decode_unsigned_12bit(&[0xAB]).is_empty());
    // Two bytes produce only the first sample of the group.
    assert_eq!(decode_unsigned_12bit(&pack_12bit_be(&[0xABC])), vec![0xABC]);
    // A complete group plus a stray trailing byte: the stray is ignored.
    let mut data = pack_12bit_be(&[0x111, 0x222]);
    data.push(0xFF);
    assert_eq!(decode_unsigned_12bit(&data), vec![0x111, 0x222]);
}

#[test]
fn eeg_samples_use_midscale_2048_and_documented_uv_scale() {
    // µV = (raw − 2048) × 0.48828125, and 0.48828125 = 1000 / 2048.
    let raw = [0x000u16, 0x800, 0xFFF];
    let got = decode_eeg_samples(&pack_12bit_be(&raw));
    let expected: Vec<f64> = raw
        .iter()
        .map(|&n| 0.48828125 * (n as f64 - 0x800 as f64))
        .collect();
    assert_eq!(got.len(), expected.len());
    for (g, e) in got.iter().zip(expected.iter()) {
        assert_f64_close(*g, *e);
    }
    assert_f64_close(got[0], -1000.0);
    assert_f64_close(got[1], 0.0);
}

#[test]
fn eeg_samples_decodes_a_full_classic_packet_of_twelve() {
    let raw: Vec<u16> = (0..12).map(|i| 0x800 + i * 8).collect();
    let got = decode_eeg_samples(&pack_12bit_be(&raw));
    assert_eq!(got.len(), 12);
    for (i, sample) in got.iter().enumerate() {
        assert_f64_close(*sample, 0.48828125 * (i as f64 * 8.0));
    }
}

// ── 24-bit PPG packing ───────────────────────────────────────────────────────

#[test]
fn unsigned_24bit_decodes_big_endian_groups() {
    let values = [0x000000u32, 0x010203, 0xAABBCC, 0xFFFFFF];
    assert_eq!(decode_unsigned_24bit(&pack_24bit_be(&values)), values);
    assert_eq!(decode_ppg_samples(&pack_24bit_be(&values)), values);
}

#[test]
fn unsigned_24bit_ignores_partial_trailing_bytes_and_empty() {
    assert!(decode_unsigned_24bit(&[]).is_empty());
    assert!(decode_unsigned_24bit(&[0x01]).is_empty());
    assert!(decode_unsigned_24bit(&[0x01, 0x02]).is_empty());
    let mut data = pack_24bit_be(&[0x010203]);
    data.extend_from_slice(&[0xAA, 0xBB]);
    assert_eq!(decode_unsigned_24bit(&data), vec![0x010203]);
}

#[test]
fn parse_ppg_reading_takes_index_channel_timestamp_and_samples() {
    let mut data = 0x1234u16.to_be_bytes().to_vec();
    data.extend(pack_24bit_be(&[
        0x010203, 0x040506, 0x070809, 0x0A0B0C, 0x0D0E0F, 0x101112,
    ]));
    let reading = parse_ppg_reading(&data, 2, 42.5).expect("ppg reading");
    assert_eq!(reading.index, 0x1234);
    assert_eq!(reading.ppg_channel, 2);
    assert_f64_close(reading.timestamp, 42.5);
    assert_eq!(
        reading.samples,
        vec![0x010203, 0x040506, 0x070809, 0x0A0B0C, 0x0D0E0F, 0x101112]
    );
}

#[test]
fn parse_ppg_reading_rejects_short_payloads_but_accepts_index_only() {
    assert!(parse_ppg_reading(&[], 0, 0.0).is_none());
    assert!(parse_ppg_reading(&[0x00], 0, 0.0).is_none());
    let reading = parse_ppg_reading(&[0x00, 0x07], 1, 1.0).expect("index only");
    assert_eq!(reading.index, 7);
    assert!(reading.samples.is_empty());
}

// ── Telemetry ────────────────────────────────────────────────────────────────

#[test]
fn parse_telemetry_applies_classic_conversions() {
    // sequence 0x0010, battery 25600 / 512 = 50 %, fuel 1000 × 2.2 = 2200 mV,
    // bytes 6–7 unused, temperature 0x00AB.
    let mut data = vec![0u8; 10];
    data[0..2].copy_from_slice(&0x0010u16.to_be_bytes());
    data[2..4].copy_from_slice(&25600u16.to_be_bytes());
    data[4..6].copy_from_slice(&1000u16.to_be_bytes());
    data[6..8].copy_from_slice(&0xFFFFu16.to_be_bytes());
    data[8..10].copy_from_slice(&0x00ABu16.to_be_bytes());
    let t = parse_telemetry(&data).expect("telemetry");
    assert_eq!(t.sequence_id, 0x0010);
    assert_f32_close(t.battery_level, 50.0);
    assert_f32_close(t.fuel_gauge_voltage, 2200.0);
    assert_eq!(t.temperature, 0x00AB);
}

#[test]
fn parse_telemetry_full_scale_and_zero() {
    let mut zero = vec![0u8; 10];
    let t = parse_telemetry(&zero).expect("zero telemetry");
    assert_eq!(t.sequence_id, 0);
    assert_f32_close(t.battery_level, 0.0);
    assert_f32_close(t.fuel_gauge_voltage, 0.0);
    assert_eq!(t.temperature, 0);

    zero[2..4].copy_from_slice(&51200u16.to_be_bytes());
    let full = parse_telemetry(&zero).expect("full battery");
    assert_f32_close(full.battery_level, 100.0);
}

#[test]
fn parse_telemetry_rejects_short_payloads_and_ignores_trailing() {
    assert!(parse_telemetry(&[]).is_none());
    assert!(parse_telemetry(&[0u8; 9]).is_none());
    let mut data = vec![0u8; 12];
    data[2..4].copy_from_slice(&512u16.to_be_bytes()); // 512 / 512 = 1 %
    let t = parse_telemetry(&data).expect("longer than 10");
    assert_f32_close(t.battery_level, 1.0);
}

// ── IMU ──────────────────────────────────────────────────────────────────────

const ACCEL_SCALE: f32 = 0.0000610352;
const GYRO_SCALE: f32 = 0.0074768;

#[test]
fn parse_accelerometer_scales_all_three_samples() {
    let samples = [[1i16, -2, 3], [i16::MIN, 0, i16::MAX], [100, -100, 0]];
    let imu = parse_accelerometer(&classic_imu(0xBEEF, samples)).expect("accel");
    assert_eq!(imu.sequence_id, 0xBEEF);
    for (got, raw) in imu.samples.iter().zip(samples.iter()) {
        assert_f32_close(got.x, ACCEL_SCALE * raw[0] as f32);
        assert_f32_close(got.y, ACCEL_SCALE * raw[1] as f32);
        assert_f32_close(got.z, ACCEL_SCALE * raw[2] as f32);
    }
}

#[test]
fn parse_gyroscope_uses_positive_classic_scale() {
    let samples = [[10i16, 20, -30], [0, 1, -1], [i16::MAX, i16::MIN, 42]];
    let imu = parse_gyroscope(&classic_imu(7, samples)).expect("gyro");
    assert_eq!(imu.sequence_id, 7);
    for (got, raw) in imu.samples.iter().zip(samples.iter()) {
        assert_f32_close(got.x, GYRO_SCALE * raw[0] as f32);
        assert_f32_close(got.y, GYRO_SCALE * raw[1] as f32);
        assert_f32_close(got.z, GYRO_SCALE * raw[2] as f32);
    }
}

#[test]
fn parse_imu_rejects_payloads_shorter_than_20_bytes() {
    assert!(parse_accelerometer(&[]).is_none());
    assert!(parse_accelerometer(&[0u8; 19]).is_none());
    assert!(parse_gyroscope(&[0u8; 19]).is_none());
    assert!(parse_accelerometer(&classic_imu(1, [[0; 3]; 3])).is_some());
}

// ── ControlAccumulator ───────────────────────────────────────────────────────

#[test]
fn control_accumulator_assembles_split_json() {
    let mut acc = ControlAccumulator::new();
    assert!(acc.push(r#"{"fw":"3.4."#).is_none());
    let json = acc.push(r#"5"}"#).expect("closed json");
    assert_eq!(json, r#"{"fw":"3.4.5"}"#);
    // Buffer is cleared; a new object can start immediately.
    assert!(acc.push(r#"{"hn":"#).is_none());
    assert_eq!(acc.push(r#""Muse"}"#).unwrap(), r#"{"hn":"Muse"}"#);
}

#[test]
fn control_accumulator_tracks_nested_braces_and_default_matches_new() {
    let mut acc = ControlAccumulator::default();
    assert!(acc.push(r#"{"a":{"b":1}"#).is_none());
    assert_eq!(acc.push("}").unwrap(), r#"{"a":{"b":1}}"#);
}

#[test]
fn control_accumulator_discards_garbage_before_and_stray_braces() {
    let mut acc = ControlAccumulator::new();
    assert!(acc.push("noise}also").is_none());
    assert_eq!(acc.push(r#"{"ok":true}"#).unwrap(), r#"{"ok":true}"#);
}

#[test]
fn control_accumulator_keeps_unicode_payloads() {
    let mut acc = ControlAccumulator::new();
    assert_eq!(
        acc.push(r#"{"hn":"Müse-ÄB12"}"#).unwrap(),
        r#"{"hn":"Müse-ÄB12"}"#
    );
}

#[test]
fn control_accumulator_returns_first_object_when_two_share_a_fragment() {
    // Current behaviour: once depth returns to 0, remaining characters in the
    // same push() call are not buffered. A later object in that fragment is
    // therefore dropped. The headset is not known to emit two objects in one
    // notification, but the edge case is recorded here.
    let mut acc = ControlAccumulator::new();
    assert_eq!(acc.push(r#"{"a":1}{"b":2}"#).unwrap(), r#"{"a":1}"#);
    assert!(acc.push("").is_none());
}
