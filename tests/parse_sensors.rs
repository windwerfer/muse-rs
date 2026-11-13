//! Athena EEG, IMU, DRL/REF, battery, and decoder edge cases not covered by
//! `tests/athena.rs` (which focuses on the optics path and packet framing).

use muse_rs::parse::{athena_payload_len, parse_athena_notification, AthenaDecoder};
use muse_rs::protocol::ATHENA_EEG_SCALE;
use muse_rs::types::{ControlResponse, MuseEvent, Sensor};

fn pack_uint_le(values: &[u32], bit_width: usize) -> Vec<u8> {
    let n_bits = values.len() * bit_width;
    let mut bytes = vec![0u8; n_bits.div_ceil(8)];
    for (i, &val) in values.iter().enumerate() {
        for bit in 0..bit_width {
            if (val >> bit) & 1 != 0 {
                let total = i * bit_width + bit;
                bytes[total / 8] |= 1 << (total % 8);
            }
        }
    }
    bytes
}

fn packet(tag: u8, counter: u8, payload: &[u8]) -> Vec<u8> {
    let pkt_len = 14 + payload.len();
    assert!(pkt_len <= 255, "packet too long for a single length byte");
    let mut p = vec![0u8; pkt_len];
    p[0] = pkt_len as u8;
    p[9] = tag;
    p[10] = counter;
    p[14..].copy_from_slice(payload);
    p
}

fn eeg_events(events: &[MuseEvent]) -> Vec<(u16, usize, Vec<f64>)> {
    events
        .iter()
        .filter_map(|e| match e {
            MuseEvent::Eeg(r) => Some((r.index, r.electrode, r.samples.clone())),
            _ => None,
        })
        .collect()
}

fn telemetry(events: &[MuseEvent]) -> Vec<(u16, f32)> {
    events
        .iter()
        .filter_map(|e| match e {
            MuseEvent::Telemetry(t) => Some((t.sequence_id, t.battery_level)),
            _ => None,
        })
        .collect()
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

// ── payload-length table ─────────────────────────────────────────────────────

#[test]
fn athena_payload_len_matches_known_tags() {
    assert_eq!(athena_payload_len(0x11), Some(28));
    assert_eq!(athena_payload_len(0x12), Some(28));
    assert_eq!(athena_payload_len(0x34), Some(30));
    assert_eq!(athena_payload_len(0x35), Some(40));
    assert_eq!(athena_payload_len(0x36), Some(40));
    assert_eq!(athena_payload_len(0x47), Some(36));
    assert_eq!(athena_payload_len(0x53), Some(24));
    assert_eq!(athena_payload_len(0x88), None);
    assert_eq!(athena_payload_len(0x98), Some(20));
    assert_eq!(athena_payload_len(0x00), None);
    assert_eq!(athena_payload_len(0xFF), None);
}

// ── EEG ──────────────────────────────────────────────────────────────────────

fn eeg_values(n_ch: usize, n_samp: usize) -> Vec<u32> {
    (0..n_samp)
        .flat_map(|s| (0..n_ch).map(move |ch| (8192 + s * n_ch + ch) as u32))
        .collect()
}

#[test]
fn eeg_4ch_decodes_sample_major_14bit_values() {
    let ev = parse_athena_notification(&packet(0x11, 9, &pack_uint_le(&eeg_values(4, 4), 14)));
    let got = eeg_events(&ev);
    assert_eq!(got.len(), 4);
    for (ch, (index, electrode, samples)) in got.iter().enumerate() {
        assert_eq!(*index, 9);
        assert_eq!(*electrode, ch);
        assert_eq!(samples.len(), 4);
        for (s, sample) in samples.iter().enumerate() {
            assert_f64_close(*sample, (s * 4 + ch) as f64 * ATHENA_EEG_SCALE);
        }
    }
}

#[test]
fn eeg_8ch_decodes_two_samples_and_midscale_is_zero() {
    let ev = parse_athena_notification(&packet(0x12, 3, &pack_uint_le(&eeg_values(8, 2), 14)));
    let got = eeg_events(&ev);
    assert_eq!(got.len(), 8);
    assert_f64_close(got[0].2[0], 0.0);
    for (ch, (index, electrode, samples)) in got.iter().enumerate() {
        assert_eq!(*index, 3);
        assert_eq!(*electrode, ch);
        assert_eq!(samples.len(), 2);
        for (s, sample) in samples.iter().enumerate() {
            assert_f64_close(*sample, (s * 8 + ch) as f64 * ATHENA_EEG_SCALE);
        }
    }
}

#[test]
fn eeg_14bit_minmax_scale() {
    let mut values = vec![0u32; 16];
    for v in values.iter_mut().skip(8) {
        *v = (1 << 14) - 1;
    }
    let ev = parse_athena_notification(&packet(0x12, 0, &pack_uint_le(&values, 14)));
    let got = eeg_events(&ev);
    assert_f64_close(got[0].2[0], (0.0 - 8192.0) * ATHENA_EEG_SCALE);
    assert_f64_close(got[0].2[1], (16383.0 - 8192.0) * ATHENA_EEG_SCALE);
}

// ── IMU ──────────────────────────────────────────────────────────────────────

fn imu_payload(vals: &[i16]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn imu_decodes_all_three_samples_and_negates_gyro() {
    // 3 samples × (acc xyz + gyro xyz), 16-bit LE. Unlike the pre-optics
    // decoder, all three samples are forwarded (not copies of the first).
    let mut vals = [0i16; 18];
    for i in 0..3 {
        vals[i * 6] = 100 + i as i16;
        vals[i * 6 + 1] = -200 - i as i16;
        vals[i * 6 + 2] = 300 + i as i16;
        vals[i * 6 + 3] = 10 + i as i16;
        vals[i * 6 + 4] = -20 - i as i16;
        vals[i * 6 + 5] = 30 + i as i16;
    }
    let ev = parse_athena_notification(&packet(0x47, 42, &imu_payload(&vals)));
    const AS: f32 = 0.0000610352;
    const GS: f32 = -0.0074768;

    let accel: Vec<_> = ev
        .iter()
        .filter_map(|e| match e {
            MuseEvent::Accelerometer(d) => Some(d),
            _ => None,
        })
        .collect();
    let gyro: Vec<_> = ev
        .iter()
        .filter_map(|e| match e {
            MuseEvent::Gyroscope(d) => Some(d),
            _ => None,
        })
        .collect();
    assert_eq!(accel.len(), 1);
    assert_eq!(gyro.len(), 1);
    assert_eq!(accel[0].sequence_id, 42);
    assert_eq!(gyro[0].sequence_id, 42);
    for i in 0..3 {
        let a = &accel[0].samples[i];
        let g = &gyro[0].samples[i];
        let i = i as f32;
        assert_f32_close(a.x, (100.0 + i) * AS);
        assert_f32_close(a.y, (-200.0 - i) * AS);
        assert_f32_close(a.z, (300.0 + i) * AS);
        assert_f32_close(g.x, (10.0 + i) * GS);
        assert_f32_close(g.y, (-20.0 - i) * GS);
        assert_f32_close(g.z, (30.0 + i) * GS);
    }
}

// ── DRL / REF ────────────────────────────────────────────────────────────────

#[test]
fn drl_ref_decodes_six_interleaved_sample_pairs() {
    let mut payload = vec![0u8; 24];
    for i in 0..6u16 {
        payload[i as usize * 4..i as usize * 4 + 2].copy_from_slice(&(8000 + i).to_le_bytes());
        payload[i as usize * 4 + 2..i as usize * 4 + 4].copy_from_slice(&(8100 + i).to_le_bytes());
    }
    let ev = parse_athena_notification(&packet(0x53, 5, &payload));
    let drl: Vec<_> = ev
        .iter()
        .filter_map(|e| match e {
            MuseEvent::DrlRef(d) => Some(d),
            _ => None,
        })
        .collect();
    assert_eq!(drl.len(), 1);
    assert_eq!(drl[0].index, 5);
    assert_eq!(drl[0].samples.len(), 6);
    for i in 0..6u16 {
        assert_eq!(drl[0].samples[i as usize], [8000 + i, 8100 + i]);
    }
}

// ── Battery ──────────────────────────────────────────────────────────────────

fn battery_payload_20(raw: u16) -> Vec<u8> {
    let mut p = vec![0u8; 20];
    p[0..2].copy_from_slice(&raw.to_le_bytes());
    p
}

#[test]
fn battery_old_firmware_divides_by_256_and_clamps() {
    let ev = parse_athena_notification(&packet(0x98, 1, &battery_payload_20(12800)));
    let got = telemetry(&ev);
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].0, 1);
    assert_f32_close(got[0].1, 50.0);

    let full = parse_athena_notification(&packet(0x98, 0, &battery_payload_20(u16::MAX)));
    assert_f32_close(telemetry(&full)[0].1, 100.0);

    let empty = parse_athena_notification(&packet(0x98, 0, &battery_payload_20(0)));
    assert_f32_close(telemetry(&empty)[0].1, 0.0);
}

#[test]
fn battery_new_firmware_same_scale_on_variable_payload() {
    let mut payload = vec![0u8; 40];
    payload[0..2].copy_from_slice(&256u16.to_le_bytes()); // 256 / 256 = 1 %
    let ev = parse_athena_notification(&packet(0x88, 7, &payload));
    let got = telemetry(&ev);
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].0, 7);
    assert_f32_close(got[0].1, 1.0);
}

#[test]
fn undersize_length_byte_is_dropped_without_hanging() {
    // A declared length shorter than the 14-byte header used to be able to
    // rewind the cursor on the old 0x88 path. The packet-by-packet splitter
    // must just drop it.
    let mut data = vec![0u8; 16];
    data[0] = 0;
    data[9] = 0x88;
    data[14..16].copy_from_slice(&256u16.to_le_bytes());
    assert!(parse_athena_notification(&data).is_empty());
}

// ── Combined subpackets and decoder ──────────────────────────────────────────

#[test]
fn eeg_and_imu_subpackets_in_one_notification_both_decode() {
    let eeg = pack_uint_le(&eeg_values(4, 4), 14);
    let mut data = packet(0x11, 1, &eeg);
    data.push(0x47);
    data.push(2); // counter
    data.extend_from_slice(&[0, 0, 0]);
    data.extend(imu_payload(&[1i16; 18]));
    data[0] = data.len() as u8;
    let ev = parse_athena_notification(&data);
    assert_eq!(eeg_events(&ev).len(), 4);
    assert_eq!(
        ev.iter()
            .filter(|e| matches!(e, MuseEvent::Accelerometer(_)))
            .count(),
        1
    );
    assert_eq!(
        ev.iter()
            .filter(|e| matches!(e, MuseEvent::Gyroscope(_)))
            .count(),
        1
    );
}

#[test]
fn decoder_wrap_from_255_to_0_is_not_a_gap() {
    let mut decoder = AthenaDecoder::new();
    let a = packet(0x12, 255, &pack_uint_le(&eeg_values(8, 2), 14));
    let b = packet(0x12, 0, &pack_uint_le(&eeg_values(8, 2), 14));
    let ev: Vec<_> = decoder
        .decode(&a)
        .into_iter()
        .chain(decoder.decode(&b))
        .collect();
    assert!(ev.iter().all(|e| !matches!(e, MuseEvent::Gap(_))));
    assert_eq!(eeg_events(&ev).len(), 16);
}

#[test]
fn decoder_treats_a_backward_step_as_a_restart() {
    let mut decoder = AthenaDecoder::new();
    // wrapping_sub(199, 200) = 255, which is outside 2..=128, so this is
    // logged as a rewind/restart rather than a gap of ~255 packets.
    let a = packet(0x12, 200, &pack_uint_le(&eeg_values(8, 2), 14));
    let b = packet(0x12, 199, &pack_uint_le(&eeg_values(8, 2), 14));
    let ev: Vec<_> = decoder
        .decode(&a)
        .into_iter()
        .chain(decoder.decode(&b))
        .collect();
    assert!(ev.iter().all(|e| !matches!(e, MuseEvent::Gap(_))));
    assert_eq!(eeg_events(&ev).len(), 16);
}

#[test]
fn decoder_reports_an_eeg_gap_in_samples() {
    let mut decoder = AthenaDecoder::new();
    let a = packet(0x12, 1, &pack_uint_le(&eeg_values(8, 2), 14));
    let b = packet(0x12, 4, &pack_uint_le(&eeg_values(8, 2), 14));
    let ev: Vec<_> = decoder
        .decode(&a)
        .into_iter()
        .chain(decoder.decode(&b))
        .collect();
    let gaps: Vec<_> = ev
        .iter()
        .filter_map(|e| match e {
            MuseEvent::Gap(g) => Some((g.sensor, g.missing_packets, g.missing_samples)),
            _ => None,
        })
        .collect();
    // 1 → 4: two missing 8-ch subpackets of two samples each.
    assert_eq!(gaps, vec![(Sensor::Eeg, 2, 4)]);
}

#[test]
fn control_response_reads_rc() {
    let mut fields = serde_json::Map::new();
    fields.insert("rc".into(), serde_json::json!(0));
    let ok = ControlResponse {
        raw: r#"{"rc":0}"#.into(),
        fields,
    };
    assert_eq!(ok.rc(), Some(0));

    let empty = ControlResponse {
        raw: "{}".into(),
        fields: serde_json::Map::new(),
    };
    assert_eq!(empty.rc(), None);
}
