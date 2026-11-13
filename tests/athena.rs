//! Athena decoding: constructed packets plus the hand-built cases from
//! eugenehp/muse-rs 0.2.3.
//!
//! Upstream's `tests/fixtures/athena_fw3.1.19_p1041.hex` (80 notifications
//! recorded on a Muse S Athena, fw 3.1.19, preset p1041) is documented in
//! `docs/athena-protocol.md` but was not committed to eugenehp/muse-rs, so
//! the recording-based tests are omitted here. Re-add them if that dump is
//! published.

use muse_rs::parse::{parse_athena_notification, parse_ppg_reading, AthenaDecoder};
use muse_rs::protocol::{athena_preset_layout, optics_channel_name, ATHENA_DEFAULT_PRESET};
use muse_rs::types::{MuseEvent, OpticsLayout, Sensor};

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

/// One Athena packet: length-prefixed 14-byte header, then a single subpacket payload.
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

fn optics_values(n_ch: usize, n_samp: usize) -> Vec<u32> {
    // sample-major: lane i, sample s → 0x10000 * i + s
    (0..n_samp)
        .flat_map(|s| (0..n_ch).map(move |ch| (0x10000 * ch + s) as u32))
        .collect()
}

fn optics_events(events: &[MuseEvent]) -> Vec<(usize, u16, Vec<u32>, OpticsLayout)> {
    events
        .iter()
        .filter_map(|e| match e {
            MuseEvent::Optics(r) => Some((r.lane, r.index, r.samples.clone(), r.layout)),
            _ => None,
        })
        .collect()
}

fn assert_no_ppg(events: &[MuseEvent]) {
    assert!(
        events.iter().all(|e| !matches!(e, MuseEvent::Ppg(_))),
        "Athena must not emit Classic Ppg"
    );
}

#[test]
fn optics4_decodes_all_four_lanes() {
    let values = optics_values(4, 3);
    let ev = parse_athena_notification(&packet(0x34, 7, &pack_uint_le(&values, 20)));
    let got = optics_events(&ev);
    assert_no_ppg(&ev);
    assert_eq!(got.len(), 4, "the old parser kept only 3 of 4 lanes");
    for (ch, g) in got.iter().enumerate() {
        assert_eq!(g.0, ch);
        assert_eq!(g.1, 7);
        assert_eq!(
            g.2,
            vec![
                0x10000 * ch as u32,
                0x10000 * ch as u32 + 1,
                0x10000 * ch as u32 + 2
            ]
        );
        assert_eq!(g.3, OpticsLayout::Ch4);
    }
    assert!(ev.iter().all(|e| !matches!(e, MuseEvent::Eeg(_))));
}

#[test]
fn optics8_decodes_all_eight_lanes() {
    let values = optics_values(8, 2);
    let ev = parse_athena_notification(&packet(0x35, 1, &pack_uint_le(&values, 20)));
    let got = optics_events(&ev);
    assert_no_ppg(&ev);
    assert_eq!(got.len(), 8);
    for (ch, g) in got.iter().enumerate() {
        assert_eq!(g.0, ch);
        assert_eq!(g.2, vec![0x10000 * ch as u32, 0x10000 * ch as u32 + 1]);
        assert_eq!(g.3, OpticsLayout::Ch8);
    }
}

#[test]
fn optics16_is_decoded_not_skipped() {
    let values = optics_values(16, 1);
    let ev = parse_athena_notification(&packet(0x36, 3, &pack_uint_le(&values, 20)));
    let got = optics_events(&ev);
    assert_no_ppg(&ev);
    assert_eq!(got.len(), 16, "tag 0x36 used to be skipped");
    for ch in 0..16 {
        assert_eq!(got[ch].0, ch);
        assert_eq!(got[ch].2, vec![0x10000 * ch as u32]);
        assert_eq!(got[ch].3, OpticsLayout::Ch16);
        assert_eq!(
            optics_channel_name(ch),
            [
                "ambient", "infrared", "red", "opt3", "opt4", "opt5", "opt6", "opt7", "opt8",
                "opt9", "opt10", "opt11", "opt12", "opt13", "opt14", "opt15"
            ][ch]
        );
    }
}

#[test]
fn short_optical_payload_is_dropped() {
    // Declared packet length matches what arrived, but the optics payload is short.
    let mut p = packet(0x34, 0, &[0u8; 30]);
    p.truncate(14 + 10);
    p[0] = p.len() as u8;
    assert!(parse_athena_notification(&p).is_empty());
    assert!(parse_athena_notification(&[]).is_empty());
}

/// Two packets in one notification decode as the two did apart.
///
/// The parser used to read on past the first packet's end, taking the
/// second's header for subpackets: any header byte equal to a sensor tag
/// produced a reading made of header bytes, and the second packet's real
/// data was lost.
#[test]
fn splits_notifications_holding_several_packets() {
    let a = packet(0x34, 0, &pack_uint_le(&optics_values(4, 3), 20));
    let b = packet(0x36, 1, &pack_uint_le(&optics_values(16, 1), 20));
    let joined: Vec<u8> = [a.clone(), b.clone()].concat();
    let apart: Vec<_> = parse_athena_notification(&a)
        .into_iter()
        .chain(parse_athena_notification(&b))
        .collect();
    let together = parse_athena_notification(&joined);
    assert_eq!(optics_events(&together), optics_events(&apart));
    assert_eq!(optics_events(&together).len(), 4 + 16);
}

/// A header byte that happens to equal a tag is not read as one.
#[test]
fn header_bytes_are_not_taken_for_tags() {
    // Two packets of one IMU subpacket each, the device clock in the second
    // header holding 0x12 — the EEG tag — at every byte.
    let packet = |counter: u8| {
        let mut p = vec![0u8; 14 + 36];
        p[0] = p.len() as u8;
        p[2..6].copy_from_slice(&[0x12; 4]);
        p[9] = 0x47;
        p[10] = counter;
        p
    };
    let joined = [packet(0), packet(1)].concat();
    let ev = parse_athena_notification(&joined);
    let accel = ev
        .iter()
        .filter(|e| matches!(e, MuseEvent::Accelerometer(_)))
        .count();
    let gyro = ev
        .iter()
        .filter(|e| matches!(e, MuseEvent::Gyroscope(_)))
        .count();
    let eeg = ev.iter().filter(|e| matches!(e, MuseEvent::Eeg(_))).count();
    assert_eq!((eeg, accel, gyro), (0, 2, 2));
}

/// An unknown tag ends its packet; the next packet still decodes.
#[test]
fn unknown_tag_ends_only_its_packet() {
    let first = packet(0x34, 0, &pack_uint_le(&optics_values(4, 3), 20));
    // Append a bogus subpacket tag after the first payload, inside a longer
    // declared length, then a second well-formed 16-ch packet.
    let mut broken = first.clone();
    broken.extend_from_slice(&[0xEE, 0, 0, 0, 0]);
    broken[0] = broken.len() as u8;
    let second = packet(0x36, 1, &pack_uint_le(&optics_values(16, 1), 20));
    let ev = parse_athena_notification(&[broken, second].concat());
    assert_eq!(optics_events(&ev).len(), 4 + 16);
}

#[test]
fn malformed_input_is_dropped() {
    let n = packet(0x34, 0, &pack_uint_le(&optics_values(4, 3), 20));
    for len in 0..n.len() {
        // A truncated packet's declared length exceeds what arrived.
        assert!(parse_athena_notification(&n[..len]).is_empty(), "len {len}");
    }
    assert!(parse_athena_notification(&[0u8; 40]).is_empty());
    assert!(parse_athena_notification(&[5u8; 40]).is_empty());
}

#[test]
fn decoder_reports_an_optical_gap() {
    let mut decoder = AthenaDecoder::new();
    let a = packet(0x36, 107, &pack_uint_le(&optics_values(16, 1), 20));
    let b = packet(0x36, 110, &pack_uint_le(&optics_values(16, 1), 20));
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
    // 107 → 110: two 16-channel subpackets of one sample each.
    assert_eq!(gaps, vec![(Sensor::Optical, 2, 2)]);
    assert_eq!(optics_events(&ev).len(), 32);
}

/// Classic 24-bit PPG still yields 6 samples (regression).
#[test]
fn classic_ppg_reading_still_has_six_samples() {
    let mut data = vec![0u8; 2 + 18]; // index + 6 × 24-bit
    data[0..2].copy_from_slice(&1u16.to_be_bytes());
    // six samples: 0x010203, 0x040506, ...
    for i in 0..6 {
        let v = 0x010203 + i * 0x030303;
        let off = 2 + i * 3;
        data[off] = ((v >> 16) & 0xff) as u8;
        data[off + 1] = ((v >> 8) & 0xff) as u8;
        data[off + 2] = (v & 0xff) as u8;
    }
    let r = parse_ppg_reading(&data, 1, 12.0).expect("classic ppg");
    assert_eq!(r.ppg_channel, 1);
    assert_eq!(r.index, 1);
    assert_eq!(r.timestamp, 12.0);
    assert_eq!(r.samples.len(), 6);
    assert_eq!(r.samples[0], 0x010203);
    assert_eq!(r.samples[5], 0x010203 + 5 * 0x030303);
}

#[test]
fn default_preset_is_16_lane_p1041() {
    assert_eq!(ATHENA_DEFAULT_PRESET, "p1041");
    assert_eq!(athena_preset_layout("p1041"), Some(OpticsLayout::Ch16));
    assert_eq!(athena_preset_layout("p1044"), Some(OpticsLayout::Ch8));
    assert_eq!(athena_preset_layout("p1045"), Some(OpticsLayout::Ch4));
    assert_eq!(athena_preset_layout("p21"), None);
    assert_eq!(OpticsLayout::Ch16.preset(), "p1041");
}
