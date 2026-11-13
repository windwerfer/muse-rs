//! Command framing, channel-name helpers, and the documented protocol constants.

use muse_rs::protocol::{
    athena_preset_layout, decode_response, eeg_channel_name, encode_command, optics_channel_name,
    ppg_channel_name, CommandTooLong, ACCELEROMETER_CHARACTERISTIC, ATHENA_DEFAULT_PRESET,
    ATHENA_EEG_CHANNELS, ATHENA_EEG_CHANNEL_NAMES, ATHENA_EEG_SAMPLES_PER_PKT, ATHENA_EEG_SCALE,
    ATHENA_PPG_CHANNELS, ATHENA_PPG_CHANNEL_NAMES, ATHENA_PPG_FREQUENCY,
    ATHENA_SENSOR_CHARACTERISTIC, CONTROL_CHARACTERISTIC, EEG_CHANNEL_NAMES, EEG_CHARACTERISTICS,
    EEG_FREQUENCY, EEG_SAMPLES_PER_READING, GYROSCOPE_CHARACTERISTIC, MAX_COMMAND_LEN,
    MUSE_SERVICE_UUID, OPTICS_CHANNEL_NAMES, PPG_CHANNEL_NAMES, PPG_CHARACTERISTICS, PPG_FREQUENCY,
    PPG_SAMPLES_PER_READING, TELEMETRY_CHARACTERISTIC,
};
use muse_rs::types::OpticsLayout;
use uuid::Uuid;

#[test]
fn encode_command_uses_length_prefix_and_newline() {
    assert_eq!(encode_command("d").unwrap(), vec![0x02, b'd', b'\n']);
    assert_eq!(encode_command("h").unwrap(), vec![0x02, b'h', b'\n']);
    assert_eq!(
        encode_command("p21").unwrap(),
        vec![0x04, b'p', b'2', b'1', b'\n']
    );
    assert_eq!(encode_command("").unwrap(), vec![0x01, b'\n']);
    let long = "v1";
    let bytes = encode_command(long).unwrap();
    assert_eq!(bytes[0] as usize, long.len() + 1);
    assert_eq!(&bytes[1..], b"v1\n");
}

#[test]
fn encode_command_rejects_a_payload_that_cannot_fit_the_length_byte() {
    let max = "x".repeat(MAX_COMMAND_LEN);
    let framed = encode_command(&max).expect("254-byte command must fit");
    assert_eq!(framed[0] as usize, MAX_COMMAND_LEN + 1);
    assert_eq!(framed.len(), MAX_COMMAND_LEN + 2); // length byte + cmd + newline

    let too_long = "x".repeat(MAX_COMMAND_LEN + 1);
    assert_eq!(
        encode_command(&too_long),
        Err(CommandTooLong {
            command_len: MAX_COMMAND_LEN + 1
        })
    );
}

#[test]
fn decode_response_honours_length_prefix() {
    assert_eq!(decode_response(&[]), "");
    assert_eq!(
        decode_response(&[0x05, b'h', b'e', b'l', b'l', b'o']),
        "hello"
    );
    assert_eq!(decode_response(&[0x0A, b'h', b'i']), "hi");
    assert_eq!(decode_response(&[0x00, b'x']), "");
}

#[test]
fn decode_response_is_lossy_for_invalid_utf8() {
    let s = decode_response(&[0x02, 0xFF, 0xFE]);
    assert!(
        s.contains('\u{FFFD}'),
        "expected replacement char, got {s:?}"
    );
}

#[test]
fn encode_then_decode_roundtrips_the_command_body() {
    for cmd in ["d", "h", "s", "v1", "p21", "p50", "p1041", "p1045"] {
        let framed = encode_command(cmd).unwrap();
        assert_eq!(decode_response(&framed), format!("{cmd}\n"));
    }
}

#[test]
fn sampling_constants_match_the_documented_rates() {
    assert_eq!(EEG_FREQUENCY, 256.0);
    assert_eq!(EEG_SAMPLES_PER_READING, 12);
    assert_eq!(PPG_FREQUENCY, 64.0);
    assert_eq!(PPG_SAMPLES_PER_READING, 6);
    assert_eq!(ATHENA_EEG_CHANNELS, 8);
    assert_eq!(ATHENA_EEG_SAMPLES_PER_PKT, 2);
    assert_eq!(ATHENA_PPG_CHANNELS, 16);
    assert_eq!(ATHENA_PPG_FREQUENCY, 64.0);
    assert_eq!(ATHENA_EEG_SCALE, 0.0885);
    assert_eq!(
        ATHENA_EEG_CHANNELS * ATHENA_EEG_SAMPLES_PER_PKT * 14,
        28 * 8
    );
}

#[test]
fn channel_names_follow_characteristic_index_order() {
    assert_eq!(EEG_CHANNEL_NAMES, ["TP9", "AF7", "AF8", "TP10", "AUX"]);
    assert_eq!(PPG_CHANNEL_NAMES, ["ambient", "infrared", "red"]);
    assert_eq!(EEG_CHARACTERISTICS.len(), EEG_CHANNEL_NAMES.len());
    assert_eq!(PPG_CHARACTERISTICS.len(), PPG_CHANNEL_NAMES.len());
    assert_eq!(
        ATHENA_EEG_CHANNEL_NAMES,
        ["TP9", "AF7", "AF8", "TP10", "FPz", "AUX_R", "AUX_L", "AUX"]
    );
    assert_eq!(OPTICS_CHANNEL_NAMES, ATHENA_PPG_CHANNEL_NAMES);
    assert_eq!(eeg_channel_name(4, false), "AUX");
    assert_eq!(eeg_channel_name(4, true), "FPz");
    assert_eq!(eeg_channel_name(7, true), "AUX");
    assert_eq!(eeg_channel_name(8, true), "?");
    assert_eq!(ppg_channel_name(2), "red");
    assert_eq!(ppg_channel_name(15), "opt15");
    assert_eq!(ppg_channel_name(16), "?");
    assert_eq!(optics_channel_name(0), "ambient");
    assert_eq!(optics_channel_name(99), "?");
}

#[test]
fn optics_layout_helpers_match_tags_and_presets() {
    assert_eq!(OpticsLayout::from_tag(0x34), Some(OpticsLayout::Ch4));
    assert_eq!(OpticsLayout::from_tag(0x35), Some(OpticsLayout::Ch8));
    assert_eq!(OpticsLayout::from_tag(0x36), Some(OpticsLayout::Ch16));
    assert_eq!(OpticsLayout::from_tag(0x12), None);
    assert_eq!(OpticsLayout::Ch4.channel_count(), 4);
    assert_eq!(OpticsLayout::Ch8.channel_count(), 8);
    assert_eq!(OpticsLayout::Ch16.channel_count(), 16);
    assert_eq!(OpticsLayout::Ch4.samples_per_packet(), 3);
    assert_eq!(OpticsLayout::Ch8.samples_per_packet(), 2);
    assert_eq!(OpticsLayout::Ch16.samples_per_packet(), 1);
    assert_eq!(OpticsLayout::Ch4.preset(), "p1045");
    assert_eq!(OpticsLayout::Ch8.preset(), "p1044");
    assert_eq!(OpticsLayout::Ch16.preset(), "p1041");
    // A few extra measured aliases beyond the three already checked in athena.rs.
    assert_eq!(athena_preset_layout("p1035"), Some(OpticsLayout::Ch4));
    assert_eq!(athena_preset_layout("p1043"), Some(OpticsLayout::Ch8));
    assert_eq!(athena_preset_layout("p1042"), Some(OpticsLayout::Ch16));
    assert_eq!(athena_preset_layout("p61"), None);
    assert_eq!(ATHENA_DEFAULT_PRESET, "p1041");
}

#[test]
fn gatt_uuids_use_the_muse_vendor_namespace() {
    let ns = "4c4d-454d-96be-f03bac821358";
    assert_eq!(
        CONTROL_CHARACTERISTIC,
        Uuid::parse_str(&format!("273e0001-{ns}")).unwrap()
    );
    assert_eq!(
        TELEMETRY_CHARACTERISTIC,
        Uuid::parse_str(&format!("273e000b-{ns}")).unwrap()
    );
    assert_eq!(
        GYROSCOPE_CHARACTERISTIC,
        Uuid::parse_str(&format!("273e0009-{ns}")).unwrap()
    );
    assert_eq!(
        ACCELEROMETER_CHARACTERISTIC,
        Uuid::parse_str(&format!("273e000a-{ns}")).unwrap()
    );
    assert_eq!(
        ATHENA_SENSOR_CHARACTERISTIC,
        Uuid::parse_str(&format!("273e0013-{ns}")).unwrap()
    );
    assert_eq!(
        EEG_CHARACTERISTICS[0],
        Uuid::parse_str(&format!("273e0003-{ns}")).unwrap()
    );
    assert_eq!(
        EEG_CHARACTERISTICS[4],
        Uuid::parse_str(&format!("273e0007-{ns}")).unwrap()
    );
    assert_eq!(
        PPG_CHARACTERISTICS[0],
        Uuid::parse_str(&format!("273e000f-{ns}")).unwrap()
    );
    assert_eq!(
        PPG_CHARACTERISTICS[2],
        Uuid::parse_str(&format!("273e0011-{ns}")).unwrap()
    );
    assert_eq!(
        MUSE_SERVICE_UUID,
        Uuid::parse_str("0000fe8d-0000-1000-8000-00805f9b34fb").unwrap()
    );
}
