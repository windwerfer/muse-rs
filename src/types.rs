/// An EEG reading — one BLE notification from a single electrode.
///
/// # Classic firmware (Muse 1 / Muse 2 / Muse S ≤ fw 3.x)
/// One notification per channel at 256 Hz, carrying **12 samples** each
/// (≈ 46.9 ms of signal per packet).  Samples are decoded from 12-bit
/// big-endian packed values and scaled by 0.48828125 µV/LSB.
///
/// # Athena firmware (Muse S fw ≥ 4.x)
/// All channels arrive on a single universal characteristic.  Each notification
/// is split into per-channel `EegReading`s carrying **2 samples** decoded from
/// 14-bit little-endian packed values and scaled by 0.0885 µV/LSB.
/// Channels 4–7 (FPz, AUX\_R, AUX\_L, AUX) are only produced by Athena hardware.
#[derive(Debug, Clone)]
pub struct EegReading {
    /// Sequential packet index emitted by the headset.
    ///
    /// **Classic:** 16-bit, per electrode; used to reconstruct timestamps.
    /// **Athena:** the EEG subpacket counter, 8-bit (wraps at 255) and shared
    /// by every electrode in the subpacket. A skip is reported as a
    /// [`MuseEvent::Gap`].
    pub index: u16,
    /// Electrode channel index.
    ///
    /// **Classic (channels 0–4):**
    /// * 0 = TP9 (left rear)
    /// * 1 = AF7 (left front)
    /// * 2 = AF8 (right front)
    /// * 3 = TP10 (right rear)
    /// * 4 = AUX (optional, `enable_aux: true` required)
    ///
    /// **Athena (channels 0–7):** same 0–3 as above, plus:
    /// * 4 = FPz (frontal midline)
    /// * 5 = AUX\_R
    /// * 6 = AUX\_L
    /// * 7 = AUX
    pub electrode: usize,
    /// Wall-clock timestamp in milliseconds since Unix epoch for the *first*
    /// sample in this packet.
    ///
    /// **Classic:** extrapolated from `index` and the known sample rate;
    /// accurate even under BLE jitter.
    ///
    /// **Athena:** always `0.0` — Athena packets do not carry a per-channel
    /// index, so timestamp reconstruction is not possible with current data.
    pub timestamp: f64,
    /// Voltage samples in µV.
    ///
    /// **Classic:** 12 samples per packet at 256 Hz.  
    /// **Athena:** 2 samples per packet at 256 Hz.
    pub samples: Vec<f64>,
}

/// A PPG (photoplethysmography) reading from the optical heart-rate sensor.
///
/// Available on Muse 2 and Muse S only. Each notification carries 6 raw
/// 24-bit samples at 64 Hz.
#[derive(Debug, Clone)]
pub struct PpgReading {
    /// Sequential packet index (16-bit, Classic GATT notifications).
    pub index: u16,
    /// Optical channel: 0 = ambient, 1 = infrared, 2 = red.
    ///
    /// Classic only. Athena optical data is emitted as [`OpticsReading`],
    /// never as `PpgReading`.
    pub ppg_channel: usize,
    /// Wall-clock timestamp in milliseconds since Unix epoch for the first sample.
    pub timestamp: f64,
    /// Raw 24-bit ADC values (not scaled to physical units).
    /// 6 samples per notification at 64 Hz.
    pub samples: Vec<u32>,
}

/// How many optical lanes an Athena preset is streaming.
///
/// Selected by the firmware preset (see [`crate::protocol::athena_preset_layout`]).
/// Labels for the lanes are [`crate::protocol::optics_channel_name`]; they are
/// names, not measured wavelengths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpticsLayout {
    /// Tag `0x34`: 4 lanes × 3 samples (e.g. `p1045`).
    Ch4,
    /// Tag `0x35`: 8 lanes × 2 samples (e.g. `p1044`).
    Ch8,
    /// Tag `0x36`: 16 lanes × 1 sample (e.g. `p1041`, the default).
    Ch16,
}

impl OpticsLayout {
    /// Layout carried by an Athena optical tag, if this is one.
    pub fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0x34 => Some(Self::Ch4),
            0x35 => Some(Self::Ch8),
            0x36 => Some(Self::Ch16),
            _ => None,
        }
    }

    pub fn channel_count(self) -> usize {
        match self {
            Self::Ch4 => 4,
            Self::Ch8 => 8,
            Self::Ch16 => 16,
        }
    }

    pub fn samples_per_packet(self) -> usize {
        match self {
            Self::Ch4 => 3,
            Self::Ch8 => 2,
            Self::Ch16 => 1,
        }
    }

    /// A representative 8-EEG-channel preset that selects this layout.
    pub fn preset(self) -> &'static str {
        match self {
            Self::Ch4 => "p1045",
            Self::Ch8 => "p1044",
            Self::Ch16 => "p1041",
        }
    }
}

/// One lane of an Athena optical (fNIRS-style) packet.
///
/// Athena only. Values are raw 20-bit ADC counts, not a derived PPG, HR or
/// SpO₂ reading and not a calibrated wavelength.
#[derive(Debug, Clone)]
pub struct OpticsReading {
    /// The optical subpacket counter (8-bit).
    pub index: u16,
    pub layout: OpticsLayout,
    /// Lane index in `0 .. layout.channel_count()`.
    /// See [`crate::protocol::optics_channel_name`].
    pub lane: usize,
    /// Wall-clock timestamp in milliseconds since Unix epoch for the first sample.
    /// Athena packets do not carry a usable sample clock here; often `0.0`.
    pub timestamp: f64,
    /// Raw 20-bit ADC counts, oldest first. Length is `layout.samples_per_packet()`.
    pub samples: Vec<u32>,
}

/// Battery and housekeeping telemetry packet.
///
/// Sent by the headset roughly once per second on both Classic and Athena
/// firmware, though the wire format differs:
///
/// | Field | Classic | Athena |
/// |---|---|---|
/// | `sequence_id` | from packet | battery subpacket counter (8-bit) |
/// | `battery_level` | u16 BE ÷ 512 | u16 LE ÷ 256 |
/// | `fuel_gauge_voltage` | u16 BE × 2.2 mV | always `0.0` |
/// | `temperature` | u16 BE raw ADC | always `0` |
#[derive(Debug, Clone)]
pub struct TelemetryData {
    /// Monotonically increasing packet counter: 16-bit on Classic, the 8-bit
    /// battery subpacket counter on Athena.
    pub sequence_id: u16,
    /// Battery state-of-charge in percent (0–100).
    /// Derived from the raw fuel-gauge reading divided by 512.
    pub battery_level: f32,
    /// Fuel-gauge terminal voltage in millivolts (Classic only).
    /// Raw reading multiplied by 2.2.  Always `0.0` for Athena.
    pub fuel_gauge_voltage: f32,
    /// Raw ADC temperature value (not converted to °C).
    /// Always `0` for Athena (field not present in Athena battery packets).
    pub temperature: u16,
}

/// A single 3-axis inertial measurement.
#[derive(Debug, Clone, Copy)]
pub struct XyzSample {
    /// X-axis value in sensor-specific units (g for accelerometer, °/s for gyroscope).
    pub x: f32,
    /// Y-axis value.
    pub y: f32,
    /// Z-axis value.
    pub z: f32,
}

/// A batch of inertial measurements from one BLE notification.
///
/// Both the accelerometer and gyroscope fire at ≈ 52 Hz and carry 3 XYZ
/// samples per notification, on both firmwares.
///
/// # Wire format differences
///
/// | Property | Classic | Athena |
/// |---|---|---|
/// | Integer type | `i16` big-endian | `i16` little-endian |
/// | Accel scale | +0.0000610352 g/LSB | +0.0000610352 g/LSB |
/// | Gyro scale | +0.0074768 °/s/LSB | −0.0074768 °/s/LSB (negated) |
/// | `sequence_id` | from packet | IMU subpacket counter (8-bit) |
#[derive(Debug, Clone)]
pub struct ImuData {
    /// Monotonically increasing packet counter: 16-bit on Classic, the 8-bit
    /// IMU subpacket counter on Athena.
    pub sequence_id: u16,
    /// Three consecutive XYZ samples; index 0 is the oldest.
    pub samples: [XyzSample; 3],
}

/// One Athena DRL/REF packet (tag `0x53`): the driven-right-leg and reference
/// electrode levels.
///
/// The layout — two channels, six samples each, as twelve little-endian `u16`
/// interleaved per sample — is the one it `DRL_REF` packet maker
/// declares, and it is confirmed by the shape of real data from a Muse S
/// Athena (fw 3.1.19): de-interleaved this way each channel holds a steady
/// level of its own (≈ 8232 and ≈ 8272) with a spread of one or two counts.
/// Any other reading smears the two together.
///
/// Values are raw. Which of the pair is DRL and which REF, and the conversion
/// to volts, are not documented anywhere this crate can cite, so neither is
/// guessed at.
#[derive(Debug, Clone)]
pub struct DrlRefData {
    /// The subpacket's sequence counter (0–255, wraps). See [`DataGap`].
    pub index: u16,
    /// Six samples, oldest first, each holding both channels.
    pub samples: Vec<[u16; 2]>,
}

/// Which stream a [`DataGap`] was found in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sensor {
    Eeg,
    Optical,
    Imu,
    DrlRef,
    Battery,
}

/// Data the headset counted out but that never arrived.
///
/// Athena numbers each sensor's subpackets with its own 8-bit counter, so a
/// counter that skips is data lost — and it is lost on the headset side: on a
/// Muse S Athena (fw 3.1.19) the EEG counter skipped roughly once every 140
/// subpackets while the counter of the packets carrying them never did. A
/// consumer that lays samples on a time axis by counting them needs this to
/// keep the axis honest; it fills the hole with NaN samples.
///
/// Only Athena produces these. A run of 256 or more lost subpackets wraps the
/// counter and cannot be seen.
#[derive(Debug, Clone)]
pub struct DataGap {
    pub sensor: Sensor,
    /// Subpackets missing between the previous one and this one.
    pub missing_packets: u32,
    /// The same gap in samples per channel, at the rate of the stream it is in.
    pub missing_samples: u32,
}

/// A time-aligned multi-channel EEG snapshot.
///
/// Produced by code that zips per-electrode [`EegReading`]s together so every
/// channel has a value for the same timestamp.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct EegSample {
    /// Packet index from the electrode that completed this sample set.
    pub index: u16,
    /// Wall-clock timestamp in milliseconds since Unix epoch.
    pub timestamp: f64,
    /// Voltage in µV for each channel in order `[TP9, AF7, AF8, TP10, AUX]`.
    /// Channels not yet received are `f64::NAN`.
    pub data: Vec<f64>,
}

/// A parsed Muse control/status response decoded from the control characteristic.
///
/// The headset replies to `v1` (device info), `s` (status), and similar
/// commands with a JSON object split across several BLE packets.
/// [`crate::parse::ControlAccumulator`] reassembles the fragments; this struct
/// carries the final result.
#[derive(Debug, Clone)]
pub struct ControlResponse {
    /// The raw, un-parsed JSON string.
    pub raw: String,
    /// Key-value pairs from the parsed JSON object.
    pub fields: serde_json::Map<String, serde_json::Value>,
}

impl ControlResponse {
    /// The return code, when this is a reply to a command.
    ///
    /// Every command the headset accepts is answered with an object carrying
    /// `rc` — `0` for success — alone or alongside the data asked for.
    pub fn rc(&self) -> Option<i64> {
        self.fields.get("rc").and_then(|v| v.as_i64())
    }
}

/// All data events emitted by [`crate::muse_client::MuseClient`].
///
/// Consumers receive these values through the `mpsc::Receiver` returned by
/// [`crate::muse_client::MuseClient::connect`] or
/// [`crate::muse_client::MuseClient::connect_to`].
///
/// # Firmware availability
///
/// | Variant | Classic | Athena |
/// |---|---|---|
/// | `Eeg` (ch 0–3) | ✓ | ✓ |
/// | `Eeg` (ch 4–7) | ✗ | ✓ |
/// | `Ppg` | ✓ (opt-in) | ✗ |
/// | `Optics` | ✗ | ✓ |
/// | `Accelerometer` | ✓ | ✓ |
/// | `Gyroscope` | ✓ | ✓ |
/// | `Telemetry` | ✓ | ✓ |
/// | `Control` | ✓ | ✓ |
/// | `DrlRef` | ✗ | ✓ |
/// | `Gap` | ✗ | ✓ |
///
/// Non-exhaustive: the headsets carry more than is decoded yet, and each new
/// stream would otherwise be a breaking change.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum MuseEvent {
    /// An EEG packet from one electrode channel.
    ///
    /// Produced for channels 0–3 on both Classic and Athena.
    /// Channels 4–7 (FPz, AUX\_R, AUX\_L, AUX) are Athena-only.
    /// See [`EegReading`] for per-firmware format differences.
    Eeg(EegReading),
    /// A Classic PPG (photoplethysmography) packet.
    ///
    /// Classic sends these only with `enable_ppg: true`. Athena never emits
    /// this variant: its optical stream is [`MuseEvent::Optics`].
    Ppg(PpgReading),
    /// An Athena optical (fNIRS-style) packet, one event per lane.
    ///
    /// 4, 8 or 16 lanes at 64 Hz depending on the preset. Default startup is
    /// `p1041` (16 lanes). See [`OpticsReading`].
    Optics(OpticsReading),
    /// Battery and housekeeping telemetry (~1 Hz).
    ///
    /// Produced by both Classic and Athena; some fields are `0` on Athena.
    /// See [`TelemetryData`] for per-firmware field availability.
    Telemetry(TelemetryData),
    /// Accelerometer batch (~52 Hz).
    ///
    /// Produced by both Classic and Athena.  The gyroscope sign convention and
    /// byte order differ between firmwares; see [`ImuData`] for details.
    Accelerometer(ImuData),
    /// Gyroscope batch (~52 Hz).
    ///
    /// Produced by both Classic and Athena.  Athena gyro values are negated
    /// relative to Classic (−0.0074768 vs +0.0074768 °/s/LSB).
    Gyroscope(ImuData),
    /// DRL/REF electrode levels (Athena; 32 Hz, six samples per packet).
    DrlRef(DrlRefData),
    /// Subpackets the headset counted out but that never arrived (Athena).
    Gap(DataGap),
    /// A complete JSON control/status response from the headset.
    ///
    /// Produced by both Classic and Athena in response to commands such as
    /// `v1` (device info) or `s` (status).
    Control(ControlResponse),
    /// The BLE link has been established and GATT services discovered.
    /// The inner `String` is the advertised device name (e.g. `"Muse-AB12"`).
    Connected(String),
    /// The BLE link was lost (headset turned off, out of range, etc.).
    ///
    /// After receiving this event the channel will be closed; no further
    /// events will arrive.  The TUI and CLI both restart scanning automatically
    /// when this event is seen.
    Disconnected,
}
