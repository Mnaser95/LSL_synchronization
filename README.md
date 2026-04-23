ECG leads:
Lead 1: LA-RA
Lead 2: LL-RA (Lead 1 + Lead 3)
Lead 3: LL-LA

Color coding for Shimmer ECG:
RA-white
LA-black
RL-...
LL-red
Vx-...

## Shimmer Packet Logic

The Shimmer data arrives as a continuous serial byte stream, not as already-separated packets.  
Because of that, the code has to figure out where each packet starts.

### ECG packet structure

For ECG, the code assumes one sample packet is 14 bytes:

- 3 bytes timestamp
- 11 bytes payload

Inside that payload, the useful values are read as:

- Lead II from one 24-bit channel field
- Lead I from another 24-bit channel field
- Lead III computed as `Lead II - Lead I`

So the ECG stream is treated as:

```text
1 packet = 1 ECG sample = 14 bytes
```

### EMG packet structure

For EMG, the code assumes one sample packet is 13 bytes:

- 1 byte packet type
- 3 bytes timestamp
- 9 bytes payload

The useful signal values are read as:

- EMG_CH1
- EMG_CH2

So the EMG stream is treated as:

```text
1 packet = 1 EMG sample = 13 bytes
```

## Sampling Rate and Delta

The Shimmer timestamp clock is treated as:

```text
32768 ticks/second
```

The sample-rate command sent to the device is computed from:

```text
register_value = 32768 / sampling_frequency
```

So for ECG:

```text
sampling_frequency = 256 Hz
delta = 32768 / 256 = 128 ticks
```

So two consecutive ECG packet timestamps should increase by about:

```text
128 ticks
```

For EMG:

```text
sampling_frequency = 512 Hz
delta = 32768 / 512 = 64 ticks
```

So two consecutive EMG packet timestamps should increase by about:

```text
64 ticks
```

## Why Packet Alignment Works

Because the serial input is just a raw byte stream, the code reads an initial chunk and tests every possible byte offset.

For each possible offset, it:

1. Pretends packets start at that offset
2. Reads timestamps separated by the assumed packet size
3. Computes the timestamp difference between consecutive packets
4. Compares that difference to the expected delta

For ECG the expected delta is:

```text
128 ticks
```

For EMG the expected delta is:

```text
64 ticks
```

The offset with the smallest timestamp error is chosen as the correct packet boundary.

That is why the packet size assumption is not just guessed once and trusted forever.  
It is checked against the observed timestamp behavior in the stream.

## Bytes per Second Example

For ECG:

```text
256 packets/second
14 bytes per packet
256 * 14 = 3584 bytes/second
```

That matches the note that one ECG sample packet is 14 bytes and the sampling rate is 256 Hz.
