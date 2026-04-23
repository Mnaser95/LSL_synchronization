import serial
import struct
import time

PORT = "COM6"
BAUD = 115200

def wait_for_ack(ser, timeout=2):
    start = time.time()
    while time.time() - start < timeout:
        b = ser.read(1)
        if b == b'\xff':
            return True
    return False

def query_status(ser):
    ser.reset_input_buffer()
    ser.write(struct.pack('B', 0x72))
    time.sleep(0.4)
    data = ser.read(4)
    if len(data) != 4:
        return None
    ack, instream, rsp_cmd, status = struct.unpack('BBBB', data)
    return {
        "ack": ack,
        "instream": instream,
        "rsp_cmd": rsp_cmd,
        "status_byte": status,
        "self_cmd": (status & 0x04) >> 2,
        "sensing": (status & 0x02) >> 1,
        "docked": (status & 0x01),
    }

def query_sampling_rate(ser):
    ser.reset_input_buffer()
    ser.write(struct.pack('B', 0x03))
    wait_for_ack(ser)
    data = ser.read(3)
    if len(data) != 3:
        return None
    # byte 0 is response/command-related, next 2 bytes are register
    reg = struct.unpack('H', data[1:3])[0]
    hz = 32768.0 / reg if reg != 0 else None
    return {"register": reg, "hz": hz}

def query_derived_channels(ser):
    ser.reset_input_buffer()
    ser.write(struct.pack('B', 0x6F))
    wait_for_ack(ser)
    data = ser.read(4)
    if len(data) != 4:
        return None
    dc0, dc1, dc2 = struct.unpack('BBB', data[1:4])
    value = dc0 + dc1 * 256 + dc2 * 65536
    return {"dc0": dc0, "dc1": dc1, "dc2": dc2, "value": value}

def main():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    print("Opened", ser.name)

    status = query_status(ser)
    print("STATUS:", status)

    sr = query_sampling_rate(ser)
    print("SAMPLING RATE:", sr)

    dc = query_derived_channels(ser)
    print("DERIVED CHANNELS:", dc)

    ser.close()
    print("Closed")

if __name__ == "__main__":
    main()