import struct

class CCSDSPacketizer:
    """Encapsulates telemetry into CCSDS Space Packet Standard (CCSDS 133.0-B-2) units."""
    
    def __init__(self, process_id: int = 0x42):
        self.process_id = process_id & 0x07FF
        self.sequence_count = 0

    def pack(self, payload: bytes) -> bytes:
        packet_type = 0  # 0 = Telemetry
        sec_hdr_flag = 0 # No secondary header
        apid = self.process_id
        
        primary_hdr_1 = (0 << 13) | (packet_type << 12) | (sec_hdr_flag << 11) | apid
        
        sequence_flags = 3 # 3 = Unsegmented data
        seq_count = self.sequence_count & 0x3FFF
        primary_hdr_2 = (sequence_flags << 14) | seq_count
        
        packet_length = len(payload) - 1
        
        header = struct.pack(">HHH", primary_hdr_1, primary_hdr_2, packet_length)
        self.sequence_count = (self.sequence_count + 1) % 16384
        return header + payload
