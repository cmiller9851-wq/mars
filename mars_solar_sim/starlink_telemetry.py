import socket
import json
import time

STARLINK_DEFAULT_IP = "192.168.100.1"
STARLINK_GRPC_PORT = 9200

def fetch_starlink_telemetry(ip=STARLINK_DEFAULT_IP, port=STARLINK_GRPC_PORT, timeout=2.0):
    """
    Queries Starlink terminal network endpoint for link status and latency.
    Bypasses high-level wrapper overhead by attempting a direct TCP/HTTP2 probe.
    """
    print(f"Connecting to Starlink Transceiver at {ip}:{port}...")
    
    telemetry = {
        "connected": False,
        "latency_ms": None,
        "downlink_bps": None,
        "status": "DISCONNECTED"
    }
    
    start_time = time.time()
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((ip, port))
        
        latency = (time.time() - start_time) * 1000.0
        telemetry["connected"] = True
        telemetry["latency_ms"] = round(latency, 2)
        telemetry["status"] = "LINK_ACTIVE"
        sock.close()
        print(f"[OK] Starlink Link Established. Ping: {telemetry['latency_ms']} ms")
    except (socket.timeout, ConnectionRefusedError, OSError) as e:
        print(f"[WARN] Starlink Transceiver unreachable ({e}). Operating in autonomous offline mode.")
        telemetry["status"] = "OFFLINE_AUTONOMOUS"

    return telemetry

if __name__ == "__main__":
    stats = fetch_starlink_telemetry()
    print("Starlink Status Payload:", json.dumps(stats, indent=2))