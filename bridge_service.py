import json
import time
import logging
from typing import Dict, Tuple, Optional, Any

import requests
import paho.mqtt.client as mqtt


# =========================
# CONFIG
# =========================

MQTT_BROKER = "localhost"          # change when needed
MQTT_PORT = 1883
MQTT_TOPIC = "localization/data"
MQTT_KEEPALIVE = 60

AWS_API_URL = "INSERT_HANKS_API_URL_HERE"

# Quentin's adjustable anchor coordinates (meters)
ANCHOR_COORDS: Dict[str, Tuple[float, float]] = {
    "anchor_1": (0.0, 0.0),
    "anchor_2": (5.0, 0.0),
    "anchor_3": (2.5, 5.0),
}

REFERENCE_RSSI = -40.0   # A
PATH_LOSS_EXPONENT = 2.0 # n

HTTP_TIMEOUT = 5
MIN_DISTANCE_METERS = 0.1
MAX_DISTANCE_METERS = 50.0

LOG_LEVEL = logging.INFO


# =========================
# LOGGING
# =========================

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("localization_bridge")


# =========================
# MATH
# =========================

def rssi_to_distance(rssi: float, reference_rssi: float, path_loss_exponent: float) -> float:
    """
    Log-Distance Path Loss Model
    distance = 10 ^ ((A - RSSI) / (10 * n))
    """
    distance = 10 ** ((reference_rssi - rssi) / (10 * path_loss_exponent))
    return max(MIN_DISTANCE_METERS, min(distance, MAX_DISTANCE_METERS))


def trilaterate(
    p1: Tuple[float, float], r1: float,
    p2: Tuple[float, float], r2: float,
    p3: Tuple[float, float], r3: float
) -> Optional[Tuple[float, float]]:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    A = 2 * (x2 - x1)
    B = 2 * (y2 - y1)
    C = r1**2 - r2**2 - x1**2 + x2**2 - y1**2 + y2**2

    D = 2 * (x3 - x1)
    E = 2 * (y3 - y1)
    F = r1**2 - r3**2 - x1**2 + x3**2 - y1**2 + y3**2

    denom = A * E - B * D
    if abs(denom) < 1e-9:
        logger.error("Trilateration failed: anchors may be collinear or invalid.")
        return None

    x = (C * E - B * F) / denom
    y = (A * F - C * D) / denom
    return (x, y)


# =========================
# NORMALIZATION
# =========================

def parse_anchor_list(anchors: list) -> Dict[str, float]:
    """
    Converts:
    [
      {"id": "anchor_1", "rssi": -45},
      {"id": "anchor_2", "rssi": -55}
    ]
    into:
    {
      "anchor_1": -45.0,
      "anchor_2": -55.0
    }
    """
    out = {}
    for item in anchors:
        if not isinstance(item, dict):
            continue
        anchor_id = item.get("id")
        rssi = item.get("rssi")
        if anchor_id is None or rssi is None:
            continue
        try:
            out[str(anchor_id)] = float(rssi)
        except (TypeError, ValueError):
            logger.warning("Invalid RSSI in anchor list for %r", anchor_id)
    return out


def normalize_input(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Supports either:
    1) MQTT sample format:
       {
         "tag_id": "tag_01",
         "timestamp": 1711658400,
         "anchors": [
           {"id": "anchor_1", "rssi": -45},
           {"id": "anchor_2", "rssi": -55},
           {"id": "anchor_3", "rssi": -60}
         ]
       }

    2) Likely backend-style format:
       {
         "device_id": "tag_0",
         "timestamp": 1711658400,
         "rssi_vector": {
           "anchor_1": -37,
           "anchor_2": -41,
           "anchor_3": -45
         },
         ...
       }
    """

    if not isinstance(data, dict):
        return None

    normalized = {
        "device_id": data.get("device_id") or data.get("tag_id"),
        "timestamp": data.get("timestamp"),
        "_raw": data
    }

    # Case 1: anchors array
    if isinstance(data.get("anchors"), list):
        normalized["rssi_vector"] = parse_anchor_list(data["anchors"])
        return normalized

    # Case 2: rssi_vector dict
    if isinstance(data.get("rssi_vector"), dict):
        vector = {}
        for k, v in data["rssi_vector"].items():
            try:
                vector[str(k)] = float(v)
            except (TypeError, ValueError):
                logger.warning("Invalid RSSI value in rssi_vector for %r", k)
        normalized["rssi_vector"] = vector
        return normalized

    return None


# =========================
# CORE PROCESSING
# =========================

def calculate_position(rssi_vector: Dict[str, float]) -> Optional[Tuple[float, float]]:
    required_ids = list(ANCHOR_COORDS.keys())

    missing = [aid for aid in required_ids if aid not in rssi_vector]
    if missing:
        logger.error("Missing required anchors: %s", missing)
        return None

    d1 = rssi_to_distance(rssi_vector[required_ids[0]], REFERENCE_RSSI, PATH_LOSS_EXPONENT)
    d2 = rssi_to_distance(rssi_vector[required_ids[1]], REFERENCE_RSSI, PATH_LOSS_EXPONENT)
    d3 = rssi_to_distance(rssi_vector[required_ids[2]], REFERENCE_RSSI, PATH_LOSS_EXPONENT)

    logger.info(
        "Distances -> %s: %.3f m, %s: %.3f m, %s: %.3f m",
        required_ids[0], d1, required_ids[1], d2, required_ids[2], d3
    )

    return trilaterate(
        ANCHOR_COORDS[required_ids[0]], d1,
        ANCHOR_COORDS[required_ids[1]], d2,
        ANCHOR_COORDS[required_ids[2]], d3
    )


def build_output_payload(raw_data: Dict[str, Any], x: float, y: float) -> Dict[str, Any]:
    """
    Preserve metadata from the incoming message and add updated x,y.
    Works for either tag_id or device_id style input.
    """
    output = dict(raw_data)

    # Preserve whichever naming the real system uses
    if "device_id" not in output and "tag_id" not in output:
        output["device_id"] = "unknown_device"

    output["x"] = round(float(x), 4)
    output["y"] = round(float(y), 4)

    return output


def process_message(payload: str) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        logger.error("Malformed JSON: %s", payload)
        return None

    normalized = normalize_input(data)
    if normalized is None:
        logger.error("Unsupported input format.")
        return None

    device_id = normalized.get("device_id")
    timestamp = normalized.get("timestamp")
    rssi_vector = normalized.get("rssi_vector", {})
    raw_data = normalized.get("_raw", {})

    if device_id is None:
        logger.error("Missing device_id/tag_id.")
        return None

    if timestamp is None:
        logger.error("Missing timestamp.")
        return None

    if not isinstance(rssi_vector, dict) or not rssi_vector:
        logger.error("Missing or empty RSSI data.")
        return None

    logger.info("Processing device=%s timestamp=%s", device_id, timestamp)

    xy = calculate_position(rssi_vector)
    if xy is None:
        return None

    x, y = xy
    return build_output_payload(raw_data, x, y)


# =========================
# AWS FORWARDING
# =========================

def send_to_aws(payload: Dict[str, Any]) -> bool:
    if not AWS_API_URL or AWS_API_URL == "INSERT_HANKS_API_URL_HERE":
        logger.warning("AWS_API_URL not set. Printing payload only.")
        logger.info("Would send to AWS: %s", json.dumps(payload))
        return False

    try:
        response = requests.post(AWS_API_URL, json=payload, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        logger.info("AWS POST success: %s %s", response.status_code, response.text)
        return True
    except requests.RequestException as e:
        logger.error("AWS POST failed: %s", e)
        return False


# =========================
# MQTT CALLBACKS
# =========================

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        logger.info("Connected to MQTT broker %s:%s", MQTT_BROKER, MQTT_PORT)
        client.subscribe(MQTT_TOPIC)
        logger.info("Subscribed to topic: %s", MQTT_TOPIC)
    else:
        logger.error("MQTT connect failed with rc=%s", rc)


def on_disconnect(client, userdata, disconnect_flags, reason_code, properties=None):
    logger.warning("Disconnected from MQTT broker. reason_code=%s", reason_code)


def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode("utf-8", errors="replace")
        logger.info("Received message on %s: %s", msg.topic, payload)

        result = process_message(payload)
        if result is None:
            logger.warning("Processing failed for current message.")
            return

        logger.info("Calculated result: %s", json.dumps(result))
        send_to_aws(result)

    except Exception as e:
        logger.exception("Unexpected error in on_message: %s", e)


# =========================
# MAIN
# =========================

def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=30)

    while True:
        try:
            logger.info("Connecting to MQTT broker...")
            client.connect(MQTT_BROKER, MQTT_PORT, MQTT_KEEPALIVE)
            client.loop_forever()
        except KeyboardInterrupt:
            logger.info("Stopped by user.")
            break
        except Exception as e:
            logger.exception("MQTT loop crashed: %s", e)
            logger.info("Retrying in 5 seconds...")
            time.sleep(5)


if __name__ == "__main__":
    main()