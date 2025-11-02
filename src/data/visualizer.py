import math
import json
import argparse
import numpy as np
import cv2
import enum
import os
import copy

# Use the MLX90640 Python port for parsing and temperature computation
from .thermal_parser import (
    mlx90640_parameters,
    mlx_frame_parameters,
    mlx_parse_eeprom_parameters,
    mlx_parse_frame_parameters,
    get_temperature_of_pixel,
    get_pixel_in_image_mode,
    SUBPAGE_NUMBER,
    MLX90640_RES_ROWS,
    MLX90640_RES_COLS,
)

IMAGE_SCALE = 15
FRAMERATE = 8
MAX_TEMP = 45


class Sensor(enum.Enum):
    THERMAL = 1
    TOF = 2
    CAMERA = 3


shapes = {Sensor.THERMAL: (24, 32), Sensor.TOF: (8, 8), Sensor.CAMERA: (96, 96)}

def HSVtoBGR(h, s, v):
    i = math.floor(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    match (i % 6):
        case 0:
            r = v
            g = t
            b = p
        case 1:
            r = q
            g = v
            b = p
        case 2:
            r = p
            g = v
            b = t
        case 3:
            r = p
            g = q
            b = v
        case 4:
            r = t
            g = p
            b = v
        case 5:
            r = v
            g = p
            b = q
    return (b * 255, g * 255, r * 255)


def temperature_to_BGR(temperature):
    return HSVtoBGR(min(temperature, MAX_TEMP) / MAX_TEMP, 1, 1)


def distance_to_BGR(distance: int) -> tuple[int, int, int]:
    """Converts the distance to a BGR color"""
    return HSVtoBGR(distance / 4000, 1, 1)


def visualize_data(
    matrix: tuple[tuple[float]] | list[list[float]], sensor=Sensor.THERMAL, scale=15, image_mode=False
):
    """Returns the Visual representation of the thermal matrix given the JSON list. The function can also be used to re-scale an existing image from (32 * oldscale x 24 * oldscale) to (32 * scale x 24 * scale)"""
    required_shape = shapes[sensor]
    if isinstance(matrix, tuple) or isinstance(matrix, list):
        match sensor:
            case Sensor.THERMAL:
                if image_mode:
                    matrix_copy = np.array(matrix, dtype=np.float32)
                    matrix_copy =  20 * ((matrix_copy - np.min(matrix_copy)) / (np.max(matrix_copy) - np.min(matrix_copy))) if (np.max(matrix_copy) != np.min(matrix_copy)) else 27 * np.ones_like(matrix_copy)
                    matrix_copy = [
                        [temperature_to_BGR(pix) for pix in row] for row in matrix_copy
                    ]
                else:
                    matrix_copy = [
                        [temperature_to_BGR(pix) for pix in row] for row in matrix
                    ]
            case Sensor.TOF:
                matrix_copy = [[distance_to_BGR(pix) for pix in row] for row in matrix]
            case Sensor.CAMERA:
                matrix_copy = [[pix[::-1] for pix in row] for row in matrix]

        matrix_copy = np.array(
            matrix_copy, dtype=np.uint8
        )  # Convert to BGR numpy array
        matrix_copy = cv2.resize(
            matrix_copy, (matrix_copy.shape[1] * scale, matrix_copy.shape[0] * scale)
        )  # 1 is width, 0 is height
        #print(matrix_copy[0])
    else:
        if (matrix.shape[1] % required_shape[1] != 0) or (
            matrix.shape[0] % required_shape[0] != 0
        ):
            raise ValueError(
                "Matrix shape must be a multiple of {}x{}",
                required_shape[1],
                required_shape[0],
            )
        if matrix.shape[2] != 3:
            raise ValueError("Matrix must be colored")
        matrix_copy = matrix.copy()
        matrix_copy = cv2.resize(
            matrix_copy, (matrix_copy.shape[1] * scale, matrix_copy.shape[0] * scale)
        )
    return matrix_copy


def load_matrix_from_json(filename: str) -> list[int, list[list[float]]]:
    """Loads the EEPROM/Subpage data from a JSON file"""
    try:
        with open(filename, "r") as f:
            data: list = json.load(
                f
            )  # data is a single frame 
            if data[-1] == []:
                data.pop()
            return data
    except FileNotFoundError:
        raise FileNotFoundError("File not found")


def load_eeprom_from_json(eeprom_path: str) -> list[int]:
    """Load EEPROM list (832 uint16) from the JSON log (first entry)."""
    data = load_matrix_from_json(eeprom_path)
    if not data:
        raise ValueError("EEPROM JSON has no entries")
    first = data[0]
    # Expected format: [timestamp, [832 ints]]
    if not isinstance(first, list) or len(first) < 2:
        raise ValueError("Invalid EEPROM JSON format")
    eeprom_list = first[1]
    if not isinstance(eeprom_list, (list, tuple)) or len(eeprom_list) != 832:
        raise ValueError("EEPROM data must contain 832 uint16 values")
    return list(int(x) for x in eeprom_list)



def load_thermal_subpages(filename: str) -> list[tuple[int, int, list[int]]]:
    """Load thermal subpages as tuples: (timestamp, [832 uint16])."""
    raw = load_matrix_from_json(filename)
    frames: list[tuple[int, list[int]]] = []
    for entry in raw:
        # Expected format: [timestamp, [subpage_number, [832 ints]]]
        ts = int(entry[0])
        words = entry[1]
        frames.append((ts, words))
    return frames


def build_temperature_frames_single_subpage(subpages: list[tuple[int, int, list[int]]], eeprom: list[int], image_mode: bool = False) -> list[tuple[int, list[list[float]]]]:
    """Build frames incrementally from single subpages.

    For each incoming subpage, compute temperatures only for pixels belonging to that subpage
    and update a rolling 24x32 frame; emit one frame per subpage.
    """
    params = mlx90640_parameters()
    mlx_parse_eeprom_parameters(params, eeprom)

    pixel_fn = get_pixel_in_image_mode if image_mode else get_temperature_of_pixel

    current = [
        [0 for _ in range(MLX90640_RES_COLS)] for _ in range(MLX90640_RES_ROWS)
    ]
    out: list[tuple[int, list[list[float]]]] = []

    for ts, words in subpages:
        fp0 = mlx_frame_parameters()
        fp1 = mlx_frame_parameters()
        control_reg_value = 0x1901
        
        mlx_parse_frame_parameters(words, words, params, fp0, fp1, control_reg_value)
        
        # Sensor sends the entire subpage now
        for idx in range(MLX90640_RES_ROWS * MLX90640_RES_COLS):
            try:
                frame_params = fp0 if (SUBPAGE_NUMBER(idx) == 0) else fp1
                t = pixel_fn(params, frame_params, words, idx)
            except Exception as e:
                t = 1000.0
                print("Error computing temperature : ", e)
                print("Using Default Value of 1000.0")
            r = idx // MLX90640_RES_COLS
            c = idx % MLX90640_RES_COLS
            current[r][MLX90640_RES_COLS-1-c] = t

        out.append((ts, copy.deepcopy(current)))

    return out


def write_video(
    outname: str, frames: list[int, list[list[float]]], sensor: Sensor, image_mode: bool = False
) -> None:
    """Writes the frames to a video file"""
    if len(frames) == 0:
        raise ValueError("No frames to write")
    required_shape = shapes[sensor]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        outname,
        fourcc,
        FRAMERATE,
        (required_shape[1] * IMAGE_SCALE, required_shape[0] * IMAGE_SCALE),
        isColor=True,
    )
    timestamp_zero = frames[0][0]
    for timestamp, framedata in frames:
        framedata = visualize_data(
            framedata, sensor, scale=IMAGE_SCALE, image_mode=image_mode
        )  # Convert the frame to a visual representation
        adjusted_timestamp = (timestamp - timestamp_zero) / 1000
        cv2.putText(
            framedata,
            f"{float(adjusted_timestamp):.2f} s",
            (10, 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        out.write(framedata)
    out.release()


def main():
    parser = argparse.ArgumentParser(description="Visualize sensor data")
    parser.add_argument(
        "filename",
        type=str,
        help="The filename of the JSON file containing the sensor data",
    )
    parser.add_argument(
        "--eeprom",
        type=str,
        default=None,
        help="Path to thermal_eeprom.json (required for --thermal when using raw subpages)",
    )
    parser.add_argument(
        "--scale", type=int, help="The scale of the image", default=IMAGE_SCALE
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="The video output filename",
        default="output.mp4",
    )
    parser.add_argument("--thermal", action="store_true", help="Visualize thermal data")
    parser.add_argument("--tof", action="store_true", help="Visualize ToF data")
    parser.add_argument("--cam", action="store_true", help="Visualize camera data")
    parser.add_argument(
        "--image-mode",
        action="store_true",
        help="Use image mode (get_pixel_in_image_mode) instead of temperature computation",
    )
    args = parser.parse_args()
    if not (args.thermal or args.tof or args.cam):
        raise ValueError("At least one of --thermal, --cam, or --tof must be specified")

    sensor = None
    match (args.thermal, args.tof, args.cam):
        case (True, False, False):
            sensor = Sensor.THERMAL
        case (False, True, False):
            sensor = Sensor.TOF
        case (False, False, True):
            sensor = Sensor.CAMERA
        case _:
            raise ValueError("Only one sensor type can be specified")

    if sensor == Sensor.THERMAL:
        # Determine EEPROM path
        eeprom_path = args.eeprom
        if eeprom_path is None:
            # Try sibling file named thermal_eeprom.json in the same directory as filename
            base_dir = os.path.dirname(os.path.abspath(args.filename))
            candidate = os.path.join(base_dir, "thermal_eeprom.json")
            eeprom_path = candidate if os.path.exists(candidate) else None
        if eeprom_path is None:
            raise ValueError("EEPROM file not specified and not found next to frames")

        eeprom_list = load_eeprom_from_json(eeprom_path)
        subpages = load_thermal_subpages(args.filename)
        frames = build_temperature_frames_single_subpage(subpages, eeprom_list, image_mode=args.image_mode)
        write_video(args.output, frames, sensor, image_mode=args.image_mode)
        #print(frames)
    else:
        data = load_matrix_from_json(args.filename)
        write_video(args.output, data, sensor)

if __name__ == "__main__":
    main()
