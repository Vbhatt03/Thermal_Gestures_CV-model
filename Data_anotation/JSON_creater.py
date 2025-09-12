import json
import bisect

# Load full dataset
with open("C:\\PARAS_SOP_DC\\micro-gestures\\data_2\\Thermal_third_Vyomesh\\thermal.json", "r") as f:
    data = json.load(f)

# Extract all timestamps for fast lookup
timestamps = [entry[0] for entry in data]

print("Total frames:", len(timestamps))
print("Timestamp range:", timestamps[0], "to", timestamps[-1])

# Input loop
while True:
    label = input("\nEnter class label (e.g. A_1), or 'exit' to stop: ").strip()
    if label.lower() == "exit":
        break

    try:
        raw_start = int(input(f"Enter START timestamp for {label}: ")) +12499
        raw_end = int(input(f"Enter END timestamp for {label}: "))+ 12499

        # Snap start to nearest timestamp <= raw_start
        start_index = bisect.bisect_right(timestamps, raw_start) - 1
        if start_index < 0:
            print("Start timestamp is before data range.")
            continue
        final_start = timestamps[start_index]   # Adjusted start time

        # Snap end to nearest timestamp >= raw_end
        end_index = bisect.bisect_left(timestamps, raw_end)
        if end_index >= len(timestamps):
            print("End timestamp exceeds data range.")
            continue
        final_end = timestamps[end_index]  # Adjusted start time

        # Extract segment
        segment = [entry for entry in data if final_start <= entry[0] <= final_end]

        if not segment:
            print("No data found in the given range.")
            continue

        # Save to file
        out_path = f"D:\\micro-gestures\\data\\Labelled_Data\\Vyomesh\\Z\\{label}.json"
        with open(out_path, "w") as out_file:
            json.dump(segment, out_file, indent=2)

        print(f"Saved {len(segment)} frames to {out_path}")
        print(f"Range used: {final_start} → {final_end}")

    except Exception as e:
        print("Error:", e)
