import glob
import os
import numpy as np
import pandas as pd

# --- 1. DYNAMIC SURVEY PARAMETERS ---
print("--- STEP 1: SURVEY CONFIGURATION ---")
print("Select Road Type: 1. Highway, 2. Major District Road (MDR), 3. Rural Road, 4. Urban Road")
choice = input("Enter selection (1-4): ").strip()
road_map_choice = {"1": "Highway", "2": "MDR", "3": "Rural", "4": "Urban"}
ROAD_TYPE = road_map_choice.get(choice, "Highway")

# Real-world dimensions for the surveyed road segment
L_PER_IMAGE = float(input("Enter Road Length captured per image (in meters): "))
W_ROAD = float(input("Enter Lane Width (in meters, e.g., 3.5): "))
IMAGE_WIDTH_PX = 1280  # Standard reference width for geometric scaling

# --- 2. IRC:82:2023 CLASS MAPPING ---
class_map = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Repaired Crack",
    4: "Pothole",
    5: "Patching",
    6: "Rutting",
}
CRACK_TYPES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Repaired Crack",
]


# --- 3. RUT DEPTH CALCULATION (Geometric Severity Method) ---
def estimate_rut_depth(w_norm, lane_width_pixels=IMAGE_WIDTH_PX, standard_lane_mm=3500, cf=0.04):
    scale = standard_lane_mm / lane_width_pixels
    pixel_width = w_norm * lane_width_pixels
    metric_width_mm = pixel_width * scale
    depth_mm = metric_width_mm * cf
    return depth_mm


# --- 4. IRC:82:2023 EMPIRICAL PCI FORMULAS ---
def get_pci_values(rtype, ce, pn, rd, pae):
    if rtype == "Highway":
        p_crack = 7231 / ((ce**2) - (0.737 * ce) + 73.09) if ce > 0 else 100.0
        p_pothole = (
            ((-80.32 * (pn**3)) + (1129 * (pn**2)) - (3524 * pn) + 3566)
            / ((pn**3) + (5.791 * (pn**2)) - (28.67 * pn) + 35.72)
            if pn > 0
            else 100.0
        )
        p_rut = 7231 / ((rd**2) - (0.737 * rd) + 73.09) if rd > 0 else 100.0
        p_patch = (
            (52.92 * np.exp(-0.02525 * pae)) + (44.1 * np.exp(-0.2899 * pae))
            if pae > 0
            else 100.0
        )

    elif rtype in ["MDR", "Rural"]:
        func_mdr = lambda x: (
            (-0.5662 * (x**3)) + (85.14 * (x**2)) - (3377 * x) + 51820
        ) / ((x**2) - (29.31 * x) + 523.4)
        p_crack = func_mdr(ce) if ce > 0 else 100.0
        p_rut = func_mdr(rd) if rd > 0 else 100.0
        p_pothole = (
            (0.0113 * (pn**4)) - (0.2825 * (pn**3)) + (2.6246 * (pn**2)) + 99.529
            if pn > 0
            else 100.0
        )
        p_patch = (
            ((-0.001723 * (pae**3)) - (0.08273 * (pae**2)) + (21.39 * pae) + 419.1)
            / (pae + 4.267)
            if pae > 0
            else 100.0
        )

    elif rtype == "Urban":
        p_crack = (100 * np.exp(-0.0534 * ce)) - (0.006641 * ce) if ce > 0 else 100.0
        p_pothole = (
            ((-130.6 * (pn**3)) + (2112 * (pn**2)) - (9550 * pn) + 14390)
            / ((pn**3) + (6.619 * (pn**2)) - (66.73 * pn) + 144.1)
            if pn > 0
            else 100.0
        )
        p_rut = 7231 / ((rd**2) - (0.737 * rd) + 73.09) if rd > 0 else 100.0
        p_patch = 7231 / ((pae**2) - (0.737 * pae) + 73.09) if pae > 0 else 100.0

    else:
        raise ValueError(f"Unsupported road type: {rtype}")

    return p_crack, p_pothole, p_rut, p_patch


def get_condition_rating(pci):
    if pci >= 90:
        return "Excellent"
    elif pci >= 80:
        return "Good"
    elif pci >= 60:
        return "Satisfactory"
    elif pci >= 40:
        return "Fair"
    elif pci >= 20:
        return "Poor"
    return "Failed"


# --- 5. DATA EXTRACTION & PCI COMPUTATION ---
def main():
    # Update this path to the folder containing your YOLO detection .txt label files
    label_folder_path = "./labels"
    files = glob.glob(os.path.join(label_folder_path, "**", "*.txt"), recursive=True)

    num_files = len(files)
    if num_files == 0:
        print(f"\n[!] Error: No .txt label files found in '{label_folder_path}'.")
        return

    total_area = num_files * L_PER_IMAGE * W_ROAD
    all_detections = []
    rut_depth_samples = []

    for file_path in files:
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 5:
                        c_id = int(parts[0])
                        w_norm = float(parts[3])
                        h_norm = float(parts[4])

                        area_m2 = (w_norm * W_ROAD) * (h_norm * L_PER_IMAGE)
                        cat = class_map.get(c_id, "Unknown")

                        all_detections.append({"category": cat, "area": area_m2})
                        if cat == "Rutting":
                            rut_depth_samples.append(estimate_rut_depth(w_norm))

    df = pd.DataFrame(all_detections)

    # Calculate distress metrics
    total_crack_area = df[df["category"].isin(CRACK_TYPES)]["area"].sum() if not df.empty else 0.0
    total_pothole_area = df[df["category"] == "Pothole"]["area"].sum() if not df.empty else 0.0
    total_patch_area = df[df["category"] == "Patching"]["area"].sum() if not df.empty else 0.0

    # IRC:82:2023 metrics
    ce = (total_crack_area / total_area) * 100
    pn = total_pothole_area / 0.1
    pae = (total_patch_area / total_area) * 100
    rd = float(np.mean(rut_depth_samples)) if rut_depth_samples else 0.0

    # Individual and weighted overall PCI
    p_c, p_p, p_r, p_pa = get_pci_values(ROAD_TYPE, ce, pn, rd, pae)
    final_pci = (0.24 * p_c) + (0.32 * p_p) + (0.28 * p_r) + (0.16 * p_pa)
    condition = get_condition_rating(final_pci)

    # --- 6. FINAL DETAILED ANALYSIS REPORT ---
    print("\n==========================================")
    print(" IRC:82:2023 PAVEMENT CONDITION REPORT")
    print("==========================================")
    print("ROAD CONTEXT:")
    print(f" - Road Type: {ROAD_TYPE}")
    print(f" - Input Parameters: {L_PER_IMAGE}m per image | {W_ROAD}m width")
    print(f" - Total Images Scanned: {num_files}")
    print(f" - Total Survey Length: {num_files * L_PER_IMAGE:.2f} m")
    print(f" - Total Survey Area: {total_area:.2f} m²")
    print("\nDYNAMICALLY CALCULATED METRICS:")
    print(f" - Total Crack Area: {total_crack_area:.4f} m²")
    print(f" - Crack Extent (CE): {ce:.4f}%")
    print(f" - Total Pothole Area: {total_pothole_area:.4f} m²")
    print(f" - Pothole Number (PN): {pn:.4f}")
    print(f" - Average Rut Depth (RD): {rd:.2f} mm")
    print(f" - Total Patch Area: {total_patch_area:.4f} m²")
    print(f" - Patch Extent (PaE): {pae:.4f}%")
    print("\nINDIVIDUAL PCI SCORES:")
    print(f" - PCI (Cracking): {p_c:.2f}")
    print(f" - PCI (Potholes): {p_p:.2f}")
    print(f" - PCI (Rutting):  {p_r:.2f}")
    print(f" - PCI (Patching): {p_pa:.2f}")
    print("\n------------------------------------------")
    print(f" FINAL WEIGHTED TOTAL PCI: {final_pci:.2f} ({condition})")
    print("------------------------------------------")


if __name__ == "__main__":
    main()
