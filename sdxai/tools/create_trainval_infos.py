"""Build nuscenes_infos_{train,val}.pkl for v1.0-trainval with the upstream converter, without LiDAR files.

Only camera keyframes (samples/CAM_*) are on disk. The upstream converter requires the keyframe
LiDAR file to exist (mmcv.check_file_exist) although inference never loads points; this wrapper
relaxes that single check for LIDAR_TOP paths and runs only the trainval part (the upstream
`--version v1.0` branch would also build test-split infos, which are not needed).
"""
import os
import sys

import mmcv

_orig = mmcv.check_file_exist


def _check(path, msg=""):
    if "LIDAR_TOP" in str(path):
        return
    return _orig(path, msg)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "./data/infos/"
    sys.argv = ["nuscenes_converter.py", "nuscenes", "--root-path", "./data/nuscenes", "--canbus", "./data/nuscenes",
                "--out-dir", out, "--extra-tag", "nuscenes", "--version", "v1.0"]
    mmcv.check_file_exist = _check
    sys.path.insert(0, "tools/data_converter")
    import nuscenes_converter as nc  # parses sys.argv at import
    nc.mmcv.check_file_exist = _check
    nc.nuscenes_data_prep(root_path="./data/nuscenes", can_bus_root_path="./data/nuscenes", info_prefix="nuscenes",
                          version="v1.0-trainval", dataset_name="NuScenesDataset", out_dir=out, max_sweeps=10)


if __name__ == "__main__":
    main()
