#!/bin/bash

set -euo pipefail

export output_dir="./addons/mujoco_model_exporter"
rm -rf "${output_dir}"
mkdir -p "${output_dir}"

cp ./*.py "${output_dir}"
find "${output_dir}" -name "*.py" -exec sed -i "s/from dm_control.blender.fake_core //g" "{}" +;
find "${output_dir}" -name "*.py" -exec sed -i "s/from dm_control.blender.mujoco_exporter/from ./g" "{}" +;
echo "Add-on exported to ${output_dir}."
echo "Copy this to Blender ./scripts/addons - see https://docs.blender.org/manual/en/latest/advanced/blender_directory_layout.html"
