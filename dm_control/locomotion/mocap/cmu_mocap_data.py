# Copyright 2020 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""File loader for DeepMind-preprocessed version of the CMU motion capture data.

The raw CMU point-cloud data is fitted onto the `dm_control.locomotion` CMU
Humanoid walker model and re-exported as an HDF5 file
(https://www.hdfgroup.org/solutions/hdf5/) that can be read by the
`dm_control.locomotion.mocap` package.

The original database is produced and hosted by Carnegie Mellon University at
http://mocap.cs.cmu.edu/, and may be copied, modified, or redistributed without
explicit permission (see http://mocap.cs.cmu.edu/faqs.php).
"""

import hashlib
import os

import requests
import tqdm

H5_FILENAME = 'cmu_2020_dfe3e9e0.h5'
H5_PATHS = (
    os.path.join(os.path.dirname(__file__), H5_FILENAME),
    os.path.join('~/.dm_control', H5_FILENAME),
)
H5_URL = 'https://storage.googleapis.com/dm_control/cmu_2020_dfe3e9e0.h5'

H5_BYTES = 476559420
H5_SHA256 = 'dfe3e9e0b08d32960bdafbf89e541339ca8908a9a5e7f4a2c986362890d72863'


def _get_cached_file_path():
  """Returns the path to the cached data file if one exists."""
  for path in H5_PATHS:
    expanded_path = os.path.expanduser(path)
    try:
      if os.path.getsize(expanded_path) != H5_BYTES:
        continue
      with open(expanded_path, 'rb'):
        return expanded_path
    except IOError:
      continue
  return None


def _download_and_cache():
  """Downloads CMU data into one of the candidate paths in H5_PATHS."""
  for path in H5_PATHS:
    expanded_path = os.path.expanduser(path)
    try:
      os.makedirs(os.path.dirname(expanded_path), exist_ok=True)
      f = open(expanded_path, 'wb+')
    except IOError:
      continue
    with f:
      try:
        _download_into_file(f)
      except:
        os.unlink(expanded_path)
        raise
    return expanded_path
  raise IOError('cannot open file to write download data into, '
                f'paths attempted: {H5_PATHS}')


def _download_into_file(f, validate_hash=True):
  """Download the CMU data into a file object that has been opened for write."""
  with requests.get(H5_URL, stream=True) as req:
    req.raise_for_status()
    total_bytes = int(req.headers['Content-Length'])
    progress_bar = tqdm.tqdm(
        desc='Downloading CMU mocap data', total=total_bytes,
        unit_scale=True, unit_divisor=1024)
    try:
      for chunk in req.iter_content(chunk_size=102400):
        if chunk:
          f.write(chunk)
        progress_bar.update(len(chunk))
    finally:
      progress_bar.close()

  if validate_hash:
    f.seek(0)
    if hashlib.sha256(f.read()).hexdigest() != H5_SHA256:
      raise RuntimeError('downloaded file is corrupted')


def get_path_for_cmu_2020():
  """Path to mocap data fitted to the 2020 version of the CMU Humanoid model."""
  path = _get_cached_file_path()
  if path is None:
    path = _download_and_cache()
  return path
