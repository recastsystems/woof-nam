"""The fetch's stall cap (fetch_case._stall_s): which hosts answer fast enough to call a silent request stalled."""
import os
from unittest import mock

from namgpu.init import fetch_case as fc
from namgpu.init.sources import SOURCES


def test_every_s3_name_of_a_bucket_takes_the_stall_cap():
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("RECAST_NAM_FETCH_STALL_S", None)
        assert fc._stall_s("https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20261114/00/x", 120) == 20
        # the ECMWF open-data bucket is read by its regional name: it kept the caller's 300 s
        assert fc._stall_s("https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/20261114/06z/x.grib2", 300) == 20
        assert fc._stall_s("https://some-bucket.s3-us-west-2.amazonaws.com/x", 300) == 20
        assert fc._stall_s("https://some-bucket.s3.dualstack.us-east-2.amazonaws.com/x", 300) == 20
        # every S3 host a source row names for its ECMWF route is covered
        for name in ("ifs", "ifs-0p25", "aifs"):
            for host in SOURCES[name]["fetch"]["hosts"]:
                if "amazonaws.com" in host:
                    assert fc._stall_s(host + "/x", 300) == 20, host
        # hosts that can be slow to start keep the caller's wait
        assert fc._stall_s("https://data.ecmwf.int/forecasts/x", 300) == 300
        assert fc._stall_s("https://storage.googleapis.com/ecmwf-open-data/x", 300) == 300
        assert fc._stall_s("https://www.ncei.noaa.gov/x", 600) == 600
        assert fc._stall_s("https://not-s3.amazonaws.com.example.org/x", 300) == 300
        # a caller's shorter wait is kept, and the cap can be turned off
        assert fc._stall_s("https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/x", 5) == 5
    with mock.patch.dict(os.environ, {"RECAST_NAM_FETCH_STALL_S": "0"}):
        assert fc._stall_s("https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/x", 300) == 300
