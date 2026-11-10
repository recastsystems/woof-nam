"""Score definitions and the actual Rust CSV/NetCDF boundary, without a forecast rerun."""
import csv
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from namgpu.io import namwinter as w


def row(t=35., td=30., obs=35., **extra):
    return dict(latitude=35.8, longitude=-80., model_t2_f=t, obs_t2_f=obs,
                model_td2_f=td, obs_td2_f=30., **extra)


def test_contingency_includes_dry_and_excludes_unknown():
    s = w.ptype_scores([(1, 1), (1, 2), (2, 2), (0, 0), (-1, 1), (3, None)])
    assert s['n'] == 4 and s['missing_pairs'] == 2
    assert s['accuracy'] == .75
    assert s['heidke'] == pytest.approx((3 - 1.25) / (4 - 1.25))
    assert s['per_type']['snow']['hit_rate'] == .5
    assert s['per_type']['rain']['far'] == .5
    assert s['both_wet']['n'] == 3
    assert w.ptype_scores([(0, 0)])['heidke'] is None
    assert w.ptype_scores([])['per_type']['sleet']['hit_rate'] is None
    with pytest.raises(ValueError):
        w.ptype_table(np.full((5, 5), .5))


def test_amounts_missing_is_not_zero():
    s = w.amount_scores([(0., 0.), (3., 2.), (math.nan, 0.), (0., math.nan), (-1., 0.)])
    assert s['n'] == 2 and s['bias'] == .5
    assert s['thresholds']['2.54']['false'] == 1
    assert w.amount_scores([])['n'] == 0


def test_erosion_is_sustained_and_censored():
    s = list(enumerate([1., .4, .8, .3, .2]))
    assert w.erosion(s) == dict(status='resolved', lead=3)
    assert w.erosion(s[:4])['status'] == 'right_censored'
    assert w.erosion([(0, .2), (1, .1)])['status'] == 'not_established'
    assert w.erosion([(0, 1.), (2, .1)])['status'] == 'missing_hours'
    assert w.erosion([(0, None)])['status'] == 'missing_hours'


def test_common_support_fixed_panel_and_observation_consistency():
    a = {(h, 'GSO'): row(t=35. if h == 0 else 45., obs=35. if h == 0 else 45.) for h in range(3)}
    a[(0, 'CLT')] = row()
    b = {k: dict(v) for k, v in a.items()}
    b[(1, 'GSO')]['model_td2_f'] = math.nan
    s = w.score_pairs(dict(control=a, option=b), 2)
    assert s['control']['coverage']['erosion_stations'] == ['GSO']
    assert s['control']['erosion']['model']['lead'] == 1
    assert s['control']['hourly']['1']['td2_f']['n'] == 0
    assert s['control']['ptype']['n'] == 0
    assert s['control']['erosion']['error_hours'] == 0
    b[(0, 'GSO')]['obs_t2_f'] = 30.
    with pytest.raises(ValueError, match='disagree on observation'):
        w.score_pairs(dict(control=a, option=b), 2)


def write_pairs(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['cycle', 'lead', 'station', *next(iter(rows.values()))])
        writer.writeheader()
        for (h, station), r in rows.items():
            writer.writerow(dict(cycle='2026012312', lead=h, station=station, **r))


def test_pair_validation(tmp_path):
    path = tmp_path / 'pairs.csv'
    write_pairs(path, {(0, 'GSO'): row()})
    assert len(w.read_pairs(path, '2026012312', 2)) == 1
    with pytest.raises(ValueError, match='cycle'):
        w.read_pairs(path, '2026012412', 2)
    with path.open('a') as f:
        f.write(path.read_text().splitlines()[-1] + '\n')
    with pytest.raises(ValueError, match='duplicate'):
        w.read_pairs(path, '2026012312', 2)
    path.write_text(path.read_text().splitlines()[0] + '\n2026012312\n')
    with pytest.raises(ValueError):
        w.read_pairs(path, '2026012312', 2)


def test_exact_report_windows_and_units(tmp_path):
    p = tmp_path / 'reports.csv'
    header = 'cycle,kind,units,source,station,start_lead,end_lead,amount\n'
    p.write_text(header + '2026012312,snowfall,mm,analysis,GSO,0,2,20\n')
    rows = {(h, 'GSO'): row(model_snow_proxy_1h_mm=10.) for h in (1, 2)}
    s = w.score_reports(p, rows, '2026012312', 2)
    assert s['snowfall']['n'] == 1 and s['snowfall']['rmse'] == 0.
    del rows[(1, 'GSO')]
    assert w.score_reports(p, rows, '2026012312', 2)['snowfall']['n'] == 0
    p.write_text(header + '2026012312,ice_thickness,mm,report,GSO,0,2,2\n')
    with pytest.raises(ValueError, match='reports require'):
        w.score_reports(p, rows, '2026012312', 2)


def test_score_door_receipts(tmp_path):
    p = tmp_path / 'pairs.csv'
    write_pairs(p, {(h, 'GSO'): row() for h in range(3)})
    out = tmp_path / 'scores'
    command = [sys.executable, '-m', 'namgpu.cli', 'score', '--winter', '--cycle', '2026012312',
               '--hours', '2', '--pairs=control=' + str(p), '--out', str(out), '--no-sheets']
    r = subprocess.run(command, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    s = json.loads((out / 'winter_scores.json').read_text())
    assert s['runs']['control']['hourly']['1']['t2_f']['n'] == 1
    assert s['runs']['control']['amounts']['status'] == 'unavailable'


@pytest.fixture
def bridge():
    path = os.environ.get('NAMGPU_WINTER_BRIDGE')
    if not path:
        pytest.skip('set NAMGPU_WINTER_BRIDGE to compiled reader; required in CI')
    assert Path(path).is_file()
    return path


@pytest.fixture
def history(tmp_path):
    import netCDF4
    run = tmp_path / 'run'
    run.mkdir()
    for h in range(3):
        with netCDF4.Dataset(run / f'wrfout_d01_2026-01-23_{12+h:02}:00:00', 'w') as nc:
            for dim, n in [('Time', 1), ('DateStrLen', 19), ('y', 1), ('x', 2), ('z', 5)]:
                nc.createDimension(dim, n)
            nc.createVariable('Times', 'S1', ('Time', 'DateStrLen'))[:] = np.array(list(f'2026-01-23_{12+h:02}:00:00'), 'S1')[None, :]
            fields = dict(XLAT=[35.8, 35.8], XLONG=[-80., -80.1], T2=[272., 274.],
                          Q2=[.003, .004], PSFC=[100000., 100000.], RAINNC=[h, 2*h],
                          RAINC=[0., 0.], SR=[.1, .8], PREC=[.00001, .00001], PTYPE=[3., 2.])
            for k, values in fields.items():
                v = nc.createVariable(k, 'f4', ('Time', 'y', 'x'), fill_value=-9999.)
                v[:] = np.array(values)[None, None, :]
            pressure = np.array([99000., 90000., 75000., 50000., 40000.])
            theta = np.array([272., 278., 275., 255., 240.]) / (pressure/1e5)**.28571 - 300.
            for k, values in [('PRES', pressure), ('T', theta)]:
                nc.createVariable(k, 'f4', ('Time', 'z', 'y', 'x'))[:] = values[None, :, None, None]
    obs = tmp_path / 'asos.csv'
    obs.write_text('station,valid,lat,lon,tmpf,dwpf,wxcodes,p01i\n' + ''.join(
        f'GSO,2026-01-23 {12+h:02}:00,35.8,-80,30,28,FZRA,0.04\n'
        f'INT,2026-01-23 {12+h:02}:00,35.8,-80.1,32,30,SN,M\n'
        for h in range(3)))
    return run, obs


def sample(bridge, history, method='stored', valid='2026-01-23T13:00:00Z'):
    run, obs = history
    return subprocess.run([bridge, 'sample', str(run / 'wrfout_d01_2026-01-23_13:00:00'),
                           str(run / 'wrfout_d01_2026-01-23_12:00:00'), str(obs), valid, '1', method, '15'],
                          capture_output=True, text=True)


def test_rust_samples_csv_netcdf_units_and_types(bridge, history):
    r = sample(bridge, history)
    assert r.returncode == 0, r.stderr
    rows = list(csv.DictReader(io.StringIO(r.stdout)))
    assert [r['station'] for r in rows] == ['GSO', 'INT']
    assert float(rows[0]['model_precip_1h_mm']) == 1.
    assert float(rows[0]['obs_precip_1h_mm']) == pytest.approx(1.016)
    assert float(rows[0]['model_t2_f']) == pytest.approx(29.93)
    assert [float(r['model_type']) for r in rows] == [3., 2.]
    assert rows[1]['obs_precip_1h_mm'] == ''
    r = sample(bridge, history, method='explicit-sr')
    assert r.returncode == 0, r.stderr
    rows = list(csv.DictReader(io.StringIO(r.stdout)))
    assert [float(r['model_type']) for r in rows] == [3., 2.]


def test_rust_rejects_time_mismatch_reset_and_masks_fill(bridge, history):
    import netCDF4
    assert 'Times differs' in sample(bridge, history, valid='2026-01-23T14:00:00Z').stderr
    run, _ = history
    with netCDF4.Dataset(run / 'wrfout_d01_2026-01-23_13:00:00', 'a') as nc:
        nc['T2'][0, 0, 0] = -9999.
    rows = list(csv.DictReader(io.StringIO(sample(bridge, history).stdout)))
    assert rows[0]['model_t2_f'] == ''
    with netCDF4.Dataset(run / 'wrfout_d01_2026-01-23_13:00:00', 'a') as nc:
        nc['RAINNC'][0, 0, 0] = -1.
    r = sample(bridge, history)
    assert r.returncode == 2 and 'accumulation reset' in r.stderr


def test_rust_history_score_door_is_read_only(bridge, history, tmp_path):
    import hashlib
    import shutil
    run, obs = history
    before_dir = tmp_path / 'before'
    shutil.copytree(run, before_dir)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.iterdir()}
    assert w.main(['--cycle', '2026012312', '--hours', '2', '--run', 'control=' + str(run),
                   '--asos', str(obs), '--ptype-method', 'stored', '--out', str(tmp_path / 'score'), '--no-sheets']) == 0
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.iterdir()}
    assert before == after
    identity = subprocess.run([sys.executable, '-m', 'namgpu.cli', 'compare', str(before_dir), str(run),
                               '--workers', '1', '--json', str(tmp_path / 'identity.json')], capture_output=True, text=True)
    assert identity.returncode == 0, identity.stdout + identity.stderr
    assert json.loads((tmp_path / 'identity.json').read_text())['bitwise'] is True
    s = json.loads((tmp_path / 'score/winter_scores.json').read_text())['runs']['control']
    assert s['ptype']['n'] == 4 and s['ptype']['heidke'] == 1.
    assert s['fzra_reporting_hours']['n'] == 2
