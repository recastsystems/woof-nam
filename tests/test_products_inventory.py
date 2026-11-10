"""Regression: an unmapped record must not disappear from the inventory gate."""
from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from namgpu.products import inventory as inv, spec


def section(number, body):
    return struct.pack(">IB", len(body) + 5, number) + body


def product(number=0, level=50000, second=None, stat=None, start=1, duration=0):
    s = bytearray(58 if stat is not None else 34)
    s[:5] = struct.pack(">IB", len(s), 4)
    s[7:9] = struct.pack(">H", 8 if stat is not None else 0)
    s[9:11] = bytes((0, number))
    s[17] = 1
    s[18:22] = struct.pack(">I", start)
    s[22:28] = bytes((100, 0)) + struct.pack(">I", level)
    s[28:34] = bytes((255 if second is None else 100, 0)) + struct.pack(">I", second or 0)
    if stat is not None:
        s[41], s[46], s[48] = 1, stat, 1
        s[49:53] = struct.pack(">I", duration)
    return bytes(s)


def message(*products, payload=b"value"):
    body = section(1, bytes(16)) + section(3, bytes(10))
    for p in products:
        body += p + section(5, bytes(16)) + section(6, bytes((255,))) + section(7, payload)
    return b"GRIB\0\0\0\2" + struct.pack(">Q", len(body) + 20) + body + b"7777"


def scan(tmp_path, *products, payload=b"value"):
    f = tmp_path / "test.grib2"
    f.write_bytes(message(*products, payload=payload))
    return inv.scan(f, hashes=True)


def row(r, source="-", note="LEFT OUT: source diagnostic is not retained"):
    return dict(r, discipline=0, category=0, number=int(r["dcp"].split(".")[-1]),
                stat=spec.STAT_NAMES.get(r["stat"], "-"), source=source, note=note,
                name="TMP", level=f"100:{r['v1']}", line=2)


def test_unmapped_and_unknown_archive_rows_are_never_skipped(tmp_path):
    records = scan(tmp_path, product(), product(number=1), product(number=2))
    result = inv.compare(records, records[:1], [row(records[0], "carrier:T"), row(records[1])])
    assert [r["status"] for r in result["records"]] == ["written", "left_out", "unexplained"]
    assert result["reference_records"] == 3 and not result["ok"]
    assert result["records"][2]["reason"] == "record absent from package table"


def test_levels_windows_and_duplicate_occurrences_are_distinct(tmp_path):
    records = scan(tmp_path, product(level=50000), product(level=85000),
                   product(stat=1, start=0, duration=1), product(stat=1, start=1, duration=1))
    assert len({inv.identity(r) for r in records}) == 4
    result = inv.compare(records + records[:1], list(reversed(records)), [row(r, "carrier:T") for r in records])
    assert result["written"] == 4 and result["unexplained"] == 1
    assert not result["ok"]


def test_mapped_missing_or_blank_reason_fails(tmp_path):
    records = scan(tmp_path, product())
    for r in (row(records[0], "carrier:T"), row(records[0], note="")):
        assert inv.compare(records, [], [r])["unexplained"] == 1


def test_unmapped_sibling_cannot_hide_mapped_record(tmp_path):
    records = scan(tmp_path, product(stat=1, start=0, duration=1))
    rows = [row(records[0]), row(records[0], "carrier:OTHER_WINDOW")]
    assert inv.compare(records, [], rows)["unexplained"] == 1


def test_two_field_messages_and_byte_preservation(tmp_path):
    records = scan(tmp_path, product(), product(number=1))
    assert [r["rec"] for r in records] == ["1.1", "1.2"]
    assert records[0]["message_sha256"] == records[1]["message_sha256"]
    added = scan(tmp_path, product(number=2))
    assert inv.preserved(records, added + records)["ok"]
    assert not inv.preserved(records, records[::-1])["ok"]
    changed = scan(tmp_path, product(), product(number=1), payload=b"other")
    assert inv.preserved(records, changed)["changed_or_missing"] == ["1.1", "1.2"]


@pytest.mark.parametrize("data", [b"", b"GRIB", message(product())[:-1],
                                  message(product()).replace(b"7777", b"bad!")])
def test_bad_file_is_refused(tmp_path, data):
    f = tmp_path / "broken.grib2"
    f.write_bytes(data)
    with pytest.raises(ValueError):
        inv.scan(f)


def test_zero_length_section_is_refused(tmp_path):
    data = bytearray(message(product()))
    data[16:20] = bytes(4)
    f = tmp_path / "broken.grib2"
    f.write_bytes(data)
    with pytest.raises(ValueError, match="invalid section length 0"):
        inv.scan(f)


def test_extra_record_and_layer_bounds(tmp_path):
    records = scan(tmp_path, product(second=60000), product(second=70000))
    assert inv.identity(records[0]) != inv.identity(records[1])
    result = inv.compare(records[:1], records, [row(records[0])])
    assert not result["ok"] and len(result["extra"]) == 1


@pytest.mark.parametrize("product_set", ("awphys", "conusnest", "alaskanest", "hawaiinest", "priconest", "firewxnest"))
def test_every_real_archive_record_is_in_the_table(product_set):
    """NOAA 2026-11-03 06Z f01 headers, including every level and both U/V fields.

    This catches newly archived parameters and levels even if they have no source
    mapping. The old products compare filtered these rows out before checking.
    """
    import json
    from namgpu.products import sets, spec
    fixture = Path(__file__).parent / "fixtures/nam-20261103-06-f01-inventory.json"
    archived = json.loads(fixture.read_text())[product_set]["records"]
    ps = sets.get(product_set)
    rows = spec.read(sets.table_path(ps))
    keys = {spec.inventory_key(r) for r in rows}
    assert all(inv.table_key(r) in keys for r in archived)
    # Treat mapped records as available to isolate the omission policy. Actual
    # writer outputs are checked separately by the CLI; this is table coverage.
    mapped = {spec.inventory_key(r) for r in rows if r["source"] != "-"}
    present = [r for r in archived if inv.table_key(r) in mapped]
    report = inv.compare(archived, present, rows)
    assert report["ok"]
    assert report["written"] + report["left_out"] == len(archived)
    missing = [r for r in archived if inv.table_key(r) not in mapped]
    assert missing  # the case must exercise omitted records
    new_level = dict(missing[0], t1=103, v1=9999)
    assert not inv.compare([*archived, new_level], present, rows)["ok"]


def test_incomplete_field_cannot_be_overwritten_by_next_product(tmp_path):
    with pytest.raises(ValueError, match="before previous field's data"):
        scan(tmp_path, product() + product(number=1))


def test_new_mappings_refuse_an_old_writer_before_reading_history(monkeypatch, tmp_path, capsys):
    from namgpu import exporter
    from namgpu.io import products
    monkeypatch.setattr(exporter, "ensure", lambda: tmp_path / "old-writer")
    monkeypatch.setattr(exporter, "capabilities", lambda path=None: {"pair_next", "mask_carrier"})
    assert products.main(["--run", str(tmp_path), "--set", "conusnest"]) == 2
    error = capsys.readouterr().err
    assert "nam_inventory_1" in error and "cloud-top temperature and LCL pressure" in error
    assert len(error.splitlines()) == 1
