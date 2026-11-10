"""asos_fetch.py OUT_CSV START END: every CONUS ASOS routine report (T, Td, wind) from IEM, 48 state networks in parallel."""
import csv, io, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
ST = ("AL AZ AR CA CO CT DE FL GA ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA "
      "RI SC SD TN TX UT VT VA WA WV WI WY").split()
out, a, b = sys.argv[1], sys.argv[2], sys.argv[3]   # YYYY-MM-DDTHH


def get(s):
    y1, m1, d1, h1 = a[:4], a[5:7], a[8:10], a[11:13]
    y2, m2, d2, h2 = b[:4], b[5:7], b[8:10], b[11:13]
    u = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?network=%s_ASOS&data=tmpf&data=dwpf&data=sknt"
         "&data=drct&year1=%s&month1=%s&day1=%s&hour1=%s&year2=%s&month2=%s&day2=%s&hour2=%s&tz=Etc/UTC"
         "&format=onlycomma&latlon=yes&missing=M&trace=T&direct=no&report_type=3" % (s, y1, m1, d1, h1, y2, m2, d2, h2))
    import time
    for k in range(6):
        try:
            return urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "curl/8"}), timeout=180).read().decode()
        except Exception as e:
            err = e; time.sleep(15 * (k + 1))
    raise err


with ThreadPoolExecutor(2) as ex:
    texts = list(ex.map(get, ST))
n = 0
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["station", "valid", "lon", "lat", "tmpf", "dwpf", "sknt", "drct"])
    for t in texts:
        r = csv.DictReader(io.StringIO(t))
        for row in r:
            w.writerow([row["station"], row["valid"], row["lon"], row["lat"], row["tmpf"], row["dwpf"], row["sknt"], row["drct"]])
            n += 1
print(out, n, "reports")
