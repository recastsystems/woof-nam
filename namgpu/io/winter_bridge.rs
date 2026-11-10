// SPDX-License-Identifier: Apache-2.0
// Winter observation normalization and history sampling. No external Rust crates.
// Build: rustc --edition=2021 -O winter_bridge.rs -o rw_namwinter (libnetcdf required).
use std::{collections::{BTreeMap, HashMap}, env, ffi::{CStr, CString}, fs,
          os::raw::{c_char, c_int}};

#[link(name = "netcdf")]
extern "C" {
    fn nc_open(p: *const c_char, mode: c_int, id: *mut c_int) -> c_int;
    fn nc_close(id: c_int) -> c_int;
    fn nc_strerror(code: c_int) -> *const c_char;
    fn nc_inq_varid(id: c_int, name: *const c_char, var: *mut c_int) -> c_int;
    fn nc_inq_varndims(id: c_int, var: c_int, n: *mut c_int) -> c_int;
    fn nc_inq_vardimid(id: c_int, var: c_int, dims: *mut c_int) -> c_int;
    fn nc_inq_dimlen(id: c_int, dim: c_int, n: *mut usize) -> c_int;
    fn nc_inq_attlen(id: c_int, var: c_int, name: *const c_char, n: *mut usize) -> c_int;
    fn nc_get_att_double(id: c_int, var: c_int, name: *const c_char, v: *mut f64) -> c_int;
    fn nc_get_vara_double(id: c_int, var: c_int, start: *const usize, count: *const usize, v: *mut f64) -> c_int;
    fn nc_get_var_text(id: c_int, var: c_int, v: *mut c_char) -> c_int;
}
type Result<T> = std::result::Result<T, String>;
fn ck(code: i32) -> Result<()> {
    if code == 0 { Ok(()) } else { Err(unsafe { CStr::from_ptr(nc_strerror(code)) }.to_string_lossy().into()) }
}
struct Nc(i32);
impl Drop for Nc { fn drop(&mut self) { unsafe { nc_close(self.0); } } }
impl Nc {
    fn open(path: &str) -> Result<Self> {
        let p = CString::new(path).map_err(|e| e.to_string())?;
        let mut id = 0; ck(unsafe { nc_open(p.as_ptr(), 0, &mut id) })?; Ok(Self(id))
    }
    fn var(&self, name: &str) -> Option<i32> {
        let s = CString::new(name).ok()?; let mut v = 0;
        (unsafe { nc_inq_varid(self.0, s.as_ptr(), &mut v) } == 0).then_some(v)
    }
    fn shape(&self, v: i32) -> Result<Vec<usize>> {
        let mut n = 0; ck(unsafe { nc_inq_varndims(self.0, v, &mut n) })?;
        let mut dims = vec![0; n as usize]; ck(unsafe { nc_inq_vardimid(self.0, v, dims.as_mut_ptr()) })?;
        dims.into_iter().map(|d| { let mut l = 0; ck(unsafe { nc_inq_dimlen(self.0, d, &mut l) })?; Ok(l) }).collect()
    }
    fn att(&self, v: i32, name: &str) -> Option<f64> {
        let s = CString::new(name).ok()?; let mut n = 0; let mut x = 0.;
        if unsafe { nc_inq_attlen(self.0, v, s.as_ptr(), &mut n) } != 0 || n != 1 { return None; }
        (unsafe { nc_get_att_double(self.0, v, s.as_ptr(), &mut x) } == 0).then_some(x)
    }
    fn read(&self, name: &str, column: Option<usize>) -> Result<Option<Vec<f64>>> {
        let Some(v) = self.var(name) else { return Ok(None) };
        let sh = self.shape(v)?;
        if sh.len() < 2 || sh.iter().any(|&n| n == 0) { return Err(format!("{name}: invalid shape {sh:?}")); }
        let mut start = vec![0; sh.len()]; let mut count = sh.clone();
        if sh.len() >= 3 { count[0] = 1; }
        if let Some(i) = column {
            if sh.len() != 4 { return Err(format!("{name}: expected Time,level,y,x")); }
            start[2] = i / sh[3]; start[3] = i % sh[3]; count[2] = 1; count[3] = 1;
        } else if sh.len() > 3 { return Err(format!("{name}: expected a surface field")); }
        let n = count.iter().product::<usize>(); let mut out = vec![0.; n];
        ck(unsafe { nc_get_vara_double(self.0, v, start.as_ptr(), count.as_ptr(), out.as_mut_ptr()) })?;
        let fill = self.att(v, "_FillValue"); let missing = self.att(v, "missing_value");
        let scale = self.att(v, "scale_factor").unwrap_or(1.); let offset = self.att(v, "add_offset").unwrap_or(0.);
        for x in &mut out {
            *x = if Some(*x) == fill || Some(*x) == missing || x.abs() > 1e30 { f64::NAN } else { *x * scale + offset };
        }
        Ok(Some(out))
    }
    fn valid_time(&self) -> Result<i64> {
        let v = self.var("Times").ok_or("history missing Times")?;
        let sh = self.shape(v)?;
        if sh != [1, 19] { return Err(format!("Times shape {sh:?}, expected one 19-character frame")); }
        let mut b = vec![0u8; 19]; ck(unsafe { nc_get_var_text(self.0, v, b.as_mut_ptr().cast()) })?;
        timestamp(&String::from_utf8_lossy(&b)).ok_or("invalid history Times".into())
    }
}

// CSV quoting, including quoted newlines. IEM's optional METAR column can contain commas.
fn csv(text: &str) -> Result<Vec<Vec<String>>> {
    let mut rows = Vec::new(); let mut row = Vec::new(); let mut val = String::new();
    let mut quoted = false; let mut it = text.chars().peekable();
    while let Some(c) = it.next() {
        match c {
            '"' if quoted && it.peek() == Some(&'"') => { val.push('"'); it.next(); },
            '"' => quoted = !quoted,
            ',' if !quoted => row.push(std::mem::take(&mut val)),
            '\n' if !quoted => { row.push(std::mem::take(&mut val)); rows.push(std::mem::take(&mut row)); },
            '\r' if !quoted => (),
            _ => val.push(c),
        }
    }
    if quoted { return Err("unclosed CSV quote".into()); }
    if !val.is_empty() || !row.is_empty() { row.push(val); rows.push(row); }
    Ok(rows)
}
fn quote(s: &str) -> String { format!("\"{}\"", s.replace('"', "\"\"")) }
fn num(s: &str) -> f64 { s.parse::<f64>().ok().filter(|v| v.is_finite()).unwrap_or(f64::NAN) }
fn show(v: f64) -> String { if v.is_finite() { format!("{v:.8}") } else { String::new() } }
fn timestamp(s: &str) -> Option<i64> {
    let p: Vec<i64> = s.split(|c: char| !c.is_ascii_digit()).filter(|x| !x.is_empty()).map(str::parse).collect::<std::result::Result<_,_>>().ok()?;
    if p.len() < 5 || p.len() > 6 { return None; }
    let (year, month, day, hour, minute) = (p[0], p[1], p[2], p[3], p[4]);
    if !(1..=12).contains(&month) || !(0..24).contains(&hour) || !(0..60).contains(&minute) { return None; }
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let mdays = [31, if leap {29} else {28}, 31,30,31,30,31,31,30,31,30,31];
    if day < 1 || day > mdays[(month-1) as usize] { return None; }
    let second = *p.get(5).unwrap_or(&0); if !(0..60).contains(&second) { return None; }
    let y = year - i64::from(month <= 2); let era = y.div_euclid(400); let yoe = y - era*400;
    let mp = month + if month > 2 {-3} else {9};
    let days = era*146097 + yoe*365 + yoe/4 - yoe/100 + (153*mp+2)/5 + day-1 - 719468;
    Some(days*86400 + hour*3600 + minute*60 + second)
}
// The draft's hazard precedence, but UP, missing values, vicinity and unsupported precipitation
// never become a known dry observation. FZDZ is kept unknown (not freezing rain).
fn weather(s: Option<&str>) -> i8 {
    let Some(s) = s else { return -1 };
    if ["M", "NULL", "NA", "NAN"].contains(&s.trim().to_uppercase().as_str()) { return -1; }
    let mut seen = [false; 5]; let mut unknown = false;
    for tok in s.split_whitespace() {
        let t = tok.trim_start_matches(['+', '-']);
        if t.starts_with("VC") { continue; }
        if t.contains("FZDZ") || t.contains("UP") || t.contains("GR") || t.contains("GS") || t.contains("IC") { unknown = true; continue; }
        if t.contains("FZRA") { seen[3] = true; }
        else if t.contains("RA") || t.contains("DZ") { seen[1] = true; }
        if t.contains("PL") || t.contains("IP") { seen[4] = true; }
        if t.contains("SN") && !t.starts_with("BL") && !t.starts_with("DR") { seen[2] = true; }
    }
    for i in [3, 4, 2, 1] { if seen[i] { return i as i8; } }
    if unknown { -1 } else { 0 }
}
#[derive(Clone)]
struct Obs { station: String, time: i64, lat: f64, lon: f64, t: f64, td: f64, cls: i8, p: f64, ice: f64 }
fn cell<'a>(hdr: &[String], row: &'a [String], name: &str) -> Option<&'a str> {
    hdr.iter().position(|h| h == name).and_then(|i| row.get(i)).map(String::as_str)
}
fn observations(path: &str, valid: i64) -> Result<Vec<Obs>> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let rows = csv(&text)?; let hdr = rows.first().ok_or("empty ASOS CSV")?;
    for name in ["station", "valid", "lat", "lon", "tmpf", "dwpf"] {
        if !hdr.iter().any(|x| x == name) { return Err(format!("ASOS CSV missing {name}")); }
    }
    let mut best: BTreeMap<String, Obs> = BTreeMap::new();
    for r in rows.iter().skip(1) {
        let get = |name| cell(hdr, r, name);
        let Some(t) = get("valid").and_then(timestamp) else { continue };
        if (t-valid).abs() > 600 { continue; }
        let st = get("station").unwrap_or("");
        if st.is_empty() { continue; }
        if best.get(st).is_some_and(|o| ((o.time-valid).abs(),o.time) <= ((t-valid).abs(),t)) { continue; }
        let n = |name| num(get(name).unwrap_or(""));
        let lat=n("lat"); let lon=n("lon");
        if !lat.is_finite() || !lon.is_finite() || lat.abs()>90. || lon.abs()>180. { continue; }
        best.insert(st.into(), Obs{station:st.into(),time:t,lat,lon,t:n("tmpf"),td:n("dwpf"),cls:weather(get("wxcodes")),p:n("p01i")*25.4,ice:n("ice_accretion_1hr")*25.4});
    }
    Ok(best.into_values().collect())
}
fn normlon(x: f64) -> f64 { (x+180.).rem_euclid(360.)-180. }
struct Grid { lat: Vec<f64>, lon: Vec<f64>, bins: HashMap<(i32,i32),Vec<usize>> }
impl Grid {
    fn new(lat: Vec<f64>, lon: Vec<f64>) -> Result<Self> {
        if lat.len()!=lon.len() || lat.is_empty() { return Err("invalid coordinate grid".into()); }
        let lon: Vec<f64> = lon.into_iter().map(normlon).collect(); let mut bins: HashMap<_,Vec<_>> = HashMap::new();
        for i in 0..lat.len() { if lat[i].is_finite() && lon[i].is_finite() { bins.entry(((lat[i]*2.).floor() as i32,(lon[i]*2.).floor() as i32)).or_default().push(i); } }
        Ok(Self{lat,lon,bins})
    }
    fn nearest(&self, la:f64, lo:f64, km:f64) -> Option<usize> {
        let lo=normlon(lo); let dy=km/111.; let dx=(dy/(la.abs()+dy).min(89.99).to_radians().cos()).min(180.);
        let mut best=(f64::INFINITY,0);
        for y in ((la-dy)*2.).floor() as i32..=((la+dy)*2.).floor() as i32 {
            for x in ((lo-dx)*2.).floor() as i32..=((lo+dx)*2.).floor() as i32 {
                if let Some(ids)=self.bins.get(&(y,(x+360).rem_euclid(720)-360)) {
                    for &i in ids {
                        let a=((self.lat[i]-la).to_radians()/2.).sin().powi(2)+la.to_radians().cos()*self.lat[i].to_radians().cos()*(normlon(self.lon[i]-lo).to_radians()/2.).sin().powi(2);
                        let d=2.*6371.*a.clamp(0.,1.).sqrt().asin(); if d<best.0 { best=(d,i); }
                    }
                }
            }
        }
        (best.0<=km).then_some(best.1)
    }
}

// Port of the recovered namwinter.py Bourgouin energy-area diagnostic, with finite
// profile/pressure checks. Temperatures are K, pressures Pa, levels bottom to top.
fn bourgouin(t: &[f64], p: &[f64], t2:f64, ps:f64) -> i8 {
    if t.len()!=p.len() || !t2.is_finite() || !ps.is_finite() { return -1; }
    let mut tt=vec![t2]; let mut pp=vec![ps];
    for (&a,&b) in t.iter().zip(p) {
        if b>=40000. && b<*pp.last().unwrap() && a.is_finite() && b.is_finite() { tt.push(a); pp.push(b); }
    }
    if tt.len()<3 || *pp.last().unwrap()>50000. { return -1; }
    let mut area=vec![0.;1]; let mut seg=0;
    for k in 0..tt.len()-1 {
        let a=tt[k]-273.15; let b=tt[k+1]-273.15; let w=287.04*(pp[k]/pp[k+1]).ln();
        if (a>0.)!=(b>0.) {
            let f=a/(a-b); area[seg]+=0.5*a*f*w; area.push(0.5*b*(1.-f)*w); seg+=1;
        } else { area[seg]+=0.5*(a+b)*w; }
    }
    let at=|i| *area.get(i).unwrap_or(&0.); let warm=tt[0]>273.15;
    if warm && !area.iter().skip(2).any(|&x|x>0.) { return if at(0)>9.4 {1} else {2}; }
    if !warm && at(1)>0. { return if at(1)<2. {2} else if -at(0)>56.+0.66*at(1) {4} else {3}; }
    if warm && at(1)<0. && at(2)>0. {
        let frozen=if at(2)<2. {2} else if -at(1)>56.+0.66*at(2) {4} else {1};
        return if at(0)>9.4 {1} else {frozen};
    }
    2
}
fn dewpoint(q:f64,p:f64) -> f64 {
    if !q.is_finite() || !p.is_finite() || q<=0. || p<=0. {return f64::NAN;}
    let a=(q*p/(0.622+q)/611.2).ln(); (273.15+243.5*a/(17.67-a)-273.15)*1.8+32.
}
const HEADER: &str="station,lead,latitude,longitude,observation_epoch,obs_t2_f,obs_td2_f,obs_type,model_t2_f,model_td2_f,model_type,model_precip_1h_mm,obs_precip_1h_mm,model_fzra_proxy_1h_mm,model_snow_proxy_1h_mm,obs_ice_1h_mm";
fn sample(args: &[String]) -> Result<()> {
    if args.len()!=8 {return Err("usage: rw_namwinter sample HISTORY PREVIOUS|- ASOS.csv VALID_UTC LEAD explicit-sr|bourgouin|stored MAX_KM".into());}
    let valid=timestamp(&args[4]).ok_or("invalid valid time")?; let lead=args[5].parse::<u32>().map_err(|e|e.to_string())?;
    let method=args[6].as_str(); if !["explicit-sr","bourgouin","stored"].contains(&method) {return Err("unknown ptype method".into());}
    let km=num(&args[7]); if !km.is_finite() || km<=0. || km>100. {return Err("max_km must be in (0,100]".into());}
    let nc=Nc::open(&args[1])?; if nc.valid_time()?!=valid {return Err("history Times differs from requested valid time".into());}
    let required=|n:&str|nc.read(n,None)?.ok_or(format!("history missing {n}"));
    let grid=Grid::new(required("XLAT")?,required("XLONG")?)?; let mut fields=HashMap::new();
    for n in ["T2","Q2","PSFC","RAINNC","RAINC","SR","PREC","PTYPE","CRAIN","CSNOW","CFRZR","CICEP"] {
        if let Some(v)=nc.read(n,None)? {
            if v.len()!=grid.lat.len() {return Err(format!("{n}: grid shape mismatch"));} fields.insert(n,v);
        }
    }
    if !fields.contains_key("T2") {return Err("history missing T2".into());}
    let mut previous=HashMap::new();
    if args[2]!="-" {
        let prev=Nc::open(&args[2])?; if prev.valid_time()?!=valid-3600 {return Err("previous history is not one hour earlier".into());}
        for (n, expected) in [("XLAT",&grid.lat),("XLONG",&grid.lon)] {
            let mut actual=prev.read(n,None)?.ok_or(format!("previous history missing {n}"))?;
            if n=="XLONG" {actual.iter_mut().for_each(|x|*x=normlon(*x));}
            if &actual!=expected {return Err("moving/different grids need station receipts with matched hourly accumulations".into());}
        }
        for n in ["RAINNC","RAINC"] {if let Some(v)=prev.read(n,None)? {previous.insert(n,v);}}
    }
    println!("{HEADER}");
    for o in observations(&args[3],valid)? {
        let Some(i)=grid.nearest(o.lat,o.lon,km) else {continue};
        let at=|n:&str|fields.get(n).map(|a|a[i]).unwrap_or(f64::NAN);
        let old=|n:&str|previous.get(n).and_then(|a|a.get(i)).copied().unwrap_or(f64::NAN);
        let raw=at("RAINNC")+at("RAINC")-old("RAINNC")-old("RAINC");
        if raw.is_finite() && raw< -0.001 {return Err("negative hourly precipitation: accumulation reset".into());}
        let p1=if raw.is_finite() {raw.max(0.)} else {f64::NAN};
        let mut c=-1;
        if method=="stored" {
            let v=at("PTYPE"); if v.is_finite() && v.fract()==0. && (0.0..=4.0).contains(&v) {c=v as i8;}
            else if ["CRAIN","CSNOW","CFRZR","CICEP"].iter().all(|n|at(n).is_finite()) {
                c=0; for (n,cls) in [("CRAIN",1),("CSNOW",2),("CICEP",4),("CFRZR",3)] {if at(n)>0.5 {c=cls;}}
            }
        } else if p1.is_finite() && at("PREC").is_finite() {
            if p1<0.25 && at("PREC")<4e-6 {c=0;}
            else if let (Some(theta),Some(p))=(nc.read("T",Some(i))?,nc.read("PRES",Some(i))?) {
                let t:Vec<_>=theta.iter().zip(&p).map(|(&th,&pr)|(th+300.)*(pr/1e5).powf(0.28571)).collect();
                let b=bourgouin(&t,&p,at("T2"),at("PSFC"));
                if method=="bourgouin" {c=b;}
                else if b>=0 && (0.0..=1.0).contains(&at("SR")) {
                    c=if at("SR")>=0.5 {if b==3 || b==4 {4} else {2}} else if at("T2")<273.15 {3} else {1};
                }
            }
        }
        let fz=if c<0 || !p1.is_finite() {f64::NAN} else if c==3 {p1} else {0.};
        let snow=if (0.0..=1.0).contains(&at("SR")) {p1*at("SR")*10.} else {f64::NAN};
        let values=[o.lat,o.lon,o.time as f64,o.t,o.td,o.cls as f64,(at("T2")-273.15)*1.8+32.,dewpoint(at("Q2"),at("PSFC")),c as f64,p1,o.p,fz,snow,o.ice];
        println!("{},{},{}",quote(&o.station),lead,values.map(show).join(","));
    }
    Ok(())
}
fn main() {
    let args:Vec<_>=env::args().skip(1).collect();
    let result=if args.first().map(String::as_str)==Some("sample") {sample(&args)} else {Err("usage: rw_namwinter sample HISTORY PREVIOUS|- ASOS.csv VALID_UTC LEAD METHOD MAX_KM".into())};
    if let Err(e)=result {eprintln!("DEFECT winter data path: {e}");std::process::exit(2);}
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn metar_classes() {
        for (s,c) in [("-SN",2),("FZRA PL",3),("RASN",2),("-PL",4),("RA",1),("VCSH BR",0),("FZFG",0),("FZDZ",-1),("UP",-1),("M",-1),("",0),("BLSN",0)] {assert_eq!(weather(Some(s)),c,"{s}");}
        assert_eq!(weather(None),-1);
    }
    #[test] fn time_and_csv() {
        assert_eq!(timestamp("1970-01-01 00:00"),Some(0));
        assert_eq!(timestamp("2024-02-29T23:59:00Z").unwrap()+60,timestamp("2024-03-01_00:00:00").unwrap());
        assert_eq!(timestamp("2023-02-29 00:00"),None);
        assert_eq!(csv("a,b\r\n\"x,y\",\"line\nnext\"\n").unwrap()[1],vec!["x,y","line\nnext"]);
    }
    #[test] fn profiles_and_nearest() {
        let p=[99000.,90000.,75000.,50000.,40000.];
        assert_eq!(bourgouin(&[270.,268.,265.,255.,240.],&p,270.,100000.),2);
        assert_eq!(bourgouin(&[280.,278.,275.,260.,240.],&p,280.,100000.),1);
        assert_eq!(bourgouin(&[272.,278.,275.,255.,240.],&p,272.,100000.),3);
        assert_eq!(bourgouin(&[260.,260.,276.,270.,240.],&p,260.,100000.),4);
        let g=Grid::new(vec![35.,35.1],vec![179.99,-179.9]).unwrap();
        assert_eq!(g.nearest(35.,-179.99,15.),Some(0)); assert_eq!(g.nearest(36.,-80.,15.),None);
    }
}
