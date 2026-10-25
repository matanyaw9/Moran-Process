use std::io::{BufRead, BufReader, Stdin};

pub struct Parser {
    buf: Option<BufReader<Stdin>>,
    line: Vec<u8>,
    pos: usize,
}

pub struct Line {
    pub r: f32,
    pub adjs: [u64; 63],
    pub report: Option<Box<[u64]>>,
    pub threads: Option<usize>,
}

pub enum Error {
    Io(std::io::Error),
    Parse(usize),
}

impl Parser {
    pub fn new() -> Parser {
        Parser {
            buf: Some(BufReader::new(std::io::stdin())),
            line: vec![],
            pos: 0,
        }
    }

    pub fn next(&mut self) -> Result<Option<Line>, Error> {
        let Some(buf) = &mut self.buf else {
            return Ok(None);
        };
        self.line.clear();
        self.pos = 0;
        buf.read_until(b'\n', &mut self.line).map_err(Error::Io)?;
        match self.line[..] {
            [.., b'\n'] => _ = self.line.pop(),
            [.., _] => self.buf = None,
            [] => {
                self.buf = None;
                return Ok(None);
            }
        }

        self.take(b"{")?;
        let mut r = None;
        let mut adjs = None;
        let mut report = None;
        let mut threads = None;
        let (r, adjs) = loop {
            if r.is_none() && self.take(b"\"fitness\"").is_ok() {
                self.take(b":")?;
                r = Some(self.float()?);
            } else if adjs.is_none() && self.take(b"\"graph\"").is_ok() {
                self.take(b":")?;
                self.take(b"[")?;
                let mut ad = [0; _];
                ad[0] = self.int_arr()?;
                self.take(b",")?;
                for (i, a) in ad[1..].iter_mut().enumerate() {
                    *a = self.int_arr()?;
                    if self.take(b"]").is_ok() {
                        break;
                    }
                    if i == 61 {
                        return Err(Error::Parse(self.pos));
                    }
                    self.take(b",")?;
                }
                adjs = Some(ad);
            } else if report.is_none() && self.take(b"\"report\"").is_ok() {
                self.take(b":")?;
                self.take(b"[")?;
                let mut rprt = vec![];
                loop {
                    rprt.push(self.int_arr()?);
                    if self.take(b"]").is_ok() {
                        break;
                    }
                    self.take(b",")?;
                }
                report = Some(rprt.into_boxed_slice());
            } else if threads.is_none() && self.take(b"\"threads\"").is_ok() {
                self.take(b":")?;
                threads = Some(self.int()?);
            } else {
                return Err(Error::Parse(self.pos));
            }
            if self.take(b",").is_ok() {
                continue;
            }
            let (Some(r), Some(adjs)) = (r, adjs) else {
                return Err(Error::Parse(self.pos));
            };
            self.take(b"}")?;
            self.skip_ws();
            if self.pos < self.line.len() {
                return Err(Error::Parse(self.pos));
            }
            break (r, adjs);
        };
        Ok(Some(Line {
            r,
            adjs,
            report,
            threads,
        }))
    }

    fn int_arr(&mut self) -> Result<u64, Error> {
        let mut res = 0;
        self.take(b"[")?;
        loop {
            res |= 1 << 63.min(self.int()?);
            if self.take(b"]").is_ok() {
                break Ok(res);
            }
            self.take(b",")?;
        }
    }

    fn take(&mut self, needle: &[u8]) -> Result<(), Error> {
        self.skip_ws();
        if self.line[self.pos..].starts_with(needle) {
            self.pos += needle.len();
            Ok(())
        } else {
            Err(Error::Parse(self.pos))
        }
    }

    fn int(&mut self) -> Result<usize, Error> {
        self.skip_ws();
        match self.line.get(self.pos) {
            Some(b'0') => {
                self.pos += 1;
                Ok(0)
            }
            Some(&n @ b'1'..=b'9') => {
                let mut res = (n - b'0') as usize;
                self.pos += 1;
                while let Some(&n @ b'0'..=b'9') = self.line.get(self.pos) {
                    res = res.saturating_mul(10).saturating_add((n - b'0') as _);
                    self.pos += 1;
                }
                Ok(res)
            }
            _ => Err(Error::Parse(self.pos)),
        }
    }

    fn float(&mut self) -> Result<f32, Error> {
        self.skip_ws();
        let start = self.pos;
        self.int()?;
        if let Some(b'.') = self.line.get(self.pos) {
            self.pos += 1;
            let Some(b'0'..=b'9') = self.line.get(self.pos) else {
                return Err(Error::Parse(self.pos));
            };
            self.pos += 1;
            while let Some(b'0'..=b'9') = self.line.get(self.pos) {
                self.pos += 1;
            }
        }
        if let Some(b'e' | b'E') = self.line.get(self.pos) {
            self.pos += 1;
            if let Some(b'+' | b'-') = self.line.get(self.pos) {
                self.pos += 1;
            }
            let Some(b'0'..=b'9') = self.line.get(self.pos) else {
                return Err(Error::Parse(self.pos));
            };
            self.pos += 1;
            while let Some(b'0'..=b'9') = self.line.get(self.pos) {
                self.pos += 1;
            }
        }
        Ok(str::from_utf8(&self.line[start..self.pos])
            .expect("bytes should be valid UTF-8")
            .parse::<f32>()
            .expect("substring should parse as f32"))
    }

    fn skip_ws(&mut self) {
        while let Some(b'\t' | b'\r' | b' ') = self.line.get(self.pos) {
            self.pos += 1;
        }
    }
}
