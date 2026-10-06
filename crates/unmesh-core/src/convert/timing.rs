use std::cell::Cell;
use std::sync::OnceLock;
use std::time::Instant;

fn switched_on(v: Option<&std::ffi::OsStr>) -> bool {
    v.is_some_and(|v| !v.is_empty() && v != "0")
}

fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| switched_on(std::env::var_os("UNMESH_TIMINGS").as_deref()))
}

thread_local! {
    static LAST: Cell<Option<Instant>> = const { Cell::new(None) };
    static SUB: Cell<Option<Instant>> = const { Cell::new(None) };
}

fn stamp(clock: &'static std::thread::LocalKey<Cell<Option<Instant>>>, stage: &str) {
    let now = Instant::now();
    clock.with(|c| {
        if let Some(t) = c.get() {
            eprintln!("unmesh-timing {stage} {:.4}", (now - t).as_secs_f64());
        }
        c.set(Some(now));
    });
}

/// Resets the stage clock. With `UNMESH_TIMINGS` set (and not `0`), `lap`
/// prints the time since the previous `start` or `lap` to stderr.
pub fn start() {
    if enabled() {
        LAST.with(|c| c.set(Some(Instant::now())));
    }
}

pub fn lap(stage: &str) {
    if enabled() {
        stamp(&LAST, stage);
    }
}

/// Resets the sub-stage clock, which times the steps inside one stage
/// without moving the stage clock.
pub fn sub_start() {
    if enabled() {
        SUB.with(|c| c.set(Some(Instant::now())));
    }
}

pub fn sub_lap(stage: &str) {
    if enabled() {
        stamp(&SUB, stage);
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;

    #[test]
    fn unset_empty_and_zero_mean_off() {
        assert!(!super::switched_on(None));
        assert!(!super::switched_on(Some(OsStr::new(""))));
        assert!(!super::switched_on(Some(OsStr::new("0"))));
        assert!(super::switched_on(Some(OsStr::new("1"))));
    }
}
