use std::cell::Cell;
use std::sync::OnceLock;
use std::time::Instant;

fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("UNMESH_TIMINGS").is_some_and(|v| !v.is_empty()))
}

thread_local! {
    static LAST: Cell<Option<Instant>> = const { Cell::new(None) };
}

/// Resets the stage clock. With `UNMESH_TIMINGS` set, `lap` prints the time
/// since the previous `start` or `lap` to stderr.
pub fn start() {
    if enabled() {
        LAST.with(|c| c.set(Some(Instant::now())));
    }
}

pub fn lap(stage: &str) {
    if !enabled() {
        return;
    }
    let now = Instant::now();
    LAST.with(|c| {
        if let Some(t) = c.get() {
            eprintln!("unmesh-timing {stage} {:.4}", (now - t).as_secs_f64());
        }
        c.set(Some(now));
    });
}
