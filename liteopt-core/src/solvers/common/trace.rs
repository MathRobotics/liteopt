use std::cell::RefCell;
use std::fmt::Write;

#[derive(Clone, Debug)]
pub struct SolverTraceRecord {
    pub solver: &'static str,
    pub iter: usize,
    pub f: Option<f64>,
    pub cost: Option<f64>,
    pub r_norm: Option<f64>,
    pub grad_norm: Option<f64>,
    pub dx_norm: Option<f64>,
    pub step_size: Option<f64>,
    pub alpha: Option<f64>,
    pub lambda: Option<f64>,
    pub dphi0: Option<f64>,
    pub note: Option<&'static str>,
}

impl SolverTraceRecord {
    fn format_line(&self) -> String {
        let mut line = format!("[{}] iter {:>6}", self.solver, self.iter);
        if let Some(v) = self.f {
            let _ = write!(line, " | f {:>13.6e}", v);
        }
        if let Some(v) = self.cost {
            let _ = write!(line, " | cost {:>13.6e}", v);
        }
        if let Some(v) = self.r_norm {
            let _ = write!(line, " | r {:>13.6e}", v);
        }
        if let Some(v) = self.grad_norm {
            let _ = write!(line, " | grad {:>13.6e}", v);
        }
        if let Some(v) = self.dx_norm {
            let _ = write!(line, " | dx {:>13.6e}", v);
        }
        if let Some(v) = self.step_size {
            let _ = write!(line, " | step {:>+9.3e}", v);
        }
        if let Some(v) = self.alpha {
            let _ = write!(line, " | alpha {:>8.3e}", v);
        }
        if let Some(v) = self.lambda {
            let _ = write!(line, " | lambda {:>9.3e}", v);
        }
        if let Some(v) = self.dphi0 {
            let _ = write!(line, " | dphi0 {:>13.6e}", v);
        }
        if let Some(note) = self.note {
            let _ = write!(line, " | note {note}");
        }
        line
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct TraceRow {
    iter: usize,
    f: Option<f64>,
    cost: Option<f64>,
    r_norm: Option<f64>,
    grad_norm: Option<f64>,
    dx_norm: Option<f64>,
    step_size: Option<f64>,
    alpha: Option<f64>,
    lambda: Option<f64>,
    dphi0: Option<f64>,
    note: Option<&'static str>,
}

impl TraceRow {
    pub(crate) fn iter(iter: usize) -> Self {
        Self {
            iter,
            f: None,
            cost: None,
            r_norm: None,
            grad_norm: None,
            dx_norm: None,
            step_size: None,
            alpha: None,
            lambda: None,
            dphi0: None,
            note: None,
        }
    }

    pub(crate) fn f(mut self, f: f64) -> Self {
        self.f = Some(f);
        self
    }

    pub(crate) fn cost(mut self, cost: f64) -> Self {
        self.cost = Some(cost);
        self
    }

    pub(crate) fn r_norm(mut self, r_norm: f64) -> Self {
        self.r_norm = Some(r_norm);
        self
    }

    pub(crate) fn grad_norm(mut self, grad_norm: f64) -> Self {
        self.grad_norm = Some(grad_norm);
        self
    }

    pub(crate) fn dx_norm(mut self, dx_norm: f64) -> Self {
        self.dx_norm = Some(dx_norm);
        self
    }

    pub(crate) fn step_size(mut self, step_size: f64) -> Self {
        self.step_size = Some(step_size);
        self
    }

    pub(crate) fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = Some(alpha);
        self
    }

    pub(crate) fn lambda(mut self, lambda: f64) -> Self {
        self.lambda = Some(lambda);
        self
    }

    pub(crate) fn dphi0(mut self, dphi0: f64) -> Self {
        self.dphi0 = Some(dphi0);
        self
    }

    pub(crate) fn note(mut self, note: &'static str) -> Self {
        self.note = Some(note);
        self
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SolverTracer {
    verbose: bool,
    solver: &'static str,
    history: Option<RefCell<Vec<SolverTraceRecord>>>,
}

impl SolverTracer {
    pub(crate) fn gd(verbose: bool) -> Self {
        Self {
            verbose,
            solver: "gd",
            history: None,
        }
    }

    pub(crate) fn gd_with_history(verbose: bool) -> Self {
        Self {
            verbose,
            solver: "gd",
            history: Some(RefCell::new(Vec::new())),
        }
    }

    pub(crate) fn gn(verbose: bool) -> Self {
        Self {
            verbose,
            solver: "gn",
            history: None,
        }
    }

    pub(crate) fn gn_with_history(verbose: bool) -> Self {
        Self {
            verbose,
            solver: "gn",
            history: Some(RefCell::new(Vec::new())),
        }
    }

    pub(crate) fn lm(verbose: bool) -> Self {
        Self {
            verbose,
            solver: "lm",
            history: None,
        }
    }

    pub(crate) fn lm_with_history(verbose: bool) -> Self {
        Self {
            verbose,
            solver: "lm",
            history: Some(RefCell::new(Vec::new())),
        }
    }

    pub(crate) fn emit(&self, row: TraceRow) {
        let record = SolverTraceRecord {
            solver: self.solver,
            iter: row.iter,
            f: row.f,
            cost: row.cost,
            r_norm: row.r_norm,
            grad_norm: row.grad_norm,
            dx_norm: row.dx_norm,
            step_size: row.step_size,
            alpha: row.alpha,
            lambda: row.lambda,
            dphi0: row.dphi0,
            note: row.note,
        };

        if let Some(history) = &self.history {
            history.borrow_mut().push(record.clone());
        }

        if self.verbose {
            println!("{}", record.format_line());
        }
    }

    pub(crate) fn into_history(self) -> Vec<SolverTraceRecord> {
        self.history
            .map(|history| history.into_inner())
            .unwrap_or_default()
    }
}
