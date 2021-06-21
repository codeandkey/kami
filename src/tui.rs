use crate::constants;
use crate::position::Position;
use crate::searcher::SearchStatus;

use std::io::stdout;
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};
use std::thread::{spawn, JoinHandle};
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event as CEvent, KeyCode},
    execute,
    terminal::{enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};

use tui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    symbols,
    text::Span,
    widgets::{
        Axis, Block, Borders, Cell, Chart, Dataset, Paragraph, Row, Table, Gauge,
        Wrap,
    },
    Terminal,
};

enum Event<I> {
    Input(I),
    Tick,
}

const TUI_TICK_RATE: u64 = 100; // 10 FPS

pub struct Tui {
    status: Arc<Mutex<SearchStatus>>,
    score_buf: Arc<Mutex<Vec<f64>>>,
    log_buf: Arc<Mutex<Vec<String>>>,
    current_pos: Arc<Mutex<Position>>,
    stop_flag: Arc<Mutex<bool>>,
    handle: Option<JoinHandle<()>>,
    exit_flag: Arc<Mutex<bool>>,
    paused: Arc<Mutex<bool>>,
}

impl Tui {
    /// Creates a new TUI instance.
    pub fn new() -> Self {
        Tui {
            status: Arc::new(Mutex::new(SearchStatus::Done)),
            log_buf: Arc::new(Mutex::new(Vec::new())),
            current_pos: Arc::new(Mutex::new(Position::new())),
            score_buf: Arc::new(Mutex::new(Vec::new())),
            stop_flag: Arc::new(Mutex::new(false)),
            exit_flag: Arc::new(Mutex::new(false)),
            paused: Arc::new(Mutex::new(false)),
            handle: None,
        }
    }

    /// Starts the TUI thread and begins displaying status data.
    pub fn start(&mut self) {
        let thr_status = self.status.clone();
        let thr_log_buf = self.log_buf.clone();
        let thr_score_buf = self.score_buf.clone();
        let thr_stop_flag = self.stop_flag.clone();
        let thr_paused = self.paused.clone();
        let thr_exit_flag = self.exit_flag.clone();
        let thr_current_pos = self.current_pos.clone();

        self.handle = Some(spawn(move || {
            // Initialize TUI
            enable_raw_mode().expect("failed setting raw mode");

            let mut stdout = stdout();
            execute!(stdout, EnterAlternateScreen).expect("failed entering alternate screen");

            // Start input thread
            let (inp_tx, inp_rx) = channel();
            let inp_thread_stopflag = Arc::new(Mutex::new(false));
            let inp_thr_thread_stopflag = inp_thread_stopflag.clone();

            let input_thread = spawn(move || {
                let tick_rate = Duration::from_millis(TUI_TICK_RATE);
                let mut last_tick = Instant::now();
                while !*inp_thr_thread_stopflag.lock().unwrap() {
                    let timeout = tick_rate
                        .checked_sub(last_tick.elapsed())
                        .unwrap_or_else(|| Duration::from_secs(0));

                    if event::poll(timeout).unwrap() {
                        if let CEvent::Key(key) = event::read().unwrap() {
                            inp_tx.send(Event::Input(key)).unwrap();
                        }
                    }

                    if last_tick.elapsed() >= tick_rate {
                        inp_tx.send(Event::Tick).unwrap();
                        last_tick = Instant::now();
                    }
                }
            });

            let backend = CrosstermBackend::new(stdout);
            let mut terminal = Terminal::new(backend).expect("failed initializing terminal");

            let mut nps_history: Vec<f64> = Vec::new();

            while !*thr_stop_flag.lock().unwrap() {
                if !*thr_paused.lock().unwrap() {
                    terminal
                        .draw(|f| {
                            // Compute layout areas
                            let rects = Layout::default()
                                .direction(Direction::Horizontal)
                                .constraints([Constraint::Length(54), Constraint::Percentage(100)])
                                .split(f.size());

                            let tree_rect = rects[0];

                            let rects = Layout::default()
                                .direction(Direction::Horizontal)
                                .constraints([
                                    Constraint::Percentage(50),
                                    Constraint::Percentage(50),
                                ])
                                .split(rects[1]);

                            let (center_rect, charts_rect) = (rects[0], rects[1]);

                            let rects = Layout::default()
                                .direction(Direction::Vertical)
                                .constraints([
                                    Constraint::Ratio(1, 3),
                                    Constraint::Ratio(1, 3),
                                    Constraint::Ratio(1, 3),
                                ])
                                .split(charts_rect);

                            let nps_rect = rects[0];
                            let mcts_rect = rects[1];
                            let log_rect = rects[2];

                            let rects = Layout::default()
                                .direction(Direction::Vertical)
                                .constraints([
                                    Constraint::Length(5),
                                    Constraint::Length(num_cpus::get() as u16 + 4),
                                    Constraint::Percentage(100),
                                    Constraint::Length(3),
                                ])
                                .split(center_rect);

                            let summary_rect = rects[0];
                            let workers_rect = rects[1];
                            let board_rect = rects[2];

                            // Render tree status, if there is one.
                            let search_status = thr_status.lock().unwrap().clone();

                            let header_cells = ["Action", "N_normalized", "N_actual", "P", "Q"]
                                .iter()
                                .map(|h| Cell::from(*h).style(Style::default().fg(Color::Blue)));

                            let header = Row::new(header_cells).height(1).bottom_margin(1);

                            let mut total_nn = 0.0;

                            let nodes = match &search_status {
                                SearchStatus::Searching(s) => match &s.tree {
                                    Some(t) => {
                                        total_nn = t.total_nn;
                                        t.nodes.clone()
                                    }
                                    None => Vec::new(),
                                },
                                SearchStatus::Stopping => Vec::new(),
                                SearchStatus::Done => Vec::new(),
                            };

                            let rows = nodes.iter().map(|nd| {
                                Row::new([
                                    nd.action.clone(),
                                    format!("{:.1}%", nd.nn * 100.0 / total_nn,),
                                    format!("{}", nd.n),
                                    format!("{:.2}%", nd.p_pct * 100.0),
                                    format!("{:+.2}", nd.q),
                                ])
                            });

                            let t = Table::new(rows)
                                .header(header)
                                .block(Block::default().borders(Borders::ALL).title("Tree"))
                                .widths(&[
                                    Constraint::Length(10),
                                    Constraint::Length(14),
                                    Constraint::Length(10),
                                    Constraint::Length(8),
                                    Constraint::Length(8),
                                ]);

                            f.render_widget(t, tree_rect);

                            // Render log

                            let log_lines = thr_log_buf.lock().unwrap().clone();
                            let log_lines = log_lines[log_lines
                                .len()
                                .checked_sub((log_rect.height - 2) as usize)
                                .unwrap_or(0)..]
                                .join("\n");

                            let lg = Paragraph::new(log_lines)
                                .block(Block::default().title("Log").borders(Borders::ALL))
                                .style(Style::default().fg(Color::White).bg(Color::Black))
                                .wrap(Wrap { trim: true });

                            f.render_widget(lg, log_rect);

                            // Render board status
                            let board_lines = thr_current_pos.lock().unwrap().to_string_pretty();

                            // Center vertically
                            let board_lines =
                                (0..(board_rect.height / 2).checked_sub(4).unwrap_or(0))
                                    .map(|_| "\n".to_string())
                                    .collect::<Vec<String>>()
                                    .join("")
                                    + &board_lines;

                            let board_widget = Paragraph::new(board_lines)
                                .block(Block::default().title("Board"))
                                .style(Style::default().fg(Color::White).bg(Color::Black))
                                .alignment(Alignment::Center)
                                .wrap(Wrap { trim: true });

                            f.render_widget(board_widget, board_rect);

                            // Render perf summary
                            let summary_lines = format!(
                                "Searcher: {}\nAvg. noderate: {}\nAvg. batchrate: {}\n",
                                match &search_status {
                                    SearchStatus::Searching(stat) =>
                                        format!("searching {}", stat.rootfen),
                                    SearchStatus::Stopping => "stopping".to_string(),
                                    SearchStatus::Done => "stopped".to_string(),
                                },
                                match &search_status {
                                    SearchStatus::Searching(stat) => {
                                        format!(
                                            "{:.1}",
                                            (stat
                                                .workers
                                                .iter()
                                                .map(|w| w.total_nodes as f64)
                                                .sum::<f64>()
                                                / stat.elapsed_ms as f64)
                                                * 1000.0
                                        )
                                    }
                                    _ => "N/A".to_string(),
                                },
                                match &search_status {
                                    SearchStatus::Searching(stat) => {
                                        format!(
                                            "{:.1}",
                                            (stat
                                                .workers
                                                .iter()
                                                .map(|w| w.batch_sizes.len() as f64)
                                                .sum::<f64>()
                                                / stat.elapsed_ms as f64)
                                                * 1000.0
                                        )
                                    }
                                    _ => "N/A".to_string(),
                                }
                            );

                            let summary_widget = Paragraph::new(summary_lines)
                                .block(Block::default().title("Performance").borders(Borders::ALL))
                                .style(Style::default().fg(Color::White).bg(Color::Black))
                                .wrap(Wrap { trim: true });

                            f.render_widget(summary_widget, summary_rect);

                            // Render search perf stats
                            let perf_header_cells = ["ID", "status", "nodes", "batches"]
                                .iter()
                                .map(|h| Cell::from(*h).style(Style::default().fg(Color::Red)));

                            let perf_header =
                                Row::new(perf_header_cells).height(1).bottom_margin(1);

                            let workers = match &search_status {
                                SearchStatus::Searching(stat) => stat.workers.clone(),
                                _ => Vec::new(),
                            };

                            let perf_rows = workers.iter().enumerate().map(|(id, w)| {
                                Row::new([
                                    id.to_string(),
                                    w.state.clone(),
                                    w.total_nodes.to_string(),
                                    w.batch_sizes.len().to_string(),
                                ])
                            });

                            let t = Table::new(perf_rows)
                                .header(perf_header)
                                .block(Block::default().borders(Borders::ALL).title("Workers"))
                                .widths(&[
                                    Constraint::Length(4),
                                    Constraint::Length(16),
                                    Constraint::Length(8),
                                    Constraint::Length(8),
                                ]);

                            f.render_widget(t, workers_rect);

                            // Render search progress
                            let mut prog_rect = board_rect.clone();

                            prog_rect.y += prog_rect.height - 2;
                            prog_rect.height = 2;

                            let prog_total_nodes = match &search_status {
                                SearchStatus::Searching(stat) => stat.total_nodes,
                                _ => 0,
                            };

                            let label = format!("{}/{}", prog_total_nodes, constants::SEARCH_MAXNODES);

                            let prog_widget = Gauge::default()
                                .block(Block::default().title("Progress"))
                                .gauge_style(
                                    Style::default()
                                        .fg(Color::Green)
                                        .bg(Color::Black)
                                        .add_modifier(Modifier::BOLD),
                                )
                                .percent(((prog_total_nodes as f32 * 100.0 / constants::SEARCH_MAXNODES as f32) as u16).min(100))
                                .label(label)
                                .use_unicode(true);
                            
                            f.render_widget(prog_widget, prog_rect);

                            // Render search score
                            let mut mcts_score_data = thr_score_buf
                                .lock()
                                .unwrap()
                                .iter()
                                .enumerate()
                                .map(|(a, b)| (a as f64, *b))
                                .collect::<Vec<(f64, f64)>>();

                            let last_ply = mcts_score_data.len() as i32 - 1;

                            if mcts_score_data.len() > 0 {
                                mcts_score_data =
                                    Tui::interpolate(mcts_score_data, mcts_rect.width as usize);
                            }

                            let score_dataset = Dataset::default()
                                .name("MCTS")
                                .marker(symbols::Marker::Braille)
                                .style(Style::default().fg(Color::Cyan))
                                .data(&mcts_score_data);

                            let baseline_data: Vec<(f64, f64)> = (0..mcts_rect.width)
                                .map(|x| {
                                    (x as f64 / mcts_rect.width as f64 * last_ply as f64, 0.0f64)
                                })
                                .collect();

                            let baseline_dataset = Dataset::default()
                                .name("baseline")
                                .marker(symbols::Marker::Braille)
                                .style(Style::default().fg(Color::Gray))
                                .data(&baseline_data);

                            let sc = Chart::new(vec![score_dataset, baseline_dataset])
                                .block(
                                    Block::default().title(Span::styled(
                                        "Value history",
                                        Style::default()
                                            .fg(Color::Cyan)
                                            .add_modifier(Modifier::BOLD),
                                    )),
                                )
                                .x_axis(
                                    Axis::default()
                                        .title("ply")
                                        .style(Style::default().fg(Color::Gray))
                                        .bounds([0.0, last_ply as f64])
                                        .labels(
                                            [0.0, (last_ply / 2) as f32, last_ply as f32]
                                                .iter()
                                                .cloned()
                                                .map(|x| Span::from(x.to_string()))
                                                .collect(),
                                        ),
                                )
                                .y_axis(
                                    Axis::default()
                                        .title("value")
                                        .style(Style::default().fg(Color::Gray))
                                        .bounds([-1.0, 1.0])
                                        .labels(
                                            ["-1.0", "0.0", "1.0"]
                                                .iter()
                                                .cloned()
                                                .map(Span::from)
                                                .collect(),
                                        ),
                                );

                            f.render_widget(sc, mcts_rect);

                            // Render nps history

                            if let SearchStatus::Searching(stat) = &search_status {
                                nps_history.push(stat.nps as f64);
                            }

                            let disp_frames = nps_history
                                .len()
                                .checked_sub(nps_rect.width as usize)
                                .unwrap_or(0);

                            if disp_frames > 0 {
                                nps_history.drain(0..disp_frames);
                            }

                            let nps_data: Vec<(f64, f64)> = nps_history
                                .iter()
                                .cloned()
                                .enumerate()
                                .map(|(x, y)| (x as f64, y as f64))
                                .collect();

                            let nps_data_len = nps_data.len();

                            let nps_dataset = Dataset::default()
                                .name("NPS")
                                .marker(symbols::Marker::Braille)
                                .style(Style::default().fg(Color::Blue))
                                .data(&nps_data);

                            let mut nps_min = f64::MAX;
                            let mut nps_max = f64::MIN;

                            for (_, y) in &nps_data {
                                if *y < nps_min {
                                    nps_min = *y;
                                }

                                if *y > nps_max {
                                    nps_max = *y;
                                }
                            }

                            let nps_backtime =
                                constants::SEARCH_STATUS_RATE * nps_rect.width as u64;

                            let nps_chart = Chart::new(vec![nps_dataset])
                                .block(
                                    Block::default().title(Span::styled(
                                        "NPS history",
                                        Style::default()
                                            .fg(Color::Cyan)
                                            .add_modifier(Modifier::BOLD),
                                    )),
                                )
                                .x_axis(
                                    Axis::default()
                                        .style(Style::default().fg(Color::Gray))
                                        .bounds([0.0, nps_data_len as f64])
                                        .labels(
                                            [format!("-{} ms", nps_backtime), "now".to_string()]
                                                .iter()
                                                .cloned()
                                                .map(Span::from)
                                                .collect(),
                                        ),
                                )
                                .y_axis(
                                    Axis::default()
                                        .title("nodes/s")
                                        .style(Style::default().fg(Color::Gray))
                                        .bounds([nps_min * 0.75, nps_max * 1.25])
                                        .labels(
                                            [
                                                nps_min * 0.75,
                                                nps_min + 0.25 * (nps_max - nps_min),
                                                nps_min + 0.5 * (nps_max - nps_min),
                                                nps_min + 0.75 * (nps_max - nps_min),
                                                nps_max * 1.25,
                                            ]
                                            .iter()
                                            .cloned()
                                            .map(|x| Span::from(format!("{}", x as usize)))
                                            .collect(),
                                        ),
                                );

                            f.render_widget(nps_chart, nps_rect);
                        })
                        .expect("terminal draw failed");
                }

                // Process user input
                match inp_rx.recv().expect("input rx failed") {
                    Event::Input(event) => match event.code {
                        KeyCode::Char('q') | KeyCode::Esc => {
                            thr_log_buf
                                .lock()
                                .unwrap()
                                .push("Received exit request.".to_string());
                            *thr_exit_flag.lock().unwrap() = true;
                        }
                        KeyCode::Char('p') => {
                            let mut lock = thr_paused.lock().unwrap();
                            *lock = !*lock;

                            if *lock {
                                thr_log_buf
                                    .lock()
                                    .unwrap()
                                    .push("Paused display".to_string());
                            } else {
                                thr_log_buf
                                    .lock()
                                    .unwrap()
                                    .push("Unpaused display".to_string());
                            }
                        }
                        _ => (),
                    },
                    Event::Tick => (),
                }
            }

            *inp_thread_stopflag.lock().unwrap() = true;
            input_thread.join().expect("failed joining input thread");

            execute!(terminal.backend_mut(), LeaveAlternateScreen)
                .expect("failed leaving alternate screen");

            terminal.show_cursor().expect("failed showing cursor");
        }));
    }

    /// Interpolates a set of points to create a line graph.
    fn interpolate(points: Vec<(f64, f64)>, width: usize) -> Vec<(f64, f64)> {
        if points.len() == 1 {
            (0..width)
                .map(|i| (i as f64 / width as f64, points[0].1))
                .collect()
        } else {
            (0..width)
                .map(|i| {
                    let x = (i * (points.len() - 1)) as f64 / width as f64;
                    let last = points[x.floor() as usize];
                    let next = points[x.ceil() as usize];

                    if x == x.floor() {
                        (x, last.1)
                    } else {
                        let last_weight = x.ceil() - x;
                        let next_weight = x - x.floor();

                        (x, next.1 * next_weight + last.1 * last_weight)
                    }
                })
                .collect()
        }
    }

    /// Stops the TUI thread and consumes this object.
    pub fn stop(self) {
        assert!(self.handle.is_some(), "TUI not running?");

        *self.stop_flag.lock().unwrap() = true;
        self.handle
            .unwrap()
            .join()
            .expect("failed joining TUI thread");
    }

    /// Adds a message to the TUI log.
    pub fn log(&self, msg: impl ToString) {
        self.log_buf.lock().unwrap().push(msg.to_string());
    }

    /// Adds a search status to the TUI search status buffer.
    pub fn push_status(&self, status: SearchStatus) {
        *self.status.lock().unwrap() = status;
    }

    /// Adds a score to the TUI score buffer.
    pub fn push_score(&self, score: f64) {
        self.score_buf.lock().unwrap().push(score);
    }

    /// Sets the current position.
    pub fn set_position(&self, pos: Position) {
        *self.current_pos.lock().unwrap() = pos;
    }

    /// Resets all per-game data, should be called between games.
    pub fn reset_game(&self) {
        self.score_buf.lock().unwrap().clear();
    }

    /// Returns true if the user has requested to quit.
    pub fn exit_requested(&self) -> bool {
        *self.exit_flag.lock().unwrap()
    }
}
