//! Terminal UI for command entry (ratatui). Runs on a dedicated thread alongside the GPU window.

use std::collections::VecDeque;
use std::io::{stdout, IsTerminal};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use crossterm::{execute, ExecutableCommand};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::{Frame, Terminal};
use winit::event_loop::EventLoopProxy;

use crate::cli::{submit, CliSubmitResult};
use crate::session::EditorSession;

#[derive(Clone, Debug)]
pub enum AppEvent {
    Quit,
}

pub fn run_tui_thread(
    session: Arc<EditorSession>,
    gui_proxy: EventLoopProxy<AppEvent>,
    gui_alive: Arc<AtomicBool>,
) {
    let mut out = stdout();
    if !out.is_terminal() {
        log::warn!("stdout is not a TTY; terminal CLI disabled (launch from a console for TUI CLI)");
        return;
    }

    if enable_raw_mode().is_err() {
        log::warn!("terminal raw mode failed; terminal CLI disabled");
        return;
    }

    let cleanup = || {
        let _ = disable_raw_mode();
        let _ = stdout().execute(crossterm::terminal::LeaveAlternateScreen);
    };

    if execute!(out, crossterm::terminal::EnterAlternateScreen).is_err() {
        let _ = disable_raw_mode();
        log::warn!("alternate screen unavailable; terminal CLI disabled");
        return;
    }

    drop(out);
    let backend = CrosstermBackend::new(stdout());
    let mut terminal = match Terminal::new(backend) {
        Ok(t) => t,
        Err(e) => {
            cleanup();
            log::warn!("ratatui init failed: {e}");
            return;
        }
    };

    let mut scrollback: VecDeque<String> = VecDeque::with_capacity(400);
    let mut history: Vec<String> = Vec::new();
    let mut hist_nav: Option<usize> = None;
    let mut input_buf: String = String::new();
    let mut cursor_col: usize = 0;

    scrollback_push(
        &mut scrollback,
        "rc3d CLI terminal — Tab completes prefixes, ArrowUp/Down history, Enter runs, Ctrl+C quits.".into(),
    );
    scrollback_push(
        &mut scrollback,
        "Hints: scene load <path> | camera orbit dx dy | help".into(),
    );

    loop {
        if !gui_alive.load(Ordering::Relaxed) {
            break;
        }

        if terminal
            .draw(|f| draw_shell(f, &scrollback, &input_buf, cursor_col))
            .is_err()
        {
            break;
        }

        if !event::poll(Duration::from_millis(32)).unwrap_or(false) {
            continue;
        }

        let Ok(Event::Key(key)) = event::read() else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }

        match key.code {
            KeyCode::Char('c') if key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) => {
                scrollback_push(&mut scrollback, "^C".into());
                let _ = gui_proxy.send_event(AppEvent::Quit);
                break;
            }
            KeyCode::Enter => {
                let line = input_buf.trim().to_string();
                hist_nav = None;
                if line.is_empty() {
                    scrollback_push(&mut scrollback, ">".into());
                    continue;
                }
                history.push(line.clone());
                scrollback_push(&mut scrollback, format!("> {line}"));

                let outcome = match session.state.try_write() {
                    Ok(mut w) => submit(&line, &mut w),
                    Err(_) => {
                        scrollback_push(&mut scrollback, "Engine busy — retry.".into());
                        continue;
                    }
                };

                match outcome {
                    CliSubmitResult::Continue { lines } => {
                        for l in lines {
                            scrollback_push(&mut scrollback, l);
                        }
                    }
                    CliSubmitResult::Quit { lines } => {
                        for l in lines {
                            scrollback_push(&mut scrollback, l);
                        }
                        let _ = gui_proxy.send_event(AppEvent::Quit);
                        break;
                    }
                }

                input_buf.clear();
                cursor_col = 0;
            }
            KeyCode::Char('\t') => {
                apply_tab_completion(&mut input_buf, &mut cursor_col);
            }
            KeyCode::Backspace => {
                if cursor_col > 0 {
                    cursor_col -= 1;
                    input_buf.remove(cursor_col);
                }
                hist_nav = None;
            }
            KeyCode::Delete => {
                if cursor_col < input_buf.len() {
                    input_buf.remove(cursor_col);
                }
                hist_nav = None;
            }
            KeyCode::Left => {
                cursor_col = cursor_col.saturating_sub(1);
                hist_nav = None;
            }
            KeyCode::Right => {
                if cursor_col < input_buf.len() {
                    cursor_col += 1;
                }
                hist_nav = None;
            }
            KeyCode::Home => {
                cursor_col = 0;
                hist_nav = None;
            }
            KeyCode::End => {
                cursor_col = input_buf.len();
                hist_nav = None;
            }
            KeyCode::Up => {
                if history.is_empty() {
                    continue;
                }
                let idx = match hist_nav {
                    None => history.len().saturating_sub(1),
                    Some(i) => i.saturating_sub(1),
                };
                hist_nav = Some(idx);
                if let Some(h) = history.get(idx) {
                    input_buf.clone_from(h);
                    cursor_col = input_buf.len();
                }
            }
            KeyCode::Down => {
                let Some(i) = hist_nav else { continue };
                if i + 1 < history.len() {
                    let next = i + 1;
                    hist_nav = Some(next);
                    input_buf.clone_from(&history[next]);
                } else {
                    hist_nav = None;
                    input_buf.clear();
                }
                cursor_col = input_buf.len();
            }
            KeyCode::Char(c) => {
                input_buf.insert(cursor_col, c);
                cursor_col += 1;
                hist_nav = None;
            }
            _ => {}
        }
    }

    drop(terminal);
    cleanup();
}

fn scrollback_push(buf: &mut VecDeque<String>, line: String) {
    const CAP: usize = 400;
    if buf.len() >= CAP {
        buf.pop_front();
    }
    buf.push_back(line);
}

fn draw_shell(f: &mut Frame, scrollback: &VecDeque<String>, input: &str, cursor_col: usize) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(6), Constraint::Length(3)])
        .split(f.area());

    let log_block = Block::default()
        .title(" command log ")
        .title_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));

    let inner_h = chunks[0].height.saturating_sub(2) as usize;
    let text: String = if scrollback.is_empty() {
        String::new()
    } else {
        let skip = scrollback.len().saturating_sub(inner_h.max(1));
        scrollback
            .iter()
            .skip(skip)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    };

    let log = Paragraph::new(text)
        .block(log_block)
        .style(Style::default().fg(Color::Gray))
        .wrap(Wrap { trim: true });
    f.render_widget(log, chunks[0]);

    let safe_col = cursor_col.min(input.len());
    let (before, after) = input.split_at(safe_col);
    let cursor_cell = if after.is_empty() {
        "_".to_string()
    } else {
        after.chars().next().map(|c| c.to_string()).unwrap_or_else(|| "_".into())
    };
    let after_rest: String = after.chars().skip(1).collect();

    let input_line = format!("rc3d> {before}{cursor_cell}{after_rest}");
    let input_block = Block::default()
        .title(" input ")
        .title_style(Style::default().fg(Color::Green).add_modifier(Modifier::BOLD))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));

    let input_para = Paragraph::new(input_line)
        .block(input_block)
        .style(Style::default().fg(Color::White))
        .wrap(Wrap { trim: false });
    f.render_widget(input_para, chunks[1]);
}

fn apply_tab_completion(buf: &mut String, cursor: &mut usize) {
    static PREFIXES: &[&str] = &[
        "camera ",
        "display ",
        "help",
        "log ",
        "prop ",
        "quit",
        "scene ",
        "select ",
        "test ",
    ];
    let line = buf.as_str();
    let prefix_len = line.len().min(*cursor);
    let head = &line[..prefix_len];
    let token_start = head
        .char_indices()
        .rev()
        .find(|(_, c)| c.is_whitespace())
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(0);
    let partial = &line[token_start..prefix_len];
    if partial.is_empty() {
        return;
    }
    let mut matches: Vec<&str> = PREFIXES
        .iter()
        .copied()
        .filter(|p| p.starts_with(partial))
        .collect();
    matches.sort_unstable();
    if let Some(chosen) = matches.first().copied() {
        buf.drain(token_start..prefix_len);
        buf.insert_str(token_start, chosen);
        *cursor = token_start + chosen.len();
    }
}
