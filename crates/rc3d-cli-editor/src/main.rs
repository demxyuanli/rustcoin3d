fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,rc3d_cli_editor=info"),
    )
    .init();
    println!("rc3d CLI Editor starting...");
}
